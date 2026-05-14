import json
import os
import random
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_DIR = Path(".")
OUTPUT_DIR = Path("output")

STUDENT_FILES = [
    ("A", ["a.json", "A.json"]),
    ("B", ["b.json", "B.json"]),
    ("C", ["c.json", "C.json"]),
]

LB_FILES = ["lb.json", "LB.json"]

SEED = 42
EPOCHS = 2500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# ------------------------------------------------------------
# UTIL
# ------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json_any(base_dir: Path, candidates: list[str]):
    for name in candidates:
        path = base_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(f"None of these files were found: {candidates}")


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


class RankNet(nn.Module):
    def __init__(self, input_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    lb_data, lb_path = load_json_any(DATA_DIR, LB_FILES)

    personal_datasets = []
    for label, candidates in STUDENT_FILES:
        data, path = load_json_any(DATA_DIR, candidates)
        personal_datasets.append((label, data, path))

    print(f"Loaded leaderboard: {lb_path.name}")
    for label, _, path in personal_datasets:
        print(f"Loaded student {label}: {path.name}")

    # --------------------------------------------------------
    # BUILD LOOKUPS
    # --------------------------------------------------------

    # Estimate total student count N for each test from known rank/percentile pairs:
    # percentile = 100 * (1 - rank/N)
    # => N = rank / (1 - percentile/100)
    n_lookup = {}

    max_marks_lookup = {}
    avg_lookup = {}
    topper_lookup = {}

    for _, student_data, _ in personal_datasets:
        for test in student_data:
            name = test.get("testName")
            score = safe_float(test.get("score"), 0.0)
            pct = safe_float(test.get("percentile"), 0.0)
            rank = safe_float(test.get("rank"), 0.0)
            max_marks = safe_float(test.get("maxMarks"), None)

            if name and max_marks and max_marks > 0:
                max_marks_lookup[name] = max_marks

            if name and score > 0 and pct > 0 and rank > 0 and pct < 100:
                N = rank / (1.0 - (pct / 100.0))
                n_lookup.setdefault(name, []).append(N)

    n_lookup = {k: float(np.mean(v)) for k, v in n_lookup.items() if v}

    for item in lb_data:
        name = item.get("testName")
        if not name:
            continue
        avg = safe_float(item.get("avg"), None)
        topper = safe_float(item.get("topper"), None)
        if avg is not None and avg > 0:
            avg_lookup[name] = avg
        if topper is not None and topper > 0:
            topper_lookup[name] = topper

    # --------------------------------------------------------
    # BUILD TRAINING POINTS
    # Target:
    #   y = percentile-like value in [0,1]
    #
    # Features:
    #   x1 = normalized score strength relative to avg/topper
    #   x2 = max marks normalized
    #   x3 = paper difficulty proxy (avg / max_marks)
    # --------------------------------------------------------

    points = []

    # 1) Leaderboard points
    for test in lb_data:
        name = test.get("testName")
        avg = safe_float(test.get("avg"), None)
        topper = safe_float(test.get("topper"), None)
        lb = test.get("leaderboard", [])

        if not name or avg is None or topper is None:
            continue
        if avg <= 0 or topper <= 0 or topper <= avg:
            continue
        if name not in n_lookup:
            continue
        if name not in max_marks_lookup:
            continue
        if not lb:
            continue

        N = n_lookup[name]
        max_marks = max_marks_lookup[name]

        x2 = max_marks / 300.0
        x3 = avg / max_marks if max_marks > 0 else 0.0

        for entry in lb:
            score = safe_float(entry.get("score"), None)
            rank = safe_float(entry.get("rank"), None)

            if score is None or rank is None:
                continue

            x1 = (score - avg) / (topper - avg)
            y = 1.0 - (rank / N)

            if 0.0 < x1 <= 1.0 and 0.0 < y < 1.0:
                points.append((x1, x2, x3, y))

    # 2) Personal student points
    for _, student_data, _ in personal_datasets:
        for test in student_data:
            name = test.get("testName")
            score = safe_float(test.get("score"), 0.0)
            pct = safe_float(test.get("percentile"), 0.0)
            rank = safe_float(test.get("rank"), 0.0)
            max_marks = safe_float(test.get("maxMarks"), None)

            if not name or max_marks is None or max_marks <= 0:
                continue
            if score <= 0 or pct <= 0 or rank <= 0 or pct >= 100:
                continue

            avg = avg_lookup.get(name)
            topper = topper_lookup.get(name)
            N = n_lookup.get(name)

            if avg is None or topper is None or N is None:
                continue
            if topper <= avg:
                continue

            x1 = (score - avg) / (topper - avg)
            x2 = max_marks / 300.0
            x3 = avg / max_marks
            y = 1.0 - (rank / N)

            if 0.0 < x1 <= 1.0 and 0.0 < y < 1.0:
                points.append((x1, x2, x3, y))

    if not points:
        raise RuntimeError("No training points found. Check your input JSON files.")

    print(f"Total training points: {len(points)}")

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ms = np.array([p[1] for p in points], dtype=np.float32)
    ds = np.array([p[2] for p in points], dtype=np.float32)
    ys = np.array([p[3] for p in points], dtype=np.float32)

    print(f"x range: {xs.min():.3f} → {xs.max():.3f}")
    print(f"max_marks range: {(ms.min() * 300):.0f} → {(ms.max() * 300):.0f}")
    print(f"difficulty proxy range: {ds.min():.3f} → {ds.max():.3f}")
    print(f"y range: {(ys.min() * 100):.1f}% → {(ys.max() * 100):.1f}%")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    X = torch.tensor(np.stack([xs, ms, ds], axis=1), dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RankNet(input_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        if (epoch + 1) % 250 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred_all = model(X)
                mae = (pred_all - Y).abs().mean().item()
            avg_loss = total_loss / len(dataset)
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Loss: {avg_loss:.6f} | "
                f"MAE: {mae * 100:.2f} percentile pts"
            )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()

    errors = np.abs(preds - ys) * 100

    worst_idx = int(np.argmax(errors))

    print(f"\nFinal MAE: {errors.mean():.2f} percentile pts")
    print(f"Max error: {errors.max():.2f} percentile pts")
    print(f"Within ±3 pts: {(errors < 3).mean() * 100:.1f}%")
    print(f"Within ±5 pts: {(errors < 5).mean() * 100:.1f}%")
    print(
        "Worst point — "
        f"x: {xs[worst_idx]:.3f}, "
        f"max_marks: {ms[worst_idx] * 300:.0f}, "
        f"difficulty: {ds[worst_idx]:.3f}, "
        f"actual: {ys[worst_idx] * 100:.1f}%, "
        f"predicted: {preds[worst_idx] * 100:.1f}%, "
        f"error: {errors[worst_idx]:.1f} pts"
    )

    # --------------------------------------------------------
    # SAVE PYTORCH MODEL
    # --------------------------------------------------------

    avg_N = float(np.mean(list(n_lookup.values()))) if n_lookup else 0.0

    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {
                "input_dim": 3,
                "n_points": len(points),
                "avg_N": avg_N,
                "max_marks_ref": 300,
                "feature_names": [
                    "score_strength",
                    "max_marks_norm",
                    "difficulty_proxy",
                ],
            },
        },
        OUTPUT_DIR / "ranknet.pt",
    )

    print("\nSaved → output/ranknet.pt")

    # --------------------------------------------------------
    # EXPORT ONNX + EXTERNAL DATA FILE
    # --------------------------------------------------------

    dummy = torch.tensor([[0.5, 1.0, 0.5]], dtype=torch.float32)

    onnx_path = OUTPUT_DIR / "ranknet.onnx"
    onnx_data_name = "ranknet.onnx.data"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["percentile"],
        dynamo=False,
    )

    # Force weights into an external .data file so browser-side loaders can fetch it.
    onnx_model = onnx.load(str(onnx_path))
    onnx.save_model(
        onnx_model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_data_name,
        size_threshold=0,
        convert_attribute=False,
    )

    print("Saved → output/ranknet.onnx")
    print("Saved → output/ranknet.onnx.data")

    print(f"\nAverage N estimate: {avg_N:.0f}")
    print("Training complete.")


if __name__ == "__main__":
    main()import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

import os

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

with open("lb.json") as f:
    lb_data = json.load(f)

personal_datasets = []

for filename in ["a.json", "b.json", "c.json"]:

    with open(filename) as f:
        personal_datasets.append(json.load(f))

# ─────────────────────────────────────────────────────────────
# 2. BUILD N LOOKUP
# ─────────────────────────────────────────────────────────────

n_lookup = {}

for student_data in personal_datasets:

    for test in student_data:

        if (
            test["score"] == 0 or
            test["percentile"] == 0
        ):
            continue

        name = test["testName"]

        percentile = test["percentile"] / 100.0

        if percentile >= 1.0:
            continue

        N = test["rank"] / (1 - percentile)

        if name not in n_lookup:
            n_lookup[name] = []

        n_lookup[name].append(N)

n_lookup = {
    k: np.mean(v)
    for k, v in n_lookup.items()
}

# ─────────────────────────────────────────────────────────────
# 3. BUILD MAX MARKS LOOKUP
# ─────────────────────────────────────────────────────────────

max_marks_lookup = {}

for student_data in personal_datasets:

    for test in student_data:

        name = test["testName"]

        if test.get("maxMarks"):
            max_marks_lookup[name] = test["maxMarks"]

# ─────────────────────────────────────────────────────────────
# 4. BUILD TRAINING POINTS
# ─────────────────────────────────────────────────────────────

points = []

# ── Leaderboard points ──────────────────────────────────────

for test in lb_data:

    name = test["testName"]

    avg = test.get("avg")
    topper = test.get("topper")

    lb = test.get("leaderboard", [])

    if not avg or not topper or not lb:
        continue

    if name not in n_lookup:
        continue

    if name not in max_marks_lookup:
        continue

    N = n_lookup[name]

    max_marks = max_marks_lookup[name]

    max_marks_norm = max_marks / 300.0

    for entry in lb:

        score = entry["score"]

        rank = entry["rank"]

        if topper == avg:
            continue

        x = (score - avg) / (topper - avg)

        y = 1.0 - (rank / N)

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append((
                x,
                max_marks_norm,
                y
            ))

# ── Personal student points ─────────────────────────────────

for student_data in personal_datasets:

    for test in student_data:

        if (
            test["score"] == 0 or
            test["percentile"] == 0
        ):
            continue

        name = test["testName"]

        lb_match = next(
            (
                l for l in lb_data
                if l["testName"] == name
            ),
            None
        )

        if not lb_match:
            continue

        avg = lb_match.get("avg")
        topper = lb_match.get("topper")

        if not avg or not topper:
            continue

        N = n_lookup.get(name)

        if not N:
            continue

        max_marks = test.get("maxMarks")

        if not max_marks:
            continue

        if topper == avg:
            continue

        x = (
            (test["score"] - avg)
            / (topper - avg)
        )

        y = 1.0 - (
            test["rank"] / N
        )

        max_marks_norm = max_marks / 300.0

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append((
                x,
                max_marks_norm,
                y
            ))

print(f"Total training points: {len(points)}")

# ─────────────────────────────────────────────────────────────
# 5. BUILD ARRAYS
# ─────────────────────────────────────────────────────────────

xs = np.array(
    [p[0] for p in points],
    dtype=np.float32
)

ms = np.array(
    [p[1] for p in points],
    dtype=np.float32
)

ys = np.array(
    [p[2] for p in points],
    dtype=np.float32
)

print(
    f"x range: {xs.min():.3f} → {xs.max():.3f}"
)

print(
    f"max_marks range: "
    f"{(ms.min()*300):.0f} → "
    f"{(ms.max()*300):.0f}"
)

print(
    f"y range: "
    f"{(ys.min()*100):.1f}% → "
    f"{(ys.max()*100):.1f}%"
)

# ─────────────────────────────────────────────────────────────
# 6. DEFINE MODEL
# ─────────────────────────────────────────────────────────────

class RankNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(2, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 16),
            nn.Tanh(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)

# ─────────────────────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────────────────────

X = torch.tensor(
    np.stack([xs, ms], axis=1)
)

Y = torch.tensor(
    ys
).unsqueeze(1)

dataset = TensorDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = RankNet()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

EPOCHS = 3000

for epoch in range(EPOCHS):

    model.train()

    for xb, yb in loader:

        pred = model(xb)

        loss = loss_fn(pred, yb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if (epoch + 1) % 300 == 0:

        model.eval()

        with torch.no_grad():

            mae = (
                model(X) - Y
            ).abs().mean().item()

        print(
            f"Epoch {epoch+1:4d} | "
            f"MAE: {mae*100:.2f} percentile pts"
        )

# ─────────────────────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────────────────────

model.eval()

with torch.no_grad():

    preds = model(X).squeeze().numpy()

errors = np.abs(preds - ys) * 100

print(
    f"\nFinal MAE: {errors.mean():.2f} percentile pts"
)

print(
    f"Max error: {errors.max():.2f} percentile pts"
)

print(
    f"Within ±3 pts: "
    f"{(errors < 3).mean()*100:.1f}%"
)

print(
    f"Within ±5 pts: "
    f"{(errors < 5).mean()*100:.1f}%"
)

# ─────────────────────────────────────────────────────────────
# 9. SAVE TORCH MODEL
# ─────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)

avg_N = float(
    np.mean(list(n_lookup.values()))
)

torch.save({

    "model_state": model.state_dict(),

    "meta": {

        "x_range": [
            float(xs.min()),
            float(xs.max())
        ],

        "n_points": len(points),

        "avg_N": avg_N,

        "max_marks_ref": 300
    }

}, "output/ranknet.pt")

print(
    "\nSaved → output/ranknet.pt"
)

# ─────────────────────────────────────────────────────────────
# 10. EXPORT ONNX
# ─────────────────────────────────────────────────────────────

dummy = torch.tensor(
    [[0.5, 1.0]],
    dtype=torch.float32
)

torch.onnx.export(

    model,
    dummy,

    "output/ranknet.onnx",

    export_params=True,

    opset_version=11,

    do_constant_folding=True,

    input_names=["x"],

    output_names=["percentile"],

    dynamic_axes={
        "x": {0: "batch"}
    }
)

print(
    "Saved → output/ranknet.onnx"
)

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────

print(
    f"\nAverage N estimate: {avg_N:.0f}"
)

print("Training complete.")
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import onnx

# ── 1. Load data ──────────────────────────────────────────────────────────────

with open("lb.json") as f:
    lb_data = json.load(f)

personal_datasets = []
for filename in ["a.json", "b.json", "c.json"]:
    with open(filename) as f:
        personal_datasets.append(json.load(f))

# ── 2. Build N lookup ─────────────────────────────────────────────────────────

n_lookup = {}
for student_data in personal_datasets:
    for test in student_data:
        name = test["testName"]
        if test["score"] == 0 or test["percentile"] == 0:
            continue
        N = test["rank"] / (1 - test["percentile"] / 100)
        if name not in n_lookup:
            n_lookup[name] = []
        n_lookup[name].append(N)

n_lookup = {k: np.mean(v) for k, v in n_lookup.items()}

# Build max_marks lookup from personal data
max_marks_lookup = {}
for student_data in personal_datasets:
    for test in student_data:
        name = test["testName"]
        if test.get("maxMarks"):
            max_marks_lookup[name] = test["maxMarks"]

# ── 3. Build leaderboard points ───────────────────────────────────────────────

points = []

for test in lb_data:
    name = test["testName"]
    avg = test.get("avg")
    topper = test.get("topper")
    lb = test.get("leaderboard", [])

    if not avg or not topper or not lb:
        continue
    if name not in n_lookup or name not in max_marks_lookup:
        continue

    N = n_lookup[name]
    max_marks = max_marks_lookup[name]
    max_marks_norm = max_marks / 300.0

    for entry in lb:
        x = (entry["score"] - avg) / (topper - avg)
        y = 1.0 - (entry["rank"] / N)
        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append((x, max_marks_norm, y))

# ── 4. Build personal data points ─────────────────────────────────────────────

for student_data in personal_datasets:
    for test in student_data:
        if test["score"] == 0 or test["percentile"] == 0:
            continue

        lb_match = next((l for l in lb_data if l["testName"] == test["testName"]), None)
        if not lb_match or not lb_match.get("avg"):
            continue

        N = n_lookup.get(test["testName"])
        max_marks = test.get("maxMarks")
        if not N or not max_marks:
            continue

        x = (test["score"] - lb_match["avg"]) / (lb_match["topper"] - lb_match["avg"])
        y = 1.0 - (test["rank"] / N)
        max_marks_norm = max_marks / 300.0

        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append((x, max_marks_norm, y))

print(f"Total training points: {len(points)}")

xs = np.array([p[0] for p in points], dtype=np.float32)
ms = np.array([p[1] for p in points], dtype=np.float32)
ys = np.array([p[2] for p in points], dtype=np.float32)

print(f"x range:         {xs.min():.3f} → {xs.max():.3f}")
print(f"max_marks range: {(ms.min()*300):.0f} → {(ms.max()*300):.0f}")
print(f"y range:         {(ys.min()*100):.1f}% → {(ys.max()*100):.1f}%")

# ── 5. Define model ───────────────────────────────────────────────────────────

class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ── 6. Train ──────────────────────────────────────────────────────────────────

X = torch.tensor(np.stack([xs, ms], axis=1))
Y = torch.tensor(ys).unsqueeze(1)

dataset = TensorDataset(X, Y)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

model     = RankNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

EPOCHS = 3000
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 300 == 0:
        model.eval()
        with torch.no_grad():
            mae = (model(X) - Y).abs().mean().item()
        print(f"Epoch {epoch+1:4d} | MAE: {mae*100:.2f} percentile pts")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    preds  = model(X).squeeze().numpy()
    errors = np.abs(preds - ys) * 100

worst_idx = np.argmax(errors)
print(f"\nFinal MAE:    {errors.mean():.2f} percentile points")
print(f"Max error:    {errors.max():.2f} percentile points")
print(f"Within ±3pts: {(errors < 3).mean()*100:.1f}% of points")
print(f"Within ±5pts: {(errors < 5).mean()*100:.1f}% of points")
print(f"Worst point — x: {xs[worst_idx]:.3f}, max_marks: {ms[worst_idx]*300:.0f}, actual: {ys[worst_idx]*100:.1f}%, predicted: {preds[worst_idx]*100:.1f}%, error: {errors[worst_idx]:.1f}pts")

# ── 8. Save ───────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)

avg_N = float(np.mean(list(n_lookup.values())))

torch.save({
    "model_state": model.state_dict(),
    "meta": {
        "x_range":       [float(xs.min()), float(xs.max())],
        "n_points":      len(points),
        "avg_N":         avg_N,
        "max_marks_ref": 300
    }
}, "output/ranknet.pt")

print(f"\nAverage N across tests: {avg_N:.0f}")
print("Saved → output/ranknet.pt")

# ── 9. Export ONNX (single file, no external data) ───────────────────────────

dummy = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
torch.onnx.export(
    model, dummy, "output/ranknet.onnx",
    input_names=["x"],
    output_names=["percentile"],
    dynamic_axes={"x": {0: "batch"}}
)

onnx_model = onnx.load("output/ranknet.onnx")
onnx.save(onnx_model, "output/ranknet.onnx", save_as_external_data=False)

print("Saved → output/ranknet.onnx (single file)")
