import json
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

print(
    "Training complete."
)import json
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
