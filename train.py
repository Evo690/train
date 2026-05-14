import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_DIR = Path(".")
OUTPUT_DIR = Path("output")

SEED = 42

EPOCHS = 2500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# ------------------------------------------------------------
# SET SEED
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

with open("lb.json", "r", encoding="utf-8") as f:
    lb_data = json.load(f)

personal_datasets = []

for filename in [
    "a.json",
    "b.json",
    "c.json"
]:
    with open(filename, "r", encoding="utf-8") as f:
        personal_datasets.append(json.load(f))

print("Loaded all JSON files.")

# ------------------------------------------------------------
# BUILD N LOOKUP
# ------------------------------------------------------------

n_lookup = {}

for student_data in personal_datasets:

    for test in student_data:

        score = test.get("score", 0)
        percentile = test.get("percentile", 0)
        rank = test.get("rank", 0)

        if (
            score == 0 or
            percentile == 0 or
            rank == 0
        ):
            continue

        if percentile >= 100:
            continue

        test_name = test["testName"]

        N = rank / (1 - percentile / 100)

        if test_name not in n_lookup:
            n_lookup[test_name] = []

        n_lookup[test_name].append(N)

n_lookup = {
    k: float(np.mean(v))
    for k, v in n_lookup.items()
}

print(f"Built N lookup for {len(n_lookup)} tests.")

# ------------------------------------------------------------
# MAX MARKS LOOKUP
# ------------------------------------------------------------

max_marks_lookup = {}

for student_data in personal_datasets:

    for test in student_data:

        name = test["testName"]

        max_marks = test.get("maxMarks")

        if max_marks:
            max_marks_lookup[name] = max_marks

# ------------------------------------------------------------
# BUILD TRAINING POINTS
# ------------------------------------------------------------

points = []

# -----------------------------
# LEADERBOARD POINTS
# -----------------------------

for test in lb_data:

    name = test.get("testName")

    avg = test.get("avg")
    topper = test.get("topper")

    leaderboard = test.get("leaderboard", [])

    if (
        not avg or
        not topper or
        not leaderboard
    ):
        continue

    if topper <= avg:
        continue

    if name not in n_lookup:
        continue

    if name not in max_marks_lookup:
        continue

    N = n_lookup[name]

    max_marks = max_marks_lookup[name]

    max_marks_norm = max_marks / 300.0

    difficulty_proxy = avg / max_marks

    for entry in leaderboard:

        score = entry.get("score")
        rank = entry.get("rank")

        if score is None or rank is None:
            continue

        x = (
            (score - avg)
            / (topper - avg)
        )

        y = 1.0 - (rank / N)

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append((
                x,
                max_marks_norm,
                difficulty_proxy,
                y
            ))

# -----------------------------
# PERSONAL DATA POINTS
# -----------------------------

for student_data in personal_datasets:

    for test in student_data:

        score = test.get("score", 0)
        percentile = test.get("percentile", 0)
        rank = test.get("rank", 0)

        if (
            score == 0 or
            percentile == 0 or
            rank == 0
        ):
            continue

        test_name = test["testName"]

        lb_match = next(
            (
                l for l in lb_data
                if l["testName"] == test_name
            ),
            None
        )

        if not lb_match:
            continue

        avg = lb_match.get("avg")
        topper = lb_match.get("topper")

        if (
            not avg or
            not topper
        ):
            continue

        if topper <= avg:
            continue

        N = n_lookup.get(test_name)

        if not N:
            continue

        max_marks = test.get("maxMarks")

        if not max_marks:
            continue

        max_marks_norm = max_marks / 300.0

        difficulty_proxy = avg / max_marks

        x = (
            (score - avg)
            / (topper - avg)
        )

        y = 1.0 - (rank / N)

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append((
                x,
                max_marks_norm,
                difficulty_proxy,
                y
            ))

print(f"Total training points: {len(points)}")

# ------------------------------------------------------------
# BUILD NUMPY ARRAYS
# ------------------------------------------------------------

xs = np.array(
    [p[0] for p in points],
    dtype=np.float32
)

ms = np.array(
    [p[1] for p in points],
    dtype=np.float32
)

ds = np.array(
    [p[2] for p in points],
    dtype=np.float32
)

ys = np.array(
    [p[3] for p in points],
    dtype=np.float32
)

print(
    f"x range: "
    f"{xs.min():.3f} → {xs.max():.3f}"
)

print(
    f"max_marks range: "
    f"{(ms.min()*300):.0f} → {(ms.max()*300):.0f}"
)

print(
    f"difficulty range: "
    f"{ds.min():.3f} → {ds.max():.3f}"
)

print(
    f"y range: "
    f"{(ys.min()*100):.1f}% → {(ys.max()*100):.1f}%"
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class RankNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(3, 32),
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

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

X = torch.tensor(
    np.stack([xs, ms, ds], axis=1),
    dtype=torch.float32
)

Y = torch.tensor(
    ys,
    dtype=torch.float32
).unsqueeze(1)

dataset = TensorDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = RankNet()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

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

        total_loss += loss.item()

    if (
        (epoch + 1) % 250 == 0 or
        epoch == 0
    ):

        model.eval()

        with torch.no_grad():

            mae = (
                model(X) - Y
            ).abs().mean().item()

        print(
            f"Epoch {epoch+1:4d} | "
            f"Loss: {total_loss:.6f} | "
            f"MAE: {mae*100:.2f} percentile pts"
        )

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    preds = model(X).squeeze().numpy()

errors = np.abs(preds - ys) * 100

print(
    f"\nFinal MAE: "
    f"{errors.mean():.2f} percentile pts"
)

print(
    f"Max error: "
    f"{errors.max():.2f} percentile pts"
)

print(
    f"Within ±3 pts: "
    f"{(errors < 3).mean()*100:.1f}%"
)

print(
    f"Within ±5 pts: "
    f"{(errors < 5).mean()*100:.1f}%"
)

# ------------------------------------------------------------
# SAVE PT MODEL
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

avg_N = float(
    np.mean(list(n_lookup.values()))
)

torch.save({

    "model_state": model.state_dict(),

    "meta": {

        "input_dim": 3,

        "avg_N": avg_N,

        "feature_names": [
            "score_strength",
            "max_marks_norm",
            "difficulty_proxy"
        ]
    }

}, OUTPUT_DIR / "ranknet.pt")

print(
    "\nSaved → output/ranknet.pt"
)

# ------------------------------------------------------------
# EXPORT SINGLE-FILE ONNX
# ------------------------------------------------------------

dummy = torch.tensor(
    [[0.5, 1.0, 0.5]],
    dtype=torch.float32
)

onnx_path = OUTPUT_DIR / "ranknet.onnx"

torch.onnx.export(

    model,
    dummy,

    str(onnx_path),

    export_params=True,

    opset_version=18,

    do_constant_folding=True,

    input_names=["x"],

    output_names=["percentile"],

    dynamo=False
)

print(
    "Saved → output/ranknet.onnx"
)

print(
    f"\nAverage N estimate: {avg_N:.0f}"
)

print(
    "Training complete."
)
