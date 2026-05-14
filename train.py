import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

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

# ── 3. Build leaderboard points ───────────────────────────────────────────────

points = []

for test in lb_data:
    name = test["testName"]
    avg = test.get("avg")
    topper = test.get("topper")
    lb = test.get("leaderboard", [])

    if not avg or not topper or not lb or name not in n_lookup:
        continue

    N = n_lookup[name]
    for entry in lb:
        x = (entry["score"] - avg) / (topper - avg)
        y = 1.0 - (entry["rank"] / N)
        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append((x, y))

# ── 4. Build personal data points ─────────────────────────────────────────────

for student_data in personal_datasets:
    for test in student_data:
        if test["score"] == 0 or test["percentile"] == 0:
            continue

        lb_match = next((l for l in lb_data if l["testName"] == test["testName"]), None)
        if not lb_match or not lb_match.get("avg"):
            continue

        N = n_lookup.get(test["testName"])
        if not N:
            continue

        x = (test["score"] - lb_match["avg"]) / (lb_match["topper"] - lb_match["avg"])
        y = 1.0 - (test["rank"] / N)
        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append((x, y))

print(f"Total training points: {len(points)}")

xs = np.array([p[0] for p in points], dtype=np.float32)
ys = np.array([p[1] for p in points], dtype=np.float32)

print(f"x range: {xs.min():.3f} → {xs.max():.3f}")
print(f"y (percentile) range: {(ys.min()*100):.1f}% → {(ys.max()*100):.1f}%")

# ── 5. Define model ───────────────────────────────────────────────────────────

class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
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

# ── 6. Train ──────────────────────────────────────────────────────────────────

X = torch.tensor(xs).unsqueeze(1)
Y = torch.tensor(ys).unsqueeze(1)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RankNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            mae = (model(X) - Y).abs().mean().item()
        print(f"Epoch {epoch+1:4d} | MAE: {mae*100:.2f} percentile pts")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    preds = model(X).squeeze().numpy()
    errors = np.abs(preds - ys) * 100

print(f"\nFinal MAE:    {errors.mean():.2f} percentile points")
print(f"Max error:    {errors.max():.2f} percentile points")
print(f"Within ±3pts: {(errors < 3).mean()*100:.1f}% of points")
print(f"Within ±5pts: {(errors < 5).mean()*100:.1f}% of points")

# ── 8. Save ───────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "meta": {
        "x_range": [float(xs.min()), float(xs.max())],
        "n_points": len(points),
        "avg_N": float(np.mean(list(n_lookup.values())))
    }
}, "output/ranknet.pt")

print("Model saved → output/ranknet.pt")
