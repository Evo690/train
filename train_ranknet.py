"""
Model 3 — RankNet
Input:  [x_norm, maxMarks_norm, difficulty_score]
Output: percentile (0-1)
Trained on leaderboard points + all student personal points.
Uses REAL difficulty values during training (not Model 1 predictions)
to prevent error propagation from baking in.
"""

import json, os, glob
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("output/ranknet", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

with open("data/lb.json") as f:
    lb_data = json.load(f)

student_files = [f for f in glob.glob("data/*.json") if os.path.basename(f).lower() != "lb.json"]
print(f"Student files found: {[os.path.basename(f) for f in student_files]}")

# ── N lookup ──────────────────────────────────────────────────────────────────

n_lookup = {}

for sf in student_files:
    with open(sf) as f:
        data = json.load(f)

    for test in data:
        if not test.get("score") or not test.get("rank") or not test.get("percentile"):
            continue
        if test["score"] == 0 or test["rank"] == 0 or test["percentile"] == 0:
            continue
        if test["percentile"] >= 100:
            continue

        name = test["testName"]
        N = test["rank"] / (1 - test["percentile"] / 100)
        if name not in n_lookup:
            n_lookup[name] = []
        n_lookup[name].append(N)

n_lookup = {k: float(np.mean(v)) for k, v in n_lookup.items()}
avg_N    = float(np.mean(list(n_lookup.values())))
print(f"N lookup built for {len(n_lookup)} tests | avg N: {avg_N:.0f}")

# max_marks lookup
max_marks_lookup = {}
for sf in student_files:
    with open(sf) as f:
        data = json.load(f)
    for test in data:
        if test.get("maxMarks"):
            max_marks_lookup[test["testName"]] = test["maxMarks"]

# ── Leaderboard points ────────────────────────────────────────────────────────

points = []

for test in lb_data:
    name    = test.get("testName")
    avg     = test.get("avg")
    topper  = test.get("topper")
    lb      = test.get("leaderboard", [])

    if not avg or not topper or not lb:
        continue
    if topper <= avg:
        continue
    if name not in n_lookup or name not in max_marks_lookup:
        continue

    N          = n_lookup[name]
    max_marks  = max_marks_lookup[name]
    max_norm   = max_marks / 300.0
    diff       = avg / max_marks

    for entry in lb:
        score = entry.get("score")
        rank  = entry.get("rank")
        if score is None or rank is None:
            continue

        x = (score - avg) / (topper - avg)
        y = 1.0 - (rank / N)

        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append([x, max_norm, diff, y])

# ── Personal points ───────────────────────────────────────────────────────────

for sf in student_files:
    with open(sf) as f:
        data = json.load(f)

    for test in data:
        # skip zero/absent tests
        if not test.get("score") or not test.get("rank") or not test.get("percentile"):
            continue
        if test["score"] == 0 or test["rank"] == 0 or test["percentile"] == 0:
            continue

        name      = test["testName"]
        max_marks = test.get("maxMarks")
        if not max_marks:
            continue

        lb = next((l for l in lb_data if l["testName"] == name), None)
        if not lb or not lb.get("avg") or not lb.get("topper"):
            continue

        avg    = lb["avg"]
        topper = lb["topper"]
        N      = n_lookup.get(name)

        if not N or topper <= avg:
            continue

        x        = (test["score"] - avg) / (topper - avg)
        y        = 1.0 - (test["rank"] / N)
        max_norm = max_marks / 300.0
        diff     = avg / max_marks

        if 0 < x <= 1.0 and 0 < y < 1.0:
            points.append([x, max_norm, diff, y])

points = np.array(points, dtype=np.float32)
X = points[:, :3]
Y = points[:, 3]

print(f"Total training points: {len(points)}")
print(f"x range:    {X[:,0].min():.3f} → {X[:,0].max():.3f}")
print(f"diff range: {X[:,2].min():.3f} → {X[:,2].max():.3f}")
print(f"y range:    {(Y.min()*100):.1f}% → {(Y.max()*100):.1f}%")

# ── Model ─────────────────────────────────────────────────────────────────────

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(1,  activation="sigmoid")
], name="ranknet")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.fit(X, Y, epochs=2500, batch_size=32, verbose=0)

preds  = model.predict(X, verbose=0).flatten()
errors = np.abs(preds - Y) * 100

worst  = np.argmax(errors)
print(f"\nFinal MAE:    {errors.mean():.2f} percentile pts")
print(f"Max error:    {errors.max():.2f} percentile pts")
print(f"Within ±3pts: {(errors < 3).mean()*100:.1f}%")
print(f"Within ±5pts: {(errors < 5).mean()*100:.1f}%")
print(f"Worst — x:{X[worst,0]:.3f} diff:{X[worst,2]:.3f} actual:{Y[worst]*100:.1f}% pred:{preds[worst]*100:.1f}%")

# ── Export ────────────────────────────────────────────────────────────────────

model.save("output/ranknet.keras")
tfjs.converters.save_keras_model(model, "output/ranknet")

with open("output/ranknet/meta.json", "w") as f:
    json.dump({
        "inputs": [
            "x_norm = (score - avg) / (topper - avg)",
            "maxMarks_norm = maxMarks / 300",
            "difficulty = avg / maxMarks"
        ],
        "output": "percentile (0-1)",
        "avg_N": avg_N,
        "maxMarks_ref": 300
    }, f, indent=2)

print("Saved → output/ranknet/")
