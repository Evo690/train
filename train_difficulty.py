"""
Model 1 — Difficulty Estimator
Input:  [maxMarks_norm, topper_norm]
Output: difficulty score = avg / maxMarks
"""

import json, os, glob
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("output/difficulty_model", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

with open("data/lb.json") as f:
    lb_data = json.load(f)

student_files = [f for f in glob.glob("data/*.json") if os.path.basename(f).lower() != "lb.json"]
print(f"Student files found: {[os.path.basename(f) for f in student_files]}")

# ── Points ────────────────────────────────────────────────────────────────────

points = []

for sf in student_files:
    with open(sf) as f:
        data = json.load(f)

    for test in data:
        # skip zero/absent tests
        if not test.get("score") or not test.get("rank") or not test.get("percentile"):
            continue
        if test["score"] == 0 or test["rank"] == 0 or test["percentile"] == 0:
            continue

        max_marks = test.get("maxMarks")
        if not max_marks:
            continue

        lb = next((l for l in lb_data if l["testName"] == test["testName"]), None)
        if not lb or not lb.get("avg") or not lb.get("topper"):
            continue

        avg    = lb["avg"]
        topper = lb["topper"]

        if topper <= avg:
            continue

        max_norm  = max_marks / 300.0
        top_norm  = topper / max_marks
        diff      = avg / max_marks          # target

        points.append([max_norm, top_norm, diff])

points = np.array(points, dtype=np.float32)
X = points[:, :2]
Y = points[:, 2]

print(f"Training points: {len(points)}")
print(f"Difficulty range: {Y.min():.3f} → {Y.max():.3f}")

# ── Model ─────────────────────────────────────────────────────────────────────

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(1,  activation="sigmoid")
], name="difficulty_model")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.fit(X, Y, epochs=2000, batch_size=16, verbose=0)

preds  = model.predict(X, verbose=0).flatten()
errors = np.abs(preds - Y)
print(f"Final MAE:  {errors.mean():.4f} difficulty units")
print(f"Max error:  {errors.max():.4f}")

# ── Export ────────────────────────────────────────────────────────────────────

model.save("output/difficulty_model.keras")
tfjs.converters.save_keras_model(model, "output/difficulty_model")

with open("output/difficulty_model/meta.json", "w") as f:
    json.dump({
        "inputs": ["maxMarks_norm (maxMarks/300)", "topper_norm (topper/maxMarks)"],
        "output": "difficulty score (avg/maxMarks)",
        "maxMarks_ref": 300
    }, f, indent=2)

print("Saved → output/difficulty_model/")
