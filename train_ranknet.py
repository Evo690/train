"""
train_ranknet.py
================
Trains a multi-output neural network that predicts:
  - rank
  - percentile
  - topper marks

Inputs (4 features):
  - score         (student's expected marks / actual marks)
  - avg           (expected average marks for the test)
  - maxMarks      (maximum marks for the test)
  - totalStudents (estimated total number of students)

The model is exported in TensorFlow.js Layers format to output/ranknet/
so it can be loaded and run directly in the browser.

Data source: data/*.json  (all per-student JSON files)
"""

import os
import json
import math
import glob
import random
import numpy as np

# ── Silence TF logs ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflowjs as tfjs

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Parse Data
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "ranknet")

print("=" * 60)
print("RankNet — Training Script")
print("=" * 60)
print(f"Data directory : {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def estimate_total_students(rank, percentile):
    """
    Derive totalStudents from rank and percentile.
    percentile = (1 - rank/total) * 100
    => total = rank / (1 - percentile/100)
    """
    if percentile is None or percentile <= 0 or percentile >= 100:
        return None
    if rank is None or rank <= 0:
        return None
    try:
        total = rank / (1.0 - percentile / 100.0)
        return round(total)
    except ZeroDivisionError:
        return None


records = []
test_meta = {}

json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
print(f"\nFound {len(json_files)} JSON file(s) in data/")

standard_files = []
leaderboard_files = []

for fpath in json_files:
    fname = os.path.basename(fpath)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if this is a leaderboard file
        is_leaderboard = False
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "leaderboard" in data[0]:
                is_leaderboard = True
        elif isinstance(data, dict) and "leaderboard" in data:
            is_leaderboard = True

        if is_leaderboard:
            leaderboard_files.append((fpath, data))
        else:
            standard_files.append((fpath, data))
    except Exception as e:
        print(f"  [WARN] Skipping/Error pre-reading {fname}: {e}")

# Pass 1: Parse standard files and build metadata index
for fpath, data in standard_files:
    fname = os.path.basename(fpath)
    if isinstance(data, dict):
        data = [data]
    for entry in data:
        score       = entry.get("score")
        max_marks   = entry.get("maxMarks")
        avg         = entry.get("avg")
        topper      = entry.get("topper")
        rank        = entry.get("rank")
        percentile  = entry.get("percentile")
        test_name   = entry.get("testName")

        # Skip entries missing core fields or with invalid scores/ranks
        if None in (score, max_marks, avg, topper, rank, percentile):
            continue
        if score <= 0 or rank is None or rank <= 0:
            continue
        if max_marks <= 0 or avg <= 0:
            continue

        total_students = estimate_total_students(rank, percentile)
        if total_students is None or total_students < rank:
            continue

        records.append({
            "score"          : float(score),
            "avg"            : float(avg),
            "maxMarks"       : float(max_marks),
            "totalStudents"  : float(total_students),
            "rank"           : float(rank),
            "percentile"     : float(percentile),
            "topper"         : float(topper),
        })

        if test_name:
            if test_name not in test_meta:
                test_meta[test_name] = {
                    "maxMarks": [],
                    "totalStudents": [],
                    "avg": [],
                    "topper": []
                }
            test_meta[test_name]["maxMarks"].append(float(max_marks))
            test_meta[test_name]["totalStudents"].append(float(total_students))
            test_meta[test_name]["avg"].append(float(avg))
            test_meta[test_name]["topper"].append(float(topper))

# Aggregate test metadata using median
aggregated_meta = {}
for test_name, vals in test_meta.items():
    aggregated_meta[test_name] = {
        "maxMarks": float(np.median(vals["maxMarks"])),
        "totalStudents": float(np.median(vals["totalStudents"])),
        "avg": float(np.median(vals["avg"])),
        "topper": float(np.median(vals["topper"])),
    }

# Pass 2: Parse leaderboard files and match them
leaderboard_samples = 0
for fpath, data in leaderboard_files:
    fname = os.path.basename(fpath)
    if isinstance(data, dict):
        data = [data]
    for test_entry in data:
        test_name = test_entry.get("testName")
        leaderboard = test_entry.get("leaderboard", [])

        meta = aggregated_meta.get(test_name)
        if not meta:
            continue

        max_marks = meta["maxMarks"]
        total_students = meta["totalStudents"]
        avg = test_entry.get("avg", meta["avg"])
        topper = test_entry.get("topper", meta["topper"])

        if max_marks <= 0 or total_students <= 0 or avg <= 0:
            continue

        for student in leaderboard:
            rank = student.get("rank")
            score = student.get("score")

            if rank is None or rank <= 0 or score is None or score <= 0:
                continue
            if rank > total_students:
                continue

            # Compute percentile: percentile = (1.0 - rank/totalStudents) * 100.0
            percentile = (1.0 - float(rank) / float(total_students)) * 100.0

            records.append({
                "score"          : float(score),
                "avg"            : float(avg),
                "maxMarks"       : float(max_marks),
                "totalStudents"  : float(total_students),
                "rank"           : float(rank),
                "percentile"     : float(percentile),
                "topper"         : float(topper),
            })
            leaderboard_samples += 1

print(f"Added {leaderboard_samples} samples from leaderboard files.")
print(f"Total valid samples collected: {len(records)}")

if len(records) < 10:
    raise ValueError(
        "Too few samples to train. "
        "Ensure data/*.json files contain valid student records with "
        "score, maxMarks, avg, topper, rank and percentile fields."
    )

# Shuffle records to ensure a representative validation split
random.seed(42)
random.shuffle(records)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering — Ratio / Standardised Inputs
# ─────────────────────────────────────────────────────────────────────────────
# All features are dimensionless ratios so the model is scale-invariant
# across tests with different max-mark values (180, 198, 300 …).
#
#  score_frac   = score / maxMarks   → fraction of paper scored (0–1)
#  avg_frac     = avg   / maxMarks   → test difficulty indicator (0–1)
#  score_vs_avg = score / avg        → above/below-average factor (>1 good)
#  log_total    = log(totalStudents) → competition scale, log-compressed
#
# Outputs are also expressed as bounded ratios:
#  rank_ratio   = rank / totalStudents  (0–1, decode → × totalStudents)
#  pct_norm     = percentile / 100      (0–1, decode → × 100)
#  topper_frac  = topper / maxMarks     (0–1, decode → × maxMarks)

# Build raw arrays
score_arr   = np.array([r["score"]         for r in records], dtype=np.float32)
avg_arr     = np.array([r["avg"]           for r in records], dtype=np.float32)
maxM_arr    = np.array([r["maxMarks"]      for r in records], dtype=np.float32)
total_arr   = np.array([r["totalStudents"] for r in records], dtype=np.float32)

rank_arr    = np.array([r["rank"]          for r in records], dtype=np.float32)
pct_arr     = np.array([r["percentile"]    for r in records], dtype=np.float32)
topper_arr  = np.array([r["topper"]        for r in records], dtype=np.float32)

# ── Input features (all ratio-based, no per-dataset mean/std needed) ─────────
score_frac   = score_arr  / maxM_arr          # 0–1
avg_frac     = avg_arr    / maxM_arr          # 0–1
score_vs_avg = score_arr  / (avg_arr + 1e-8)  # ratio; clipped below
log_total    = np.log1p(total_arr)            # log(1 + N)

# Clip score_vs_avg to a sane range (handles extreme outliers)
score_vs_avg = np.clip(score_vs_avg, 0.0, 5.0)

X = np.column_stack([
    score_frac,
    avg_frac,
    score_vs_avg,
    log_total,
]).astype(np.float32)

# ── Output targets (ratio-based, naturally bounded) ───────────────────────────
Y_rank   = (rank_arr   / total_arr).reshape(-1, 1)          # 0–1
Y_pct    = (pct_arr    / 100.0).reshape(-1, 1)              # 0–1
Y_topper = (topper_arr / maxM_arr).reshape(-1, 1)           # 0–1

# Clip targets to valid range
Y_rank   = np.clip(Y_rank,   0.0, 1.0).astype(np.float32)
Y_pct    = np.clip(Y_pct,    0.0, 1.0).astype(np.float32)
Y_topper = np.clip(Y_topper, 0.0, 1.0).astype(np.float32)

print(f"\nFeature matrix shape : {X.shape}")
print(f"  score_frac   — min={score_frac.min():.3f}, max={score_frac.max():.3f}, mean={score_frac.mean():.3f}")
print(f"  avg_frac     — min={avg_frac.min():.3f}, max={avg_frac.max():.3f}, mean={avg_frac.mean():.3f}")
print(f"  score_vs_avg — min={score_vs_avg.min():.3f}, max={score_vs_avg.max():.3f}, mean={score_vs_avg.mean():.3f}")
print(f"  log_total    — min={log_total.min():.2f}, max={log_total.max():.2f}, mean={log_total.mean():.2f}")
print(f"\nTarget ranges:")
print(f"  rank_ratio (rank/total) — min={Y_rank.min():.4f}, max={Y_rank.max():.4f}")
print(f"  pct_norm  (pct/100)     — min={Y_pct.min():.4f}, max={Y_pct.max():.4f}")
print(f"  topper_frac(top/maxM)   — min={Y_topper.min():.4f}, max={Y_topper.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build Model  (Shared trunk → 3 output heads)
# ─────────────────────────────────────────────────────────────────────────────

def math_preprocessing(x):
    # Columns: 0: score_frac, 1: avg_frac, 2: score_vs_avg, 3: log_total
    sf  = x[:, 0:1]
    af  = x[:, 1:2]
    sva = x[:, 2:3]
    lt  = x[:, 3:4]

    # Mathematical transformations for standardization
    diff = sf - af
    z    = diff / 0.15
    phi  = 0.5 * (1.0 + tf.math.erf(z / 1.41421356)) # Normal CDF prior
    log_sva = tf.math.log(sva + 0.1)                 # Log-scaled score vs average (offset 0.1 for stability)
    lt_norm = lt / 10.0                              # Scale competition log_total

    # Student-specific features (full set)
    student_feats = tf.concat([x, diff, z, phi, log_sva, lt_norm], axis=-1)
    
    # Test-wide features (only avg_frac and lt_norm)
    test_feats = tf.concat([af, lt_norm], axis=-1)

    return student_feats, test_feats

tf.random.set_seed(42)

inputs = tf.keras.Input(shape=(4,), name="features")

# Apply mathematical preprocessing inside the Keras graph
preprocessed, test_features = tf.keras.layers.Lambda(math_preprocessing, name="math_preprocess")(inputs)

reg = tf.keras.regularizers.l2(1e-4)

# Project preprocessed features to a 128-dimensional space
x = tf.keras.layers.Dense(128, activation="swish", kernel_regularizer=reg, name="shared_proj")(preprocessed)

# First Residual Block
r1 = tf.keras.layers.Dense(128, activation="swish", kernel_regularizer=reg, name="res1_d1")(x)
r1 = tf.keras.layers.Dropout(0.1, name="res1_drop")(r1)
r1 = tf.keras.layers.Dense(128, kernel_regularizer=reg, name="res1_d2")(r1)
x  = tf.keras.layers.add([x, r1], name="res1_add")
x  = tf.keras.layers.Activation("swish", name="res1_act")(x)

# Second Residual Block
r2 = tf.keras.layers.Dense(128, activation="swish", kernel_regularizer=reg, name="res2_d1")(x)
r2 = tf.keras.layers.Dropout(0.1, name="res2_drop")(r2)
r2 = tf.keras.layers.Dense(128, kernel_regularizer=reg, name="res2_d2")(r2)
x  = tf.keras.layers.add([x, r2], name="res2_add")
x  = tf.keras.layers.Activation("swish", name="res2_act")(x)

# Final shared representation projection
shared_out = tf.keras.layers.Dense(64, activation="swish", kernel_regularizer=reg, name="shared_out")(x)

# NOTE: outputs are all ratio targets in [0, 1] so we use sigmoid activations

# ── Output heads (sigmoid → outputs are ratios in [0, 1]) ───────────────────
rank_head = tf.keras.layers.Dense(32, activation="swish", kernel_regularizer=reg, name="rank_h1")(shared_out)
rank_out  = tf.keras.layers.Dense(1,  activation="sigmoid", name="rank")(rank_head)

pct_head  = tf.keras.layers.Dense(32, activation="swish", kernel_regularizer=reg, name="pct_h1")(shared_out)
pct_out   = tf.keras.layers.Dense(1,  activation="sigmoid", name="percentile")(pct_head)

# Topper head only receives test-wide features (avg and total students)
# This prevents predicted topper score from fluctuating per student scorecard.
test_proj = tf.keras.layers.Dense(32, activation="swish", kernel_regularizer=reg, name="test_proj")(test_features)
top_head  = tf.keras.layers.Dense(32, activation="swish", kernel_regularizer=reg, name="top_h1")(test_proj)
top_out   = tf.keras.layers.Dense(1,  activation="sigmoid", name="topper")(top_head)

model = tf.keras.Model(
    inputs=inputs,
    outputs=[rank_out, pct_out, top_out],
    name="RankNet"
)

# Enforce rank and percentile mathematical consistency
# Since rank_ratio = rank/total and pct_norm = percentile/100, we expect rank_ratio + pct_norm = 1.0.
# We add this mathematical constraint directly to the model's loss objective.
consistency_loss = tf.reduce_mean(tf.square(rank_out + pct_out - 1.0))
model.add_loss(consistency_loss * 1.0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "rank"      : "mse",
        "percentile": "mse",
        "topper"    : "mse",
    },
    loss_weights={
        "rank"      : 1.0,
        "percentile": 1.0,
        "topper"    : 0.8,
    },
    metrics={
        "rank"      : ["mae"],
        "percentile": ["mae"],
        "topper"    : ["mae"],
    }
)

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
    ),
]

print("\n── Training ─────────────────────────────────────────────────────────")
history = model.fit(
    X,
    {"rank": Y_rank, "percentile": Y_pct, "topper": Y_topper},
    epochs=500,
    batch_size=min(32, max(8, len(records) // 5)),
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Evaluation (full dataset) ────────────────────────────────────────")
preds = model.predict(X, verbose=0)

# De-normalise predictions back to original units
pred_rank_ratio   = preds[0].flatten()  # rank / totalStudents
pred_pct_norm     = preds[1].flatten()  # percentile / 100
pred_topper_frac  = preds[2].flatten()  # topper / maxMarks

pred_rank   = pred_rank_ratio  * total_arr
pred_pct    = pred_pct_norm    * 100.0
pred_topper = pred_topper_frac * maxM_arr

mae_rank    = np.mean(np.abs(pred_rank   - rank_arr))
mae_pct     = np.mean(np.abs(pred_pct    - pct_arr))
mae_topper  = np.mean(np.abs(pred_topper - topper_arr))

# Calculate percentile accuracy within ±5% and ±3% thresholds
pct_diff = np.abs(pred_pct - pct_arr)
within_5 = np.mean(pct_diff <= 5.0) * 100.0
within_3 = np.mean(pct_diff <= 3.0) * 100.0

print(f"  MAE rank       : {mae_rank:.2f}")
print(f"  MAE percentile : {mae_pct:.4f}")
print(f"  MAE topper     : {mae_topper:.2f}")
print(f"  Within ±5%ile  : {within_5:.2f}% of samples")
print(f"  Within ±3%ile  : {within_3:.2f}% of samples")

# Sample predictions vs actuals
print("\n── Sample predictions (first 5 records) ────────────────────────────")
print(f"{'score/max':>10} {'avg/max':>8} {'s/avg':>6} {'logN':>6} "
      f"|| {'rank_act':>9} {'rank_pred':>10} "
      f"|| {'pct_act':>8} {'pct_pred':>9} "
      f"|| {'top_act':>8} {'top_pred':>9}")
print("-" * 100)
for i in range(min(5, len(records))):
    r = records[i]
    sf  = r['score'] / r['maxMarks']
    af  = r['avg']   / r['maxMarks']
    sva = r['score'] / (r['avg'] + 1e-8)
    lt  = math.log1p(r['totalStudents'])
    print(f"{sf:>10.3f} {af:>8.3f} {sva:>6.2f} {lt:>6.2f} "
          f"|| {r['rank']:>9.0f} {pred_rank[i]:>10.1f} "
          f"|| {r['percentile']:>8.2f} {pred_pct[i]:>9.2f} "
          f"|| {r['topper']:>8.0f} {pred_topper[i]:>9.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Export Model as TensorFlow.js Layers Format
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n── Exporting TF.js model to {OUTPUT_DIR} ─────────────────────────")
tfjs.converters.save_keras_model(model, OUTPUT_DIR)
print("  Model exported.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Normalisation Metadata (for use during inference in the browser)
# ─────────────────────────────────────────────────────────────────────────────

metadata = {
    "description": (
        "RankNet — multi-output neural network. "
        "Inputs are dimensionless ratios (scale-invariant) preprocessed with mathematical "
        "transformations (Z-score, normal CDF prior, log-scale, and scaling) inside the Keras graph. "
        "Outputs are ratios decoded with the same raw values passed at inference."
    ),
    "inputs": {
        "score_frac"   : {"formula": "score / maxMarks",          "range": "0–1"},
        "avg_frac"     : {"formula": "avg / maxMarks",            "range": "0–1"},
        "score_vs_avg" : {"formula": "clip(score / avg, 0, 5)",   "range": "0–5"},
        "log_total"    : {"formula": "log1p(totalStudents)",       "range": "open"},
    },
    "outputs": {
        "rank"       : {"formula": "model_out[0] * totalStudents", "unit": "rank number"},
        "percentile" : {"formula": "model_out[1] * 100",           "unit": "percentile 0–100"},
        "topper"     : {"formula": "model_out[2] * maxMarks",      "unit": "marks"},
    },
    "training": {
        "samples"    : len(records),
        "mae_rank"   : round(float(mae_rank), 2),
        "mae_pct"    : round(float(mae_pct), 4),
        "mae_topper" : round(float(mae_topper), 2),
        "within_5_pct": round(float(within_5), 2),
        "within_3_pct": round(float(within_3), 2),
        "epochs_run" : len(history.history["loss"]),
    },
    "input_order"  : ["score_frac", "avg_frac", "score_vs_avg", "log_total"],
    "output_order" : ["rank_ratio", "pct_norm", "topper_frac"],
    "usage": {
        "step1": "Compute score_frac = score / maxMarks",
        "step2": "Compute avg_frac = avg / maxMarks",
        "step3": "Compute score_vs_avg = Math.min(score / avg, 5.0)",
        "step4": "Compute log_total = Math.log1p(totalStudents)",
        "step5": "Run model.predict(tf.tensor2d([[score_frac, avg_frac, score_vs_avg, log_total]]))",
        "step6": "rank = outputs[0][0] * totalStudents",
        "step7": "percentile = outputs[1][0] * 100",
        "step8": "topper = outputs[2][0] * maxMarks",
    }
}

meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"  Metadata saved → {meta_path}")
print("\n✅ Training complete!")
print(f"   Artifact directory: output/ranknet/")
print(f"   Files: model.json, group1-shard*.bin, metadata.json")
