# train_ranknet.py

import json, os, glob
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("output/ranknet", exist_ok=True)

# ── Fitted statistical constants ──────────────────────────────────────────────

SLOPE_TGN     = -0.6776
INTERCEPT_TGN =  0.8130
SLOPE_K       = -4.6774
INTERCEPT_K   =  4.5740

# ── Statistical normalization ─────────────────────────────────────────────────

def estimate_topper(avg, maxMarks):
    difficulty      = avg / maxMarks
    topper_gap_norm = SLOPE_TGN * difficulty + INTERCEPT_TGN
    topper          = avg + topper_gap_norm * maxMarks
    return min(topper, maxMarks)

def dynamic_k(difficulty):
    return SLOPE_K * difficulty + INTERCEPT_K

def normalize_input(score, avg, maxMarks):
    difficulty    = avg / maxMarks
    topper        = estimate_topper(avg, maxMarks)
    k             = dynamic_k(difficulty)
    sigma         = (topper - avg) / k if k > 0 else 1.0
    z             = (score - avg) / sigma if sigma > 0 else 0.0
    # tanh-squash z so extreme outliers stay bounded in (-1, 1)
    z_sq          = float(np.tanh(z / 3.0))
    gap           = topper - avg
    x_norm        = (score - avg) / gap if gap > 0 else 0.0
    x_norm        = max(0.01, min(x_norm, 1.0))
    # Use maxMarks / 300 but also keep a separate flag for non-300 papers
    maxMarks_norm = maxMarks / 300.0
    return z_sq, x_norm, difficulty, maxMarks_norm

# ── Load data ─────────────────────────────────────────────────────────────────

LB_PATH = "data/lb.json"
with open(LB_PATH) as f:
    lb_data = json.load(f)

student_files = [
    f for f in glob.glob("data/*.json")
    if os.path.basename(f).lower() != "lb.json"
]
print(f"Leaderboard: {LB_PATH}")
print(f"Student files ({len(student_files)}): {[os.path.basename(f) for f in student_files]}")

# ── N lookup ──────────────────────────────────────────────────────────────────

n_lookup = {}
for sf in student_files:
    with open(sf) as f:
        data = json.load(f)
    for test in data:
        if not test.get("score")      or test["score"] == 0:      continue
        if not test.get("rank")       or test["rank"] == 0:       continue
        if not test.get("percentile") or test["percentile"] == 0: continue
        if test["percentile"] >= 100:                              continue
        name = test["testName"]
        N    = test["rank"] / (1 - test["percentile"] / 100)
        if name not in n_lookup:
            n_lookup[name] = []
        n_lookup[name].append(N)

n_lookup = {k: float(np.mean(v)) for k, v in n_lookup.items()}
avg_N    = float(np.mean(list(n_lookup.values())))
print(f"N lookup built for {len(n_lookup)} tests | avg N: {avg_N:.0f}")

# ── max_marks lookup ──────────────────────────────────────────────────────────

max_marks_lookup = {}
for sf in student_files:
    with open(sf) as f:
        data = json.load(f)
    for test in data:
        if test.get("maxMarks"):
            max_marks_lookup[test["testName"]] = test["maxMarks"]

# ── Fit statistical constants dynamically ─────────────────────────────────────
print("\nFitting statistical constants dynamically...")

# 1. Fit Topper Gap Relation
difficulties = []
gaps = []
for test in lb_data:
    name = test.get("testName")
    avg = test.get("avg")
    topper = test.get("topper")
    if not avg or not topper or name not in max_marks_lookup:
        continue
    maxMarks = max_marks_lookup[name]
    if maxMarks <= 0 or topper <= avg:
        continue
    difficulties.append(avg / maxMarks)
    gaps.append((topper - avg) / maxMarks)

if len(difficulties) > 1:
    SLOPE_TGN, INTERCEPT_TGN = np.polyfit(difficulties, gaps, 1)
    print(f"Fitted Topper Gap: SLOPE_TGN = {SLOPE_TGN:.6f}, INTERCEPT_TGN = {INTERCEPT_TGN:.6f}")
else:
    print("Warning: Insufficient data to fit topper gap. Using defaults.")

# 2. Fit Dynamic K Relation
from scipy.stats import norm
test_k_values = {}
for sf in student_files:
    with open(sf) as f:
        data = json.load(f)
    for test in data:
        name = test["testName"]
        score = test.get("score")
        percentile = test.get("percentile")
        maxMarks = test.get("maxMarks")
        if not score or not percentile or not maxMarks:
            continue
        lb = next((l for l in lb_data if l["testName"] == name), None)
        if not lb or not lb.get("avg") or not lb.get("topper") or lb["topper"] <= lb["avg"]:
            continue
        if percentile <= 1.0 or percentile >= 99.0 or abs(score - lb["avg"]) < 1e-2:
            continue
        z = norm.ppf(percentile / 100.0)
        if z * (score - lb["avg"]) <= 0:
            continue
        sigma_est = (score - lb["avg"]) / z
        k_est = (lb["topper"] - lb["avg"]) / sigma_est
        if 0 < k_est <= 15:
            if name not in test_k_values:
                test_k_values[name] = []
            test_k_values[name].append(k_est)

diff_k = []
k_values = []
for name, ks in test_k_values.items():
    maxMarks = max_marks_lookup.get(name)
    lb = next((l for l in lb_data if l["testName"] == name), None)
    if not lb or not maxMarks:
        continue
    diff_k.append(lb["avg"] / maxMarks)
    k_values.append(np.mean(ks))

if len(diff_k) > 1:
    SLOPE_K, INTERCEPT_K = np.polyfit(diff_k, k_values, 1)
    print(f"Fitted Dynamic K: SLOPE_K = {SLOPE_K:.6f}, INTERCEPT_K = {INTERCEPT_K:.6f}")
else:
    print("Warning: Insufficient data to fit dynamic K. Using defaults.")

# ── Build training points ─────────────────────────────────────────────────────

points = []

# Leaderboard points
for test in lb_data:
    name   = test.get("testName")
    avg    = test.get("avg")
    topper = test.get("topper")
    lb     = test.get("leaderboard", [])

    if not avg or not topper or not lb:    continue
    if topper <= avg:                      continue
    if name not in n_lookup:               continue
    if name not in max_marks_lookup:       continue

    N        = n_lookup[name]
    maxMarks = max_marks_lookup[name]

    for entry in lb:
        score = entry.get("score")
        rank  = entry.get("rank")
        if score is None or rank is None:  continue

        y = 1.0 - (rank / N)
        if not (0 < y < 1.0):             continue

        z_sq, x_norm, difficulty, maxMarks_norm = normalize_input(score, avg, maxMarks)
        if x_norm <= 0:                    continue

        points.append([z_sq, x_norm, difficulty, maxMarks_norm, y])

lb_count = len(points)
print(f"Leaderboard points: {lb_count}")

# Student personal points
for sf in student_files:
    with open(sf) as f:
        data = json.load(f)
    for test in data:
        if not test.get("score")      or test["score"] == 0:      continue
        if not test.get("rank")       or test["rank"] == 0:       continue
        if not test.get("percentile") or test["percentile"] == 0: continue

        name     = test["testName"]
        maxMarks = test.get("maxMarks")
        if not maxMarks:                                           continue

        lb = next((l for l in lb_data if l["testName"] == name), None)
        if not lb or not lb.get("avg") or not lb.get("topper"):   continue

        avg    = lb["avg"]
        topper = lb["topper"]
        N      = n_lookup.get(name)
        if not N or topper <= avg:                                 continue

        y = 1.0 - (test["rank"] / N)
        if not (0 < y < 1.0):                                     continue

        z_sq, x_norm, difficulty, maxMarks_norm = normalize_input(test["score"], avg, maxMarks)
        if x_norm <= 0:                                            continue

        points.append([z_sq, x_norm, difficulty, maxMarks_norm, y])

print(f"Student personal points: {len(points) - lb_count}")
print(f"Total training points: {len(points)}")

points = np.array(points, dtype=np.float32)

# ── Feature-level standardization ────────────────────────────────────────────
# Computed on ALL data; saved to meta.json so JS inference replicates it.

X_raw = points[:, :4]
Y     = points[:, 4]

feat_mean = X_raw.mean(axis=0)
feat_std  = X_raw.std(axis=0) + 1e-8

X = (X_raw - feat_mean) / feat_std

print(f"\nFeature means: {feat_mean.tolist()}")
print(f"Feature stds:  {feat_std.tolist()}")
print(f"\nz_sq range:       {X[:,0].min():.3f} -> {X[:,0].max():.3f}")
print(f"x_norm range:     {X[:,1].min():.3f} -> {X[:,1].max():.3f}")
print(f"difficulty range: {X[:,2].min():.3f} -> {X[:,2].max():.3f}")
print(f"maxMarks range:   {X[:,3].min():.3f} -> {X[:,3].max():.3f}")
print(f"y range:          {(Y.min()*100):.1f}% -> {(Y.max()*100):.1f}%")

# ── Train / validation split ──────────────────────────────────────────────────

rng   = np.random.default_rng(SEED)
idx   = rng.permutation(len(X))
split = int(0.85 * len(X))
tr_idx, val_idx = idx[:split], idx[split:]

X_tr,  Y_tr  = X[tr_idx],  Y[tr_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
print(f"\nTrain: {len(X_tr)}  Val: {len(X_val)}")

# ── Model ─────────────────────────────────────────────────────────────────────
# No BatchNormalization.
# Small dataset → small network + L2 + Dropout to fight overfitting.
# swish activation generalises better than tanh for this shape of data.

L2 = 2e-4

def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(
            64, activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(L2),
            kernel_initializer="he_normal"
        ),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(
            64, activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(L2),
            kernel_initializer="he_normal"
        ),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(
            32, activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(L2),
            kernel_initializer="he_normal"
        ),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name="ranknet")

model = make_model()

# Cosine-annealed LR: warm start at 1e-3, cools to 1e-5
lr_sched = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=4000,
    alpha=1e-2,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_sched),
    loss="huber",        # robust to the few noisy labels in the dataset
    metrics=["mae"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=200,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=80,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    X_tr, Y_tr,
    epochs=4000,
    batch_size=16,          # smaller batch → noisier gradients → less overfitting
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    verbose=0,
)
print(f"Stopped at epoch {len(history.history['loss'])}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

def report(tag, Xb, Yb):
    preds  = model.predict(Xb, verbose=0).flatten()
    errors = np.abs(preds - Yb) * 100
    worst  = np.argmax(errors)
    print(f"\n{tag} MAE:    {errors.mean():.2f} percentile pts")
    print(f"{tag} Max err: {errors.max():.2f} percentile pts")
    print(f"{tag} ±3pts:   {(errors < 3).mean()*100:.1f}%")
    print(f"{tag} ±5pts:   {(errors < 5).mean()*100:.1f}%")
    print(f"{tag} Worst — actual:{Yb[worst]*100:.1f}% pred:{preds[worst]*100:.1f}%")

report("Train", X_tr, Y_tr)
report("Val  ", X_val, Y_val)

# ── Save ──────────────────────────────────────────────────────────────────────

tfjs.converters.save_keras_model(model, "output/ranknet")

meta = {
    "inputs":       ["z_sq", "x_norm", "difficulty", "maxMarks_norm"],
    "output":       "percentile (0-1)",
    "avg_N":        avg_N,
    "maxMarks_ref": 300,
    "normalization": {
        "feat_mean": feat_mean.tolist(),
        "feat_std":  feat_std.tolist(),
    },
    "stat_constants": {
        "slope_tgn":     SLOPE_TGN,
        "intercept_tgn": INTERCEPT_TGN,
        "slope_k":       SLOPE_K,
        "intercept_k":   INTERCEPT_K
    }
}

with open("output/ranknet/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved -> output/ranknet/")
print(f"Saved -> output/ranknet/meta.json")
