import json
import os
import random

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEED = 42

EPOCHS = 2500
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("output", exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

with open("lb.json", "r", encoding="utf-8") as f:
    lb_data = json.load(f)

student_files = [
    "a.json",
    "b.json",
    "c.json"
]

personal_datasets = []

for filename in student_files:

    with open(filename, "r", encoding="utf-8") as f:

        personal_datasets.append(
            json.load(f)
        )

print("Loaded all datasets.")

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

        name = test["testName"]

        N = rank / (
            1 - percentile / 100
        )

        if name not in n_lookup:
            n_lookup[name] = []

        n_lookup[name].append(N)

n_lookup = {
    k: float(np.mean(v))
    for k, v in n_lookup.items()
}

print(
    f"Built N lookup for "
    f"{len(n_lookup)} tests."
)

# ------------------------------------------------------------
# BUILD MAX MARKS LOOKUP
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

    leaderboard = test.get(
        "leaderboard",
        []
    )

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

    max_marks_norm = (
        max_marks / 300.0
    )

    difficulty_proxy = (
        avg / max_marks
    )

    for entry in leaderboard:

        score = entry.get("score")
        rank = entry.get("rank")

        if (
            score is None or
            rank is None
        ):
            continue

        x = (
            (score - avg)
            / (topper - avg)
        )

        y = 1.0 - (
            rank / N
        )

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append([
                x,
                max_marks_norm,
                difficulty_proxy,
                y
            ])

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

        max_marks = test.get(
            "maxMarks"
        )

        if not max_marks:
            continue

        max_marks_norm = (
            max_marks / 300.0
        )

        difficulty_proxy = (
            avg / max_marks
        )

        x = (
            (score - avg)
            / (topper - avg)
        )

        y = 1.0 - (
            rank / N
        )

        if (
            0 < x <= 1.0 and
            0 < y < 1.0
        ):
            points.append([
                x,
                max_marks_norm,
                difficulty_proxy,
                y
            ])

print(
    f"Total training points: "
    f"{len(points)}"
)

# ------------------------------------------------------------
# BUILD ARRAYS
# ------------------------------------------------------------

points = np.array(
    points,
    dtype=np.float32
)

X = points[:, :3]
Y = points[:, 3]

print(
    f"Feature shape: {X.shape}"
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(3,)
    ),

    tf.keras.layers.Dense(
        32,
        activation="tanh"
    ),

    tf.keras.layers.Dense(
        32,
        activation="tanh"
    ),

    tf.keras.layers.Dense(
        16,
        activation="tanh"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="mse",

    metrics=["mae"]

)

model.summary()

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

history = model.fit(

    X,
    Y,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=0

)

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

preds = model.predict(
    X,
    verbose=0
).flatten()

errors = np.abs(
    preds - Y
) * 100

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
# SAVE KERAS MODEL
# ------------------------------------------------------------

model.save(
    "output/ranknet.keras"
)

print(
    "\nSaved → output/ranknet.keras"
)

# ------------------------------------------------------------
# EXPORT TENSORFLOW.JS
# ------------------------------------------------------------

print(
    "\nExporting TensorFlow.js model..."
)

os.system(

    "tensorflowjs_converter "
    "--input_format=keras "
    "output/ranknet.keras "
    "output/tfjs_model"

)

print(
    "Saved → output/tfjs_model/"
)

print(
    "\nTraining complete."
)
