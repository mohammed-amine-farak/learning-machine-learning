import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

NUM_ROWS = 10000

rows = []

for _ in range(NUM_ROWS):

    age = random.randint(18, 85)

    gender = random.choice(["Male", "Female"])

    bmi = round(np.random.normal(26, 4), 1)

    smoker = random.choice([0, 1])

    family_history = random.choice([0, 1])

    caffeine = max(0, int(np.random.normal(180, 90)))

    stress = random.randint(1, 10)

    sleep = round(random.uniform(4, 9), 1)

    beta_blocker = random.choice([0, 1])

    # احتمال الإصابة
    probability = 0.20

    if family_history:
        probability += 0.10

    if caffeine > 300:
        probability += 0.10

    if stress >= 8:
        probability += 0.10

    if sleep < 5.5:
        probability += 0.05

    svt = 1 if random.random() < probability else 0

    ##################################################
    # Features depending on SVT
    ##################################################

    if svt:

        heart_rate = random.randint(150, 240)

        systolic = random.randint(95, 140)

        diastolic = random.randint(60, 90)

        oxygen = random.randint(94, 100)

        duration = random.randint(2, 60)

        episodes = random.randint(1, 10)

        chest_pain = random.choices([0,1],[0.6,0.4])[0]

        dizziness = random.choices([0,1],[0.2,0.8])[0]

        sob = random.choices([0,1],[0.3,0.7])[0]

        fainting = random.choices([0,1],[0.85,0.15])[0]

        ecg = 1

    else:

        heart_rate = random.randint(60, 100)

        systolic = random.randint(100, 135)

        diastolic = random.randint(65, 85)

        oxygen = random.randint(97, 100)

        duration = 0

        episodes = 0

        chest_pain = random.choices([0,1],[0.95,0.05])[0]

        dizziness = random.choices([0,1],[0.9,0.1])[0]

        sob = random.choices([0,1],[0.95,0.05])[0]

        fainting = 0

        ecg = random.choices([0,1],[0.97,0.03])[0]

    rows.append([
        age,
        gender,
        bmi,
        smoker,
        family_history,
        caffeine,
        stress,
        sleep,
        beta_blocker,
        heart_rate,
        systolic,
        diastolic,
        oxygen,
        duration,
        episodes,
        chest_pain,
        dizziness,
        sob,
        fainting,
        ecg,
        svt
    ])

columns = [
    "age",
    "gender",
    "bmi",
    "smoker",
    "family_history",
    "caffeine_mg_day",
    "stress_level",
    "sleep_hours",
    "beta_blocker",
    "heart_rate_bpm",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_sat",
    "episode_duration_min",
    "episodes_per_month",
    "chest_pain",
    "dizziness",
    "shortness_of_breath",
    "fainting",
    "ecg_abnormal",
    "svt"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("svt_dataset_10000.csv", index=False)

print(df.head())
print(df["svt"].value_counts())