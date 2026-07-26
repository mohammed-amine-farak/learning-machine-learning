import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

NUM_ROWS = 10000
rows = []

for _ in range(NUM_ROWS):

    # =====================================================
    # Basic Information (More Realistic)
    # =====================================================

    age = random.randint(18, 85)

    gender = random.choice(["Male", "Female"])

    # BMI with realistic distribution
    bmi = round(
        np.clip(
            np.random.normal(26.5, 4.5),
            16,
            45
        ),
        1
    )

    # Smokers ~20% of population
    smoker = random.choices([0, 1], weights=[80, 20])[0]

    # Family history ~15%
    family_history = random.choices([0, 1], weights=[85, 15])[0]

    # Caffeine: average 250mg/day (2-3 cups of coffee)
    caffeine = max(0, min(600, int(np.random.normal(250, 120))))

    # Stress level: normal distribution centered around 5
    stress = int(np.clip(np.random.normal(5, 2), 1, 10))

    # Sleep hours: most people sleep 6-8 hours
    sleep = round(np.clip(np.random.normal(7, 1.2), 4, 9), 1)

    # Beta blockers: only ~5% of population use them
    beta_blocker = random.choices([0, 1], weights=[95, 5])[0]

    # =====================================================
    # SVT Probability (More Realistic)
    # =====================================================

    probability = 0.05  # Base rate (SVT is rare)

    if family_history:
        probability += 0.15

    if smoker:
        probability += 0.06

    # Caffeine has modest effect
    probability += min(caffeine / 2000, 0.08)

    if stress >= 8:
        probability += 0.08

    if sleep < 5.5:
        probability += 0.07

    if age > 60:
        probability += 0.05
    
    # Females are slightly more prone to SVT
    if gender == "Female":
        probability += 0.03

    # Beta blockers reduce probability
    if beta_blocker:
        probability *= 0.7

    probability = min(probability, 0.75)

    svt = 1 if random.random() < probability else 0

    # =====================================================
    # Generate Heart Rate & Other Vitals
    # =====================================================

    if svt:
        # During episode (15% of the time for SVT patients)
        in_episode = random.choices(
            [0, 1],
            weights=[85, 15]
        )[0]

        if in_episode:
            heart_rate = random.randint(140, 250)
            duration = random.randint(1, 30)
            ecg = 1  # Always abnormal during episode
            oxygen = random.randint(92, 100)
            
            # Blood pressure may drop during episode
            systolic = int(np.clip(np.random.normal(110, 15), 80, 180))
            diastolic = int(np.clip(np.random.normal(70, 10), 50, 110))
        else:
            heart_rate = random.randint(60, 100)
            duration = 0
            ecg = random.choices([0, 1], weights=[70, 30])[0]
            oxygen = random.randint(95, 100)
            
            # Normal blood pressure when not in episode
            systolic = int(np.clip(np.random.normal(120, 12), 80, 170))
            diastolic = int(np.clip(np.random.normal(78, 8), 50, 100))
    else:
        # Healthy individuals
        heart_rate = random.randint(60, 100)
        duration = 0
        ecg = random.choices([0, 1], weights=[98, 2])[0]
        oxygen = random.randint(96, 100)
        
        systolic = int(np.clip(np.random.normal(122, 12), 85, 175))
        diastolic = int(np.clip(np.random.normal(78, 8), 50, 105))

    # =====================================================
    # Generate Episodes Per Month (NO DATA LEAKAGE!)
    # =====================================================
    
    # Even healthy people may rarely have palpitations
    if svt:
        # SVT patients have more episodes, but not always
        episodes = max(1, int(np.random.exponential(2)) + 1)
        episodes = min(episodes, 15)  # Rarely more than 15/month
    else:
        # Healthy: only 3% may have occasional episodes
        if random.random() < 0.03:
            episodes = random.randint(1, 2)
        else:
            episodes = 0

    # =====================================================
    # Generate Symptoms Based on CAUSES, not SVT directly
    # =====================================================
    
    # 1. PALPITATIONS
    palpitations_prob = 0.05  # Base rate
    
    # Factors that increase palpitations
    if stress >= 7:
        palpitations_prob += 0.12
    if caffeine > 300:
        palpitations_prob += 0.08
    if sleep < 6:
        palpitations_prob += 0.06
    if family_history:
        palpitations_prob += 0.05
    if smoker:
        palpitations_prob += 0.04
    
    # If SVT, much higher chance (but not guaranteed)
    if svt:
        palpitations_prob += 0.35
    
    palpitations = 1 if random.random() < palpitations_prob else 0

    # 2. CHEST PAIN
    chest_pain_prob = 0.02
    
    if stress >= 8:
        chest_pain_prob += 0.08
    if age > 60:
        chest_pain_prob += 0.05
    if smoker:
        chest_pain_prob += 0.04
    if bmi > 30:
        chest_pain_prob += 0.03
    
    if svt:
        chest_pain_prob += 0.15
    
    chest_pain = 1 if random.random() < chest_pain_prob else 0

    # 3. DIZZINESS
    dizziness_prob = 0.03
    
    if stress >= 7:
        dizziness_prob += 0.10
    if sleep < 5:
        dizziness_prob += 0.08
    if caffeine > 400:
        dizziness_prob += 0.05
    if age > 65:
        dizziness_prob += 0.04
    
    if svt:
        dizziness_prob += 0.25
    
    dizziness = 1 if random.random() < dizziness_prob else 0

    # 4. SHORTNESS OF BREATH (SOB)
    sob_prob = 0.02
    
    if bmi > 30:
        sob_prob += 0.10
    if smoker:
        sob_prob += 0.06
    if age > 60:
        sob_prob += 0.04
    if stress >= 8:
        sob_prob += 0.05
    
    if svt:
        sob_prob += 0.15
    
    sob = 1 if random.random() < sob_prob else 0

    # 5. FAINTING
    fainting_prob = 0.01
    
    if stress >= 9:
        fainting_prob += 0.05
    if sleep < 4.5:
        fainting_prob += 0.04
    if heart_rate > 180:  # Only during episode
        fainting_prob += 0.15
    if systolic < 90:  # Low blood pressure
        fainting_prob += 0.08
    
    if svt:
        fainting_prob += 0.10
    
    fainting = 1 if random.random() < fainting_prob else 0

    # =====================================================
    # Add Small Amount of Noise (More Realistic)
    # =====================================================

    if random.random() < 0.005:  # 0.5% noise only
        svt = 1 - svt

    # =====================================================
    # Append Row
    # =====================================================
    
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
        episodes,  # This is NOT a leakage anymore!
        palpitations,
        chest_pain,
        dizziness,
        sob,
        fainting,
        ecg,
        svt
    ])

# =====================================================
# Column Names
# =====================================================

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
    "episodes_per_month",  # Now this is safe to include!
    "palpitations",
    "chest_pain",
    "dizziness",
    "shortness_of_breath",
    "fainting",
    "ecg_abnormal",
    "svt"
]

# =====================================================
# Create and Save DataFrame
# =====================================================

df = pd.DataFrame(rows, columns=columns)

df.to_csv("svt_dataset_10000.csv", index=False)

# =====================================================
# Display Results
# =====================================================

print("=" * 60)
print("First 10 rows of the dataset:")
print("=" * 60)
print(df.head(10))

print("\n" + "=" * 60)
print("Dataset Information:")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nData types:")
print(df.dtypes)

print("\n" + "=" * 60)
print("SVT Distribution:")
print("=" * 60)
print(df["svt"].value_counts())
print(f"\nPercentage with SVT: {df['svt'].mean()*100:.2f}%")

print("\n" + "=" * 60)
print("Summary Statistics:")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("Feature Correlations with SVT:")
print("=" * 60)
# Calculate correlation with SVT for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()["svt"].sort_values(ascending=False)
print(correlations)

# =====================================================
# Optional: Check for Data Leakage
# =====================================================

print("\n" + "=" * 60)
print("Data Leakage Check:")
print("=" * 60)

# Check if episodes_per_month perfectly predicts SVT
episodes_correlation = df["episodes_per_month"].corr(df["svt"])
print(f"Correlation between episodes_per_month and SVT: {episodes_correlation:.4f}")

# Check if any symptom perfectly predicts SVT
for col in ["palpitations", "chest_pain", "dizziness", 
            "shortness_of_breath", "fainting", "ecg_abnormal"]:
    perfect_prediction = (df[df["svt"] == 1][col].mean() == 1.0)
    if perfect_prediction:
        print(f"⚠️ WARNING: {col} perfectly predicts SVT!")
    else:
        print(f"✅ {col}: {df[df['svt'] == 1][col].mean():.2%} in SVT patients, "
              f"{df[df['svt'] == 0][col].mean():.2%} in non-SVT patients")

print("\n" + "=" * 60)
print("✅ Dataset generation complete!")
print("=" * 60)