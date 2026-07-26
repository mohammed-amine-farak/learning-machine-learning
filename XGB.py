# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# ==========================================
# 2. Read Dataset
# ==========================================

df = pd.read_csv("svt_dataset_10000.csv")

# ==========================================
# 3. Split Features and Target
# ==========================================

X = df.drop(["svt", "episodes_per_month"], axis=1)
y = df["svt"]

# ==========================================
# 4. Encode Text Columns
# ==========================================

gender_encoder = LabelEncoder()

X["gender"] = gender_encoder.fit_transform(X["gender"])

# ==========================================
# 5. Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# 6. Create Model
# ==========================================

model = XGBClassifier(

    n_estimators=200,

    learning_rate=0.01,

    max_depth=5,

    random_state=42,

    eval_metric="logloss"

)

# ==========================================
# 7. Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 8. Prediction
# ==========================================

prediction = model.predict(X_test)

# ==========================================
# 9. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy")

print(accuracy)

# ==========================================
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    prediction
)

print("\nConfusion Matrix")

print(cm)

# ==========================================
# 11. Classification Report
# ==========================================

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

# ==========================================
# 12. Feature Importance
# ==========================================

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance_df)

# ==========================================
# 13. Plot Feature Importance
# ==========================================

plt.figure(figsize=(10,6))

plt.bar(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xticks(rotation=90)

plt.title("Feature Importance")

plt.tight_layout()

plt.show()

# ==========================================
# 14. Predict New Patient
# ==========================================

new_patient = pd.DataFrame([{

    "age":22,

    "gender":"Male",

    "bmi":25.5,

    "smoker":0,

    "family_history":1,

    "caffeine_mg_day":350,

    "stress_level":9,

    "sleep_hours":4.5,

    "beta_blocker":0,

    "heart_rate_bpm":190,

    "systolic_bp":110,

    "diastolic_bp":70,

    "oxygen_sat":98,

    "episode_duration_min":15,

    "episodes_per_month":5,

    "chest_pain":1,

    "dizziness":1,

    "shortness_of_breath":1,

    "fainting":0,

    "ecg_abnormal":1

}])

# Encode gender

new_patient["gender"] = gender_encoder.transform(
    new_patient["gender"]
)

# Prediction

result = model.predict(new_patient)

print("\nPrediction")

if result[0] == 1:

    print("SVT")

else:

    print("Normal")