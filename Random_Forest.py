# ======================================================
# 1. استيراد المكتبات
# ======================================================

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ======================================================
# 2. قراءة البيانات
# ======================================================

df = pd.read_csv("crane_safety_10000.csv")


# ======================================================
# 3. فصل المدخلات والمخرجات
# ======================================================

X = df.drop(
    columns=[
        "risk_score",
        "safety_status"
    ]
)

y = df["safety_status"]


# ======================================================
# 4. تحويل النصوص إلى أرقام
# ======================================================

le_crane = LabelEncoder()
le_load = LabelEncoder()

X["crane_type"] = le_crane.fit_transform(
    X["crane_type"]
)

X["load_type"] = le_load.fit_transform(
    X["load_type"]
)


# ======================================================
# 5. تقسيم البيانات
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


# ======================================================
# 6. إنشاء نموذج Random Forest
# ======================================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)


# ======================================================
# 7. تدريب النموذج
# ======================================================

model.fit(

    X_train,

    y_train

)


# ======================================================
# 8. التنبؤ
# ======================================================

prediction = model.predict(

    X_test

)


# ======================================================
# 9. حساب الدقة
# ======================================================

accuracy = model.score(

    X_test,

    y_test

)


# ======================================================
# 10. أهمية كل متغير
# ======================================================

importance = model.feature_importances_

for feature, value in zip(

        X.columns,

        importance

):

    print(feature, value)


# ======================================================
# 11. مصفوفة الأخطاء
# ======================================================

cm = confusion_matrix(

    y_test,

    prediction

)

print(cm)


# ======================================================
# 12. التقرير الكامل
# ======================================================

print(

    classification_report(

        y_test,

        prediction

    )

)


# ======================================================
# 13. الدقة النهائية
# ======================================================

print(

    "Accuracy:",

    accuracy

)