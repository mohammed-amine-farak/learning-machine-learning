import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("crane_safety_10000.csv")

X = df.drop(
    columns=[
        "risk_score",
        "safety_status"
    ]
)
y = df["safety_status"]

le_crane = LabelEncoder()
le_load = LabelEncoder()

X["crane_type"] = le_crane.fit_transform(X["crane_type"])
X["load_type"] = le_load.fit_transform(X["load_type"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
    random_state=42
)


model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(prediction)

accuracy = model.score(X_test, y_test)


cm = confusion_matrix(y_test, prediction)


print("Accuracy:", accuracy)