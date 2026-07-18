from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



farms = [
    {"N": 90, "P": 45, "K": 55, "moisture": 32, "yield": 6.8},
    {"N": 85, "P": 40, "K": 50, "moisture": 30, "yield": 6.3},
    {"N": 70, "P": 35, "K": 45, "moisture": 28, "yield": 5.5},
    {"N": 60, "P": 30, "K": 40, "moisture": 25, "yield": 4.8},
    {"N": 95, "P": 50, "K": 60, "moisture": 35, "yield": 7.2},
    {"N": 75, "P": 38, "K": 48, "moisture": 29, "yield": 5.9},
    {"N": 55, "P": 25, "K": 35, "moisture": 22, "yield": 4.1},
    {"N": 80, "P": 42, "K": 52, "moisture": 31, "yield": 6.1},
    {"N": 65, "P": 32, "K": 42, "moisture": 26, "yield": 5.0},
    {"N": 50, "P": 20, "K": 30, "moisture": 20, "yield": 3.7},
]
X = []
y = []

for f in farms:
    X.append([f["N"], f["P"], f["K"], f["moisture"]])
    y.append(f["yield"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
new_farm = model.predict([[95, 50, 60, 35]])  # Example input for a new farm  
total_error = 0

for i in range(len(y_test)):
    error = y_test[i] - predictions[i]              
    total_error += error ** 2

mse = total_error / len(y_test)

print("Predictions:", predictions)
print("New Farm Prediction:", new_farm)
print("MSE:", mse)