import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Training Data
# ==========================

X = np.array([
    [55, 2, 18, 12],
    [70, 3, 15, 10],
    [85, 3, 12, 8],
    [100, 4, 10, 7]
], dtype=float)

y_true = np.array([
    135000,
    165000,
    195000,
    225000
], dtype=float)

# ==========================
# Initial Weights
# ==========================

W = np.random.randn(4) * 0.01

b = 10000

learning_rate = 0.01
epochs = 10000

X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

print(X_scaled)
# ==========================
# Training
# ==========================

for epoch in range(epochs):

    # Prediction
    predictions = X_scaled @ W + b

    # Error
    errors = y_true - predictions

    # Mean Squared Error
    mse = np.mean(errors ** 2)

    # Gradient of W
    gradient_W = (-2 / len(X_scaled)) * (X_scaled.T @ errors)

    # Gradient of b
    gradient_b = (-2 / len(X_scaled)) * np.sum(errors)

    # Update weights
    W = W - learning_rate * gradient_W

    # Update bias
    b = b - learning_rate * gradient_b

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        print("MSE =", mse)
        print("W =", W)
        print("b =", b)
        print("---------------------")

# ==========================
# Final Results
# ==========================

print("\nFinal Weights:")
print(W)
new_house = np.array([
    90,
    3,
    9,
    7
], dtype=float)

new_house_scaled = (new_house - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
prediction = new_house_scaled @ W + b
print(prediction)
print("\nFinal Bias:")
print(b)

print("\nPredictions:")
print(predictions)

print("\nFinal MSE:")
print(mse)
