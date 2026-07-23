import numpy as np

# =====================================
# Dataset
# =====================================

X = np.array([
    [2, 55],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85],
    [9, 90]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=float)

# =====================================
# Feature Scaling
# =====================================

X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)

X_scaled = (X - X_min) / (X_max - X_min)

# =====================================
# Initialize Parameters
# =====================================

np.random.seed(42)

W = np.random.randn(2) * 0.01
b = 0.0

learning_rate = 0.1
epochs = 100000

# =====================================
# Sigmoid
# =====================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =====================================
# Training
# =====================================

for epoch in range(epochs):

    # Forward
    z = X_scaled @ W + b
    predictions = sigmoid(z)

    # Cross Entropy Loss
    loss = -np.mean(
        y * np.log(predictions + 1e-10)
        +
        (1 - y) * np.log(1 - predictions + 1e-10)
    )

    # Gradients
    gradient_W = (X_scaled.T @ (predictions - y)) / len(X_scaled)

    gradient_b = np.mean(predictions - y)

    # Update
    W -= learning_rate * gradient_W
    b -= learning_rate * gradient_b

    if epoch % 500 == 0:
        print(f"Epoch {epoch}")
        print("Loss =", loss)
        print("W =", W)
        print("b =", b)
        print("----------------------------")

# =====================================
# Final Prediction
# =====================================

probabilities = sigmoid(X_scaled @ W + b)

predicted_classes = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predicted_classes == y) * 100

print("\nFinal Weights")
print(W)

print("\nFinal Bias")
print(b)

print("\nProbabilities")
print(probabilities)

print("\nPredicted Classes")
print(predicted_classes)

print("\nAccuracy")
print(f"{accuracy:.2f}%")

# =====================================
# Predict New Student
# =====================================

new_student = np.array([
    6,
    78
], dtype=float)

new_student_scaled = (
    new_student - X_min
) / (X_max - X_min)

probability = sigmoid(new_student_scaled @ W + b)

prediction = 1 if probability >= 0.5 else 0

print("\n========================")
print("New Student Prediction")
print("========================")

print("Probability =", probability)

if prediction == 1:
    print("Prediction : PASS")
else:
    print("Prediction : FAIL")