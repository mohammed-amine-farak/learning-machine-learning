houses = [
    {"size": 50,  "price": 120000},
    {"size": 65,  "price": 145000},
    {"size": 80,  "price": 175000},
    {"size": 95,  "price": 210000},
    {"size": 110, "price": 240000},
    {"size": 125, "price": 270000},
    {"size": 140, "price": 305000},
    {"size": 155, "price": 335000},
]

# Initial values
W = 0
b = 0

learning_rate = 0.00001
epochs = 1000

for epoch in range(epochs):

    predictions = []
    errors = []
    errorpower2 = []
    house_error_list = []
    bias_error_list = []

    # Forward pass
    for house in houses:

        prediction = W * house["size"] + b
        error = house["price"] - prediction

        predictions.append(prediction)
        errors.append(error)
        errorpower2.append(error ** 2)

        house_error_list.append(house["size"] * error)
        bias_error_list.append(error)

    # Loss
    mse = sum(errorpower2) / len(houses)

    # Gradients
    gradient_w = (-2 / len(houses)) * sum(house_error_list)
    gradient_b = (-2 / len(houses)) * sum(bias_error_list)

    # Update parameters
    W = W - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")
        print(f"MSE: {mse:.2f}")
        print(f"W: {W:.4f}")
        print(f"b: {b:.4f}")
        print("-" * 30)

print("\nTraining Finished")
print("Final W:", W)
print("Final b:", b)

# Prediction for a new house
new_size = 100
predicted_price = W * new_size + b

print(f"\nHouse Size: {new_size}")
print(f"Predicted Price: {predicted_price:.2f}")