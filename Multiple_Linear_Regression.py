houses = [
    {"size": 55,  "rooms": 2, "age": 18, "distance": 12, "price": 135000},
    {"size": 70,  "rooms": 3, "age": 15, "distance": 10, "price": 165000},
    {"size": 85,  "rooms": 3, "age": 12, "distance": 8,  "price": 195000},
    {"size": 100, "rooms": 4, "age": 10, "distance": 7,  "price": 225000},
    {"size": 115, "rooms": 4, "age": 8,  "distance": 6,  "price": 255000},
    {"size": 130, "rooms": 5, "age": 6,  "distance": 5,  "price": 290000},
    {"size": 145, "rooms": 5, "age": 4,  "distance": 4,  "price": 325000},
    {"size": 160, "rooms": 6, "age": 2,  "distance": 3,  "price": 360000},
]

# Initial weights
W_size = 0
W_rooms = 0
W_age = 0
W_distance = 0
b = 0

learning_rate = 0.000001
epochs = 1000

for epoch in range(epochs):

    size_gradient_list = []
    rooms_gradient_list = []
    age_gradient_list = []
    distance_gradient_list = []
    bias_gradient_list = []
    squared_errors = []

    for house in houses:

        # Prediction
        prediction = (
            house["size"] * W_size +
            house["rooms"] * W_rooms +
            house["age"] * W_age +
            house["distance"] * W_distance +
            b
        )

        # Error
        error = house["price"] - prediction

        # MSE
        squared_errors.append(error ** 2)

        # Gradients
        size_gradient_list.append(house["size"] * error)
        rooms_gradient_list.append(house["rooms"] * error)
        age_gradient_list.append(house["age"] * error)
        distance_gradient_list.append(house["distance"] * error)
        bias_gradient_list.append(error)

    n = len(houses)

    mse = sum(squared_errors) / n

    gradient_size = (-2 / n) * sum(size_gradient_list)
    gradient_rooms = (-2 / n) * sum(rooms_gradient_list)
    gradient_age = (-2 / n) * sum(age_gradient_list)
    gradient_distance = (-2 / n) * sum(distance_gradient_list)
    gradient_b = (-2 / n) * sum(bias_gradient_list)

    # Update weights
    W_size = W_size - learning_rate * gradient_size
    W_rooms = W_rooms - learning_rate * gradient_rooms
    W_age = W_age - learning_rate * gradient_age
    W_distance = W_distance - learning_rate * gradient_distance
    b = b - learning_rate * gradient_b

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")
        print(f"MSE: {mse:.2f}")
        print(f"W_size = {W_size:.4f}")
        print(f"W_rooms = {W_rooms:.4f}")
        print(f"W_age = {W_age:.4f}")
        print(f"W_distance = {W_distance:.4f}")
        print(f"b = {b:.4f}")
        print("-" * 40)

print("\nTraining Finished\n")

print("Final Weights")
print("W_size =", W_size)
print("W_rooms =", W_rooms)
print("W_age =", W_age)
print("W_distance =", W_distance)
print("b =", b)