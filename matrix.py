X = [
    [50, 2, 20, 15],
    [65, 3, 15, 12],
    [80, 3, 10, 10],
    [95, 4, 8, 8],
    [110, 4, 6, 6],
    [125, 5, 5, 5],
    [140, 5, 4, 4],
    [160, 6, 2, 3]
]
W = [
    1000,    # Size
    20000,   # Rooms
    -1500,   # Age
    -3000    # Distance
]
predictions = []

# Matrix × Vector
for row in X:

    prediction = 0

    for i in range(len(W)):
        prediction += row[i] * W[i]

    predictions.append(prediction)

print(predictions)