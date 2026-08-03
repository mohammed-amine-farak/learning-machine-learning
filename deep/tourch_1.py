import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================
# Build Model
# ==========================

class Model(nn.Module):

    def __init__(self, in_feature=4, h1=8, h2=9, out_feature=3):
        super().__init__()

        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_feature)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


# ==========================
# Random Seed
# ==========================

torch.manual_seed(32)

model = Model()

# ==========================
# Load Dataset
# ==========================

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"

my_df = pd.read_csv(url)

# Convert labels to integers
my_df["species"] = my_df["species"].map({
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}).astype(int)

# ==========================
# Features & Labels
# ==========================

X = my_df.drop("species", axis=1).values
y = my_df["species"].values

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=32
)

# ==========================
# Convert to Tensor
# ==========================

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# ==========================
# Loss Function
# ==========================

criterion = nn.CrossEntropyLoss()

# ==========================
# Optimizer
# ==========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ==========================
# Training
# ==========================

epochs = 1000
losses = []

for epoch in range(epochs):

    # Forward
    y_pred = model(X_train)

    # Calculate Loss
    loss = criterion(y_pred, y_train)

    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.4f}")

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================
# Plot Loss
# ==========================

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ==========================
# Test Accuracy
# ==========================

with torch.no_grad():

    y_eval = model(X_test)

    predicted = torch.argmax(y_eval, dim=1)

    correct = (predicted == y_test).sum().item()

    accuracy = correct / len(y_test)

print(f"\nAccuracy = {accuracy * 100:.2f}%")

# ==========================
# Predict One Flower
# ==========================

# One flower (must have shape [1,4])
test = torch.FloatTensor([[5.7,3.0,4.2,1.2]])

classes = ["Setosa", "Versicolor", "Virginica"]

model.eval()

with torch.no_grad():

    output = model(test)

    print("\nRaw Output:")
    print(output)

    prediction = torch.argmax(output, dim=1)

    print("\nPredicted Class Number:", prediction.item())
    print("Predicted Flower:", classes[prediction.item()])