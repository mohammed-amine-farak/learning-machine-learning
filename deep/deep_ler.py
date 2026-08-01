import torch
import torch.nn as nn
import torch.optim as optim

# ============================
# البيانات
# ============================

X = torch.tensor([
    [22., 190., 9., 98.],
    [45., 170., 8., 97.],
    [30., 80., 2., 99.]
], dtype=torch.float32)

y = torch.tensor([
    [1.],
    [1.],
    [0.]
], dtype=torch.float32)

# ============================
# بناء الشبكة العصبية
# ============================

model = nn.Sequential(

    nn.Linear(4, 8),   # 4 مدخلات → 8 خلايا عصبية

    nn.ReLU(),

    nn.Linear(8, 1),   # 8 خلايا → مخرج واحد

    nn.Sigmoid()

)

# ============================
# دالة الخطأ
# ============================

criterion = nn.BCELoss()

# ============================
# Optimizer
# ============================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ============================
# التدريب
# ============================

for epoch in range(500):

    prediction = model(X)
    
    loss = criterion(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}  Loss = {loss.item():.4f}"
        )

# ============================
# الاختبار
# ============================

print("\nPrediction")

print(model(X))
