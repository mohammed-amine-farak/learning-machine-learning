import torch
import numpy as np
import pandas as pd
df = pd.read_csv("svt_dataset_10000.csv")
df["gender"] = df["gender"].map({
    "Male": 0,
    "Female": 1
})
X = torch.tensor([
    [22., 190., 9., 98.],
    [45., 170., 8., 97.],
    [30., 80., 2., 99.],
     [30., 80., 2., 99.]
], dtype=torch.float32)
print(X.dim())
y = torch.randn(4,5)
b = X@y

A = torch.rand(5,4)

B = torch.rand(5)

C = A + B

#if you want only own number
#x = torch.tensor(5)
#if you want a lot of number



print(C)


