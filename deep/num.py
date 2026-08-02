import numpy as np

matrix_ints = np.random.randint(10, 100, size=(100, 20))
rng = np.random.default_rng()
w = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 


print(np.mean(matrix_ints, axis=0))
mean = np.mean(matrix_ints, axis=0)
contidtion = mean >=50
print(matrix_ints[:,contidtion])
print(np.std(matrix_ints))
std_column = np.std(matrix_ints, axis=0)
std_column_condition = std_column <= 5
new_matrix = np.delete(matrix_ints, std_column_condition, axis=1)

y = (matrix_ints - matrix_ints.mean(axis=0)) / matrix_ints.std(axis=0)
print(y.shape)
z = y[:5, :] 



print(np.dot(y,w))
