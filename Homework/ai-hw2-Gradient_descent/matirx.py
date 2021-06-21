import numpy as np

# 初始化A矩阵
A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])

print('矩阵 A：')
print(A)

# 求A 的逆
A_inv = np.linalg.inv(A)

print('A 的逆：')
print(A_inv)

# 初始化向量b
b = np.array([[3], [-2], [13.5]])
print('向量 b：')
print(b)

# 求解x
print('求解 x：')
x = A_inv.dot(b)
print(x)