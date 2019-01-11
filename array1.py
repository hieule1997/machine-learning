import numpy as np
# tạo ma trân chéo 
# [[0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 2. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 3. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 4. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 5. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 6. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 7. 0.]]
# n = 8
# a = np.eye(n, k= -1)
# for i in range(n):
#     for j in range(n):
#         if(i-1 == j):
#             a[i,j] = i
# print (a)


# tổng các đơn vị trong ma trận có chỉ số chẵn

# a = a = np.floor(10*np.random.random((3,4)))
# b = 0
# for i in range(a.shape[1]):
#     if(i%2 == 0):
#         for j in range(a.shape[0]):
#             print(a[j][i])
#             b += a[j][i]

# print(a)
# print (b)
A = np.arange(1,13).reshape(3, 4)
B = np.transpose(A)
print (B)

