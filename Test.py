import  numpy as np

arr = np.arange(3).reshape(3,1)
print(arr)
print(arr.squeeze())
print(arr.squeeze().shape)