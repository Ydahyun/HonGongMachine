import numpy as np

test_arr = np.array([1,2,3,4])
print(test_arr)
print(test_arr.shape)

test_arr_reshape1 = test_arr.reshape(2,2)
print(test_arr_reshape1)
print(test_arr_reshape1.shape)

test_arr_reshape2 = test_arr.reshape(-1,1)
print(test_arr_reshape2)
print(test_arr_reshape2.shape)