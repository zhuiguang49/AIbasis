import numpy as np

# 创建一个从1到20的NumPy一维数组
array = np.arange(1, 21)

# 筛选出偶数
even_numbers = array[array % 2 == 0]
print("从1到20的偶数：", even_numbers)
# 计算均值
mean_value = np.mean(array)
print("均值:", mean_value)

# 计算标准差
std_dev = np.std(array)
print("标准差:", std_dev)

# 计算最大值
max_value = np.max(array)
print("最大值:", max_value)

# 计算最小值
min_value = np.min(array)
print("最小值:", min_value)
# 创建一个2x5的二维数组
array_2d = np.arange(1, 11).reshape(2, 5)
print("原始二维数组:\n", array_2d)

# 将二维数组形状变换为5x2
reshaped_array = array_2d.reshape(5, 2)
print("变换后的数组:\n", reshaped_array)
# 创建两个二维数组
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# 进行矩阵乘法
matrix_product = np.dot(matrix_a, matrix_b)
print("矩阵乘法结果:\n", matrix_product)
# 生成4行3列的随机数组
random_array = np.random.rand(4, 3)
print("随机数组:\n", random_array)
