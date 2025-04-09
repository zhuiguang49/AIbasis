import numpy as np

# RNN 参数
W1, W2, W3 = 0.4, 0.5, 0.3  # 输入权重、隐含层权重、输出权重
alpha, beta = 0, 0.1         # 偏置参数
S_prev = 0                   # 初始隐藏状态
input_sequence = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]  # 更新后的输入序列

# 激活函数 ReLU
def relu(x):
    return max(0, x)

# 计算 RNN 输出序列
outputs = []
for x_t in input_sequence:
    # 隐藏状态更新
    S_t = relu(W1 * sum(x_t) + W2 * S_prev + alpha)
    # 输出计算
    Y_t = W3 * S_t + beta
    outputs.append(Y_t)
    S_prev = S_t  # 更新隐藏状态

print("RNN 输出序列:", outputs)
