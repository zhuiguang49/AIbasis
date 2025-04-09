import numpy as np
from sklearn.preprocessing import StandardScaler

# 数据输入：年龄, 薪资（月）, 债务（万）
X = np.array([[16, 0, 0],
              [24, 5000, 0],
              [23, 6000, 10],
              [28, 5000, 50],
              [40, 15000, 200]])

# 目标输出（0: 否, 1: 是）
y = np.array([0, 1, 1, 0, 1])

# 数据归一化
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

def sigmoid(x):
    x = np.clip(x, -500, 500)  # 限制输入范围，避免溢出
    return 1 / (1 + np.exp(-x))

# 初始化权重和偏置
weights = np.random.rand(3)  # 3个特征
bias = np.random.rand(1)

def gradient_descent(X, y, weights, bias, lr=0.01, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(X)):
            # 预测值
            z = np.dot(X[i], weights) + bias
            y_pred = sigmoid(z)
            # 误差
            error = y_pred - y[i]
            # 更新权重和偏置
            weights -= lr * error * X[i]
            bias -= lr * error
    return weights, bias

def predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        y_pred = sigmoid(z)
        predictions.append(1 if y_pred >= 0.5 else 0)
    return predictions

# 训练模型
weights, bias = gradient_descent(X_normalized, y, weights, bias)

# 预测
predictions = predict(X_normalized, weights, bias)

print("最终权重:", weights)
print("最终偏置:", bias)
print("预测结果:", predictions)
