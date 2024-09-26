import numpy as np
import matplotlib.pyplot as plt
# 创建x轴数据
x = np.linspace(0, 2 * np.pi, 100)

# 计算复合函数值
y = np.sin(x) * np.cos(x)

# 绘制图像
plt.plot(x, y, label='sin(x) * cos(x)', color='green')

# 添加标题和轴标签
plt.title('Composite Function y = sin(x) * cos(x)')
plt.xlabel('x')
plt.ylabel('y')

# 显示网格线和图例
plt.grid(True)
plt.legend()

# 显示图像
plt.show()
