import numpy as np
import matplotlib.pyplot as plt
# 生成随机数数据
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 100  # 随机点大小
colors = np.random.rand(50)       # 随机颜色


plt.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis')

# 添加标题和轴标签
plt.title('Scatter Plot Example')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 显示颜色条
plt.colorbar()

# 显示图像
plt.show()
