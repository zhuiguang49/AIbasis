import numpy as np

# 苹果的特征向量
apple = np.array([1, 1, 0, 1])
# 大米的特征向量
rice = np.array([0, 1, 1, 1])

# 计算余弦相似度
cosine_sim = np.dot(apple, rice) / (np.sqrt(np.sum(apple**2)) * np.sqrt(np.sum(rice**2)))
print(f"余弦相似度: {cosine_sim:.4f}")  # 应该约等于 0.5774

# 计算广义Jaccard相似度
intersection = np.sum(np.minimum(apple, rice))
union = np.sum(np.maximum(apple, rice))
jaccard_sim = intersection / union
print(f"广义Jaccard相似度: {jaccard_sim:.4f}")  # 应该约等于 0.5000