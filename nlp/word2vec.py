import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 1. 词袋模型实践
sentence1 = "我今天去图书馆看书学习"
sentence2 = "我昨天在图书馆学习"

# 使用jieba进行分词
def jieba_cut(sentence):
    return " ".join(jieba.cut(sentence))

sentence1_cut = jieba_cut(sentence1)
sentence2_cut = jieba_cut(sentence2)

print("分词结果：")
print(f"句子1分词: {sentence1_cut}")
print(f"句子2分词: {sentence2_cut}")

# 构建词袋模型
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([sentence1_cut, sentence2_cut])

# 获取词汇表
vocabulary = vectorizer.get_feature_names_out()
print("\n词汇表：", vocabulary)

# 转换为数组以便查看词频
word_freq = X.toarray()
print("\n词频矩阵：")
print(word_freq)

# 计算余弦相似度
cosine_sim = cosine_similarity(X[0:1], X[1:2])
print(f"\n两个句子的余弦相似度为: {cosine_sim[0][0]:.4f}")

# 2. Word2Vec实践
sentences = [
    "我今天去图书馆看书学习",
    "我昨天在图书馆学习",
    "我喜欢在图书馆学习",
    "图书馆是个安静的地方",
    "学习需要专注的环境"
]

# 分词
tokenized_sentences = [list(jieba.cut(sentence)) for sentence in sentences]
print("\n所有句子的分词结果：")
for sentence in tokenized_sentences:
    print(" ".join(sentence))

# 训练Word2Vec模型
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# 计算句子向量
def sentence_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# 获取每个句子的词向量
sentence_vectors = [sentence_vector(sentence) for sentence in tokenized_sentences]

# 计算所有句子间的余弦相似度
cosine_sim_matrix = cosine_similarity(sentence_vectors)

# 可视化结果
plt.figure(figsize=(12, 8))
sns.heatmap(cosine_sim_matrix, annot=True, cmap="coolwarm",
            xticklabels=[f"句子{i+1}\n{sentences[i][:10]}..." for i in range(len(sentences))],
            yticklabels=[f"句子{i+1}\n{sentences[i][:10]}..." for i in range(len(sentences))],
            fmt=".2f")
plt.title("句子余弦相似度热图")
plt.tight_layout()
plt.show()