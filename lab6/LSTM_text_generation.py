import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 数据加载与预处理
data = open('./text.txt', 'r').read().replace('\n', '').replace('\r', '')

# 创建字符字典
chars = list(set(data))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
num_chars = len(chars)

# 数据序列化
sequence_length = 20  # 滑动窗口长度
sequences = []
next_chars = []

for i in range(len(data) - sequence_length):
    sequences.append(data[i:i + sequence_length])
    next_chars.append(data[i + sequence_length])

# 转换字符到整数
X = np.array([[char_to_idx[char] for char in seq] for seq in sequences])
y = np.array([char_to_idx[char] for char in next_chars])

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 构建 PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 构建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                torch.zeros(num_layers, batch_size, hidden_size).to(device))

# 模型参数
embed_size = 128
hidden_size = 256
num_layers = 2
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel(num_chars, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
epochs = 20
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 动态初始化隐藏状态
        hidden = model.init_hidden(batch_size=X_batch.size(0))

        optimizer.zero_grad()
        outputs, hidden = model(X_batch, hidden)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    losses.append(epoch_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# 绘制损失曲线并保存
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 保存为本地文件 (PNG 格式)
plt.savefig("training_loss_curve.png")
plt.show()


# 验证模型
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        hidden = model.init_hidden(batch_size=X_batch.size(0))
        outputs, _ = model(X_batch, hidden)
        predicted = torch.argmax(outputs, dim=1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# 文本生成函数
def generate_text(model, start_text, length, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([char_to_idx[char] for char in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(batch_size=1)

    generated_text = start_text
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            output_dist = output.squeeze().div(temperature).exp()
            next_char_idx = torch.multinomial(output_dist, 1).item()
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char

        input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    return generated_text

# 测试文本生成
start_text = "Once upon a time"
for temp in [0.5, 1.0, 1.5]:
    generated_text = generate_text(model, start_text, length=200, temperature=temp)
    print(f"\nGenerated Text with Temperature {temp}:\n{generated_text}\n")
