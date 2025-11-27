import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)

# 设置matplotlib使用英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义字符集
chars = ['d', 'l', 'e', 'a', 'r', 'n']
char2idx = {char: i for i, char in enumerate(chars)}
idx2char = {i: char for i, char in enumerate(chars)}

# 参数设置
input_size = len(chars)
hidden_size = 12
batch_size = 1
learning_rate = 0.1
num_epochs = 100

# 准备数据 - 修复性能警告
x_data = [char2idx[ch] for ch in "dlearn"]
y_data = [char2idx[ch] for ch in "lanrla"]

# 使用numpy.array()避免性能警告
one_hot_lookup = np.eye(input_size)
x_one_hot = np.array([one_hot_lookup[x] for x in x_data])
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)

print("Sequence learning: 'dlearn' -> 'lanrla'")

# 存储训练数据
history = {
    'rnncell_loss': [],
    'rnn_loss': [],
    'rnncell_preds': [],
    'rnn_preds': []
}


# 1. RNNCell 模型
class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.rnncell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return self.fc(hidden), hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


# 2. RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), hidden_size)
        out, _ = self.rnn(x, h0)
        return self.fc(out)


# 初始化模型和优化器
rnncell_model = RNNCellModel(input_size, hidden_size, batch_size)
rnn_model = RNNModel(input_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer_cell = torch.optim.Adam(rnncell_model.parameters(), lr=learning_rate)
optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

# 训练 RNNCell
print("\nTraining RNNCell...")
for epoch in range(num_epochs):
    loss = 0
    optimizer_cell.zero_grad()
    hidden = rnncell_model.init_hidden()

    for input, label in zip(inputs, labels):
        output, hidden = rnncell_model(input, hidden)
        loss += criterion(output, label)

    loss.backward()
    optimizer_cell.step()
    history['rnncell_loss'].append(loss.item())

    if (epoch + 1) % 20 == 0:
        hidden = rnncell_model.init_hidden()
        pred_chars = []
        for input in inputs:
            output, hidden = rnncell_model(input, hidden)
            _, idx = output.max(1)
            pred_chars.append(idx2char[idx.item()])
        pred_str = ''.join(pred_chars)
        history['rnncell_preds'].append((epoch + 1, pred_str))
        print(f'Epoch {epoch + 1:3d}, Loss: {loss.item():.4f}, Pred: "{pred_str}"')

# 训练 RNN
print("\nTraining RNN...")
for epoch in range(num_epochs):
    optimizer_rnn.zero_grad()
    outputs = rnn_model(inputs)
    loss = criterion(outputs.view(-1, input_size), labels.view(-1))

    loss.backward()
    optimizer_rnn.step()
    history['rnn_loss'].append(loss.item())

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            outputs = rnn_model(inputs)
            _, pred_indices = outputs.max(2)
            pred_chars = [idx2char[idx.item()] for idx in pred_indices.view(-1)]
            pred_str = ''.join(pred_chars)
            history['rnn_preds'].append((epoch + 1, pred_str))
            print(f'Epoch {epoch + 1:3d}, Loss: {loss.item():.4f}, Pred: "{pred_str}"')

# 绘制损失函数 - 使用英文标题
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['rnncell_loss'], label='RNNCell', linewidth=2)
plt.plot(history['rnn_loss'], label='RNN', linewidth=2)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(history['rnncell_loss'], label='RNNCell', linewidth=2)
plt.semilogy(history['rnn_loss'], label='RNN', linewidth=2)
plt.title('Loss Curve (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss (log)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最终预测结果
print("\nFinal Results:")
with torch.no_grad():
    # RNNCell 预测
    hidden = rnncell_model.init_hidden()
    cell_pred = []
    for input in inputs:
        output, hidden = rnncell_model(input, hidden)
        _, idx = output.max(1)
        cell_pred.append(idx2char[idx.item()])

    # RNN 预测
    outputs = rnn_model(inputs)
    _, rnn_indices = outputs.max(2)
    rnn_pred = [idx2char[idx.item()] for idx in rnn_indices.view(-1)]

print(f"RNNCell: '{''.join(cell_pred)}'")
print(f"RNN:     '{''.join(rnn_pred)}'")
print(f"Target:  'lanrla'")
print(f"RNNCell Correct: {'Yes' if ''.join(cell_pred) == 'lanrla' else 'No'}")
print(f"RNN Correct:     {'Yes' if ''.join(rnn_pred) == 'lanrla' else 'No'}")

# 打印训练历史
print("\nTraining History:")
print("RNNCell predictions:", history['rnncell_preds'])
print("RNN predictions:", history['rnn_preds'])