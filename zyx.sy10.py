import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 核心数据预处理（生成模型可输入的数值序列）
char_set = ['d', 'l', 'e', 'a', 'r', 'n']
char2idx = {c: i for i, c in enumerate(char_set)}
idx2char = char_set
input_str = "dlearn"
target_str = "lanrla"

# 超参数（模型核心配置）
input_size = len(char_set)
hidden_size = 6
batch_size = 1
epochs = 30

# 索引→one-hot编码（RNNCell输入需数值向量）
x_data = [char2idx[c] for c in input_str]
y_data = [char2idx[c] for c in target_str]
x_one_hot = torch.eye(input_size)[x_data].view(-1, batch_size, input_size)  # (6,1,6)
labels = torch.LongTensor(y_data).view(-1, 1)  # (6,1)


# 3. 核心模型定义（RNNCell结构与前向传播）
class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, x, h):  # 前向传播核心：单时间步计算
        return self.rnn_cell(x, h)

    def init_hidden(self):  # 隐藏状态初始化（核心）
        return torch.zeros(self.batch_size, self.hidden_size)

# 4. 核心训练逻辑（模型学习与参数更新）
model = RNNCellModel(input_size, hidden_size, batch_size)
criterion = nn.CrossEntropyLoss()  # 分类损失核心
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 优化器核心
train_history = []  # 保存训练数据（用于绘图）

for epoch in range(epochs):
    loss = 0.0
    optimizer.zero_grad()  # 梯度清零（核心步骤）
    hidden = model.init_hidden()  # 初始化隐藏态（核心步骤）

    # 逐时间步处理序列（RNNCell手动循环的核心）
    for x, label in zip(x_one_hot, labels):
        hidden = model(x, hidden)  # 单时间步前向传播
        loss += criterion(hidden, label)  # 累加损失

    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    # 保存训练历史（绘图依赖）
    _, idx = hidden.max(dim=1)
    train_history.append({'epoch': epoch + 1, 'loss': loss.item()})

# 5. 核心绘图逻辑（损失曲线生成）
epochs_list = [h['epoch'] for h in train_history]
losses_list = [h['loss'] for h in train_history]

plt.figure(figsize=(10, 6))
plt.plot(epochs_list, losses_list, color='#1f77b4', linewidth=2, marker='o')  # 折线图核心
plt.xlabel('Training Epochs'), plt.ylabel('Training Loss')
plt.title('RNNCell Loss Curve ("dlearn"→"lanrla")'), plt.grid(True, alpha=0.3)
plt.xticks(epochs_list[::2]), plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 核心数据预处理
char_set = ['d', 'l', 'e', 'a', 'r', 'n']
char2idx = {c: i for i, c in enumerate(char_set)}  # 字符→索引映射
input_str = "dlearn"
target_str = "lanrla"

# 超参数（模型核心配置，决定模型结构与训练节奏）
input_size = len(char_set)  # 输入维度=字符种类数
hidden_size = 6  # 隐藏层维度
num_layers = 1  # RNN层数（单层）
batch_size = 1  # 批次大小
seq_len = len(input_str)  # 序列长度（6）
epochs = 30  # 训练轮次

# 字符→索引→one-hot编码（RNN输入需满足 (seq_len, batch_size, input_size)）
x_data = [char2idx[c] for c in input_str]
y_data = [char2idx[c] for c in target_str]
x_one_hot = torch.eye(input_size)[x_data].view(seq_len, batch_size, input_size)  # (6,1,6)
labels = torch.LongTensor(y_data)  # (6,)，匹配CrossEntropyLoss输入格式


# 3. 核心模型定义（封装nn.RNN，实现序列自动处理）
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        # 初始化RNN核心模块（自动处理多时间步，无需手动循环）
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False  # 输入格式：(seq_len, batch, input_size)
        )
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, x):  # 前向传播核心：一次性处理整个序列
        # 初始化隐藏状态：(num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(x, h0)  # RNN自动处理序列，输出 (seq_len, batch, hidden_size)
        return out.view(-1, self.hidden_size)  # reshape适配损失函数


# 4. 核心训练逻辑（模型学习闭环）
model = RNNModel(input_size, hidden_size, num_layers, batch_size)
criterion = nn.CrossEntropyLoss()  # 分类任务核心损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 核心优化器
train_history = []  # 保存训练数据（支撑后续绘图）

for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零（避免梯度累积，核心步骤）
    outputs = model(x_one_hot)  # 前向传播：一次性处理整个序列
    loss = criterion(outputs, labels)  # 计算全序列损失

    loss.backward()  # 反向传播（求梯度，核心步骤）
    optimizer.step()  # 参数更新（优化模型，核心步骤）

    # 保存训练历史（仅保留绘图必需的epoch和loss）
    train_history.append({"epoch": epoch + 1, "loss": loss.item()})

# 5. 核心绘图逻辑（损失可视化）
# 提取绘图数据
epochs_list = [h["epoch"] for h in train_history]
losses_list = [h["loss"] for h in train_history]

# 生成损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, losses_list, color='#2ca02c', linewidth=2, marker='s', markersize=4)
plt.xlabel('Training Epochs'), plt.ylabel('Training Loss')  # 标签（兼容所有系统）
plt.title('RNN Model Loss Curve ("dlearn"→"lanrla")'), plt.grid(True, alpha=0.3)
plt.xticks(epochs_list[::2])  # 调整刻度避免拥挤
plt.tight_layout(), plt.show()