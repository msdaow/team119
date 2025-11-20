import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 基础配置与数据准备 --------------------------
# 字符映射
chars = ['d', 'l', 'e', 'a', 'r', 'n']
idx2char = {i: c for i, c in enumerate(chars)}
char2idx = {c: i for i, c in enumerate(chars)}

# 输入输出序列
input_str = "dlearn"
target_str = "lanrla"

# 模型参数
input_size = len(chars)
hidden_size = 8
batch_size = 1
seq_len = len(input_str)
epochs = 30
lr = 0.1

# 数据转换
x_data = [char2idx[c] for c in input_str]
x_one_hot = torch.eye(input_size)[x_data].view(seq_len, batch_size, input_size)
y_data = [char2idx[c] for c in target_str]
y_tensor = torch.LongTensor(y_data).view(-1, 1)

# 初始化训练日志（记录每轮数据，用于绘图）
rnncell_log = {"loss": [], "acc": []}
rnn_log = {"loss": [], "acc": []}


# -------------------------- 2. 工具函数（计算准确率） --------------------------
def cal_acc(pred_idx, target_idx):
    """计算预测准确率：正确字符数/总字符数"""
    correct = sum(p == t for p, t in zip(pred_idx, target_idx))
    return correct / len(target_idx)


# -------------------------- 3. RNNCell模型训练（带日志记录） --------------------------
class SimpleRNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        outputs = []
        for step in range(seq_len):
            hidden = self.rnn_cell(x[step], hidden)
            outputs.append(self.fc(hidden))
        return torch.stack(outputs), hidden

    def init_hidden(self):
        return torch.zeros(batch_size, hidden_size)


# 初始化与训练
rnncell_model = SimpleRNNCell()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnncell_model.parameters(), lr=lr)

print("RNNCell 训练中...")
for epoch in range(epochs):
    optimizer.zero_grad()
    hidden = rnncell_model.init_hidden()
    outputs, _ = rnncell_model(x_one_hot, hidden)

    # 计算损失
    loss = 0
    for step in range(seq_len):
        loss += criterion(outputs[step], y_tensor[step])

    # 计算准确率
    _, pred_idx = outputs.max(dim=2)
    pred_idx = pred_idx.squeeze().numpy()  # 转为numpy数组
    acc = cal_acc(pred_idx, y_data)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 记录日志
    rnncell_log["loss"].append(loss.item())
    rnncell_log["acc"].append(acc)


# -------------------------- 4. RNN模型训练（带日志记录） --------------------------
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        outputs, hidden = self.rnn(x, hidden)
        return self.fc(outputs), hidden

    def init_hidden(self):
        return torch.zeros(1, batch_size, hidden_size)  # (num_layers, batch, hidden)


# 初始化与训练
rnn_model = SimpleRNN()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)

print("RNN 训练中...")
for epoch in range(epochs):
    optimizer.zero_grad()
    hidden = rnn_model.init_hidden()
    outputs, _ = rnn_model(x_one_hot, hidden)

    # 计算损失
    outputs = outputs.view(-1, input_size)
    y_flat = y_tensor.view(-1)
    loss = criterion(outputs, y_flat)

    # 计算准确率
    _, pred_idx = outputs.max(dim=1)
    pred_idx = pred_idx.numpy()
    acc = cal_acc(pred_idx, y_data)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 记录日志
    rnn_log["loss"].append(loss.item())
    rnn_log["acc"].append(acc)

# -------------------------- 5. 绘制训练过程图（2个子图：损失+准确率） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.figure(figsize=(12, 5))  # 画布大小

# 子图1：损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), rnncell_log["loss"], label="RNNCell", marker='o', markersize=3)
plt.plot(range(1, epochs + 1), rnn_log["loss"], label="RNN", marker='s', markersize=3)
plt.xlabel("训练轮次（Epoch）")
plt.ylabel("损失值（Loss）")
plt.title("RNNCell vs RNN 损失变化")
plt.legend()
plt.grid(alpha=0.3)  # 显示网格，便于读数

# 子图2：准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), rnncell_log["acc"], label="RNNCell", marker='o', markersize=3, color='#ff7f0e')
plt.plot(range(1, epochs + 1), rnn_log["acc"], label="RNN", marker='s', markersize=3, color='#2ca02c')
plt.xlabel("训练轮次（Epoch）")
plt.ylabel("准确率（Accuracy）")
plt.title("RNNCell vs RNN 准确率变化")
plt.ylim(0, 1.1)  # 准确率范围0~1，留少量余量
plt.legend()
plt.grid(alpha=0.3)

# 保存图片（可选，也可直接显示）
plt.tight_layout()  # 调整子图间距
plt.savefig("rnn_training_plot.png", dpi=300, bbox_inches='tight')  # 高分辨率保存
plt.show()

# -------------------------- 6. 打印最终结果（量化验证） --------------------------
print("\n" + "=" * 60)
print("最终训练结果")
print("=" * 60)

# RNNCell最终结果
rnncell_final_pred = [idx2char[np.argmax(outputs[step].detach().numpy())]
                      for step in range(seq_len)]
rnncell_final_pred = ''.join(rnncell_final_pred)
print(
    f"RNNCell：预测={rnncell_final_pred} | 目标={target_str} | 最终准确率={rnncell_log['acc'][-1]:.2%} | 最终损失={rnncell_log['loss'][-1]:.4f}")

# RNN最终结果
rnn_final_outputs, _ = rnn_model(x_one_hot, rnn_model.init_hidden())
rnn_final_outputs = rnn_final_outputs.view(-1, input_size)
rnn_final_pred_idx = torch.argmax(rnn_final_outputs, dim=1).numpy()
rnn_final_pred = ''.join([idx2char[idx] for idx in rnn_final_pred_idx])
print(
    f"RNN：预测={rnn_final_pred} | 目标={target_str} | 最终准确率={rnn_log['acc'][-1]:.2%} | 最终损失={rnn_log['loss'][-1]:.4f}")