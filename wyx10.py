import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. 使用nn.RNNCell训练模型
print("=" * 50)
print("使用RNNCell训练模型")
print("=" * 50)

# 参数设置
input_size = 6  # 6个不同字符: a,d,e,l,n,r
hidden_size = 6
batch_size = 1

# 创建字典
idx2char = ['a', 'd', 'e', 'l', 'n', 'r']
char2idx = {char: idx for idx, char in enumerate(idx2char)}

# 准备数据
x_data = [char2idx[ch] for ch in "dlearn"]  # [1, 3, 2, 0, 4, 5]
y_data = [char2idx[ch] for ch in "lanrla"]  # [3, 0, 4, 5, 3, 0]

# one-hot查找表
one_hot_lookup = [
    [1, 0, 0, 0, 0, 0],  # a
    [0, 1, 0, 0, 0, 0],  # d
    [0, 0, 1, 0, 0, 0],  # e
    [0, 0, 0, 1, 0, 0],  # l
    [0, 0, 0, 0, 1, 0],  # n
    [0, 0, 0, 0, 0, 1]  # r
]

x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # (seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)  # (seq_len, 1)


# 定义模型
class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(RNNCellModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


# 创建模型、损失函数和优化器
net_cell = RNNCellModel(input_size, hidden_size, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_cell.parameters(), lr=0.1)

# 训练过程数据存储
cell_losses = []
cell_predictions = []

# 训练
print("开始训练RNNCell模型...")
for epoch in range(100):
    loss = 0
    optimizer.zero_grad()
    hidden = net_cell.init_hidden()

    # 存储预测结果
    predicted_chars = []

    for input, label in zip(inputs, labels):
        hidden = net_cell(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        predicted_chars.append(idx2char[idx.item()])

    loss.backward()
    optimizer.step()

    # 保存训练过程数据
    cell_losses.append(loss.item())
    cell_predictions.append(''.join(predicted_chars))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}, Predicted: {"".join(predicted_chars)}')

print(f"最终预测结果: {cell_predictions[-1]}")
print(f"目标结果: lanrla")

# 2. 使用nn.RNN训练模型
print("\n" + "=" * 50)
print("使用RNN训练模型")
print("=" * 50)

# 参数设置
num_layers = 1
seq_len = 6


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


# 准备数据 (保持与之前相同)
inputs_rnn = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels_rnn = torch.LongTensor(y_data)

# 创建模型、损失函数和优化器
net_rnn = RNNModel(input_size, hidden_size, batch_size, num_layers)
criterion_rnn = nn.CrossEntropyLoss()
optimizer_rnn = torch.optim.Adam(net_rnn.parameters(), lr=0.05)

# 训练过程数据存储
rnn_losses = []
rnn_predictions = []

print("开始训练RNN模型...")
for epoch in range(100):
    optimizer_rnn.zero_grad()
    outputs = net_rnn(inputs_rnn)
    loss = criterion_rnn(outputs, labels_rnn)
    loss.backward()
    optimizer_rnn.step()

    # 获取预测结果
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    predicted_string = ''.join([idx2char[x] for x in idx])

    # 保存训练过程数据
    rnn_losses.append(loss.item())
    rnn_predictions.append(predicted_string)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}, Predicted: {predicted_string}')

print(f"最终预测结果: {rnn_predictions[-1]}")
print(f"目标结果: lanrla")

# 3. 可视化训练过程
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(cell_losses, label='RNNCell Loss')
plt.plot(rnn_losses, label='RNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)


# 绘制准确率曲线
def calculate_accuracy(predictions, target):
    accuracies = []
    for pred in predictions:
        correct = sum(1 for p, t in zip(pred, target) if p == t)
        accuracies.append(correct / len(target))
    return accuracies


cell_accuracies = calculate_accuracy(cell_predictions, "lanrla")
rnn_accuracies = calculate_accuracy(rnn_predictions, "lanrla")

plt.subplot(1, 2, 2)
plt.plot(cell_accuracies, label='RNNCell Accuracy')
plt.plot(rnn_accuracies, label='RNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. 保存训练过程数据
training_data = {
    'rnncell': {
        'losses': cell_losses,
        'predictions': cell_predictions,
        'accuracies': cell_accuracies
    },
    'rnn': {
        'losses': rnn_losses,
        'predictions': rnn_predictions,
        'accuracies': rnn_accuracies
    }
}

print("\n训练过程数据已保存!")
print("RNNCell最终准确率: {:.2f}%".format(cell_accuracies[-1] * 100))
print("RNN最终准确率: {:.2f}%".format(rnn_accuracies[-1] * 100))

# 5. 测试最终模型
print("\n" + "=" * 50)
print("最终模型测试")
print("=" * 50)

# 测试RNNCell模型
print("RNNCell模型测试:")
hidden = net_cell.init_hidden()
final_prediction_cell = []
for input in inputs:
    hidden = net_cell(input, hidden)
    _, idx = hidden.max(dim=1)
    final_prediction_cell.append(idx2char[idx.item()])
print(f"预测: {''.join(final_prediction_cell)}")
print(f"目标: lanrla")
print(f"是否匹配: {''.join(final_prediction_cell) == 'lanrla'}")

# 测试RNN模型
print("\nRNN模型测试:")
outputs = net_rnn(inputs_rnn)
_, idx = outputs.max(dim=1)
final_prediction_rnn = ''.join([idx2char[x] for x in idx.data.numpy()])
print(f"预测: {final_prediction_rnn}")
print(f"目标: lanrla")
print(f"是否匹配: {final_prediction_rnn == 'lanrla'}")