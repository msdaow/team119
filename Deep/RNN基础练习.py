import torch
import torch.nn as nn

# ---------------------- 数据预处理 ----------------------
idx2char = ['a', 'd', 'e', 'l', 'n', 'r']
char2idx = {c: i for i, c in enumerate(idx2char)}
vocab_size = len(idx2char)

input_str = "dlearn"
target_str = "lanrla"
x_data = [char2idx[c] for c in input_str]
y_data = [char2idx[c] for c in target_str]

one_hot_lookup = torch.eye(vocab_size)
x_one_hot = [one_hot_lookup[x] for x in x_data]

batch_size = 1
seq_len = len(x_data)
input_size = vocab_size
hidden_size = 8

x_one_hot_tensor = torch.stack(x_one_hot)
inputs = x_one_hot_tensor.view(seq_len, batch_size, input_size)
labels = torch.tensor(y_data, dtype=torch.long).view(seq_len, 1)

# ---------------------- RNNCell模型训练（仅保留轮数+损失） ----------------------
class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        output = self.fc(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, hidden_size)

net_cell = RNNCellModel(input_size, hidden_size, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_cell.parameters(), lr=0.01)

print("===== RNNCell 训练日志 =====")
for epoch in range(300):
    loss = 0
    optimizer.zero_grad()
    hidden = net_cell.init_hidden()

    # 仅训练，不打印任何预测内容
    for input, label in zip(inputs, labels):
        output, hidden = net_cell(input, hidden)
        loss += criterion(output, label)

    loss.backward()
    optimizer.step()

    # 每20轮只打印“轮数+损失”
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

# ---------------------- RNN模型训练（仅保留轮数+损失） ----------------------
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, hidden_size)
        out, _ = self.rnn(input, hidden)
        out = self.fc(out.view(-1, hidden_size))
        return out.view(seq_len, batch_size, vocab_size)

net_rnn = RNNModel(input_size, hidden_size, batch_size)
optimizer_rnn = torch.optim.Adam(net_rnn.parameters(), lr=0.01)

print("\n===== RNN 训练日志 =====")
for epoch in range(300):
    loss = 0
    optimizer_rnn.zero_grad()
    outputs = net_rnn(inputs)

    # 仅训练，不打印任何预测内容
    for t in range(seq_len):
        loss += criterion(outputs[t], labels[t])

    loss.backward()
    optimizer_rnn.step()

    # 每20轮只打印“轮数+损失”
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")