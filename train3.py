import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\Users\86131\Downloads\train.csv")  # 请确保文件路径正确，如果文件不在当前工作目录下，需要使用完整路径

# 使用 inner join 合并 x 和 y 数据，确保数据量一致
joined_data = df[['x', 'y']].dropna(how='any')

# 提取特征和目标变量
x = torch.tensor(joined_data['x'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(joined_data['y'].values, dtype=torch.float32).unsqueeze(1)

# 定义线性模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 100
w_values = []
loss_values = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录权重和损失
    for name, param in model.named_parameters():
        if name == 'linear.weight':
            w_values.append(param.item())
    loss_values.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置字体大小
plt.rcParams.update({'font.size': 4})

# 绘制权重和损失的变化
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), w_values, label='权重 w')
plt.xlabel('迭代次数')
plt.xticks(rotation=45)
plt.ylabel('权重值')
plt.title('权重 w 的变化')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), loss_values, label='损失 loss', color='orange')
plt.xlabel('迭代次数')
plt.xticks(rotation=45)
plt.ylabel('损失值')
plt.title('损失 loss 的变化')
plt.legend()

plt.tight_layout()
plt.show()