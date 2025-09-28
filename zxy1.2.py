# 导入所需要的工具包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# 读取数据
data = pd.read_csv('train.csv')

# 数据清洗：去除空值和异常值
data = data.dropna()  # 删除空值
data = data[(data['x'] > 0) & (data['y'] > 0)]

# 准备数据集
x_data = data['x'].values
y_data = data['y'].values

print("after clean:")
for i in range(len(x_data)):
    print(f"x={x_data[i]:.1f}, y={y_data[i]:.1f}")

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x_data).view(-1, 1)
y_tensor = torch.FloatTensor(y_data).view(-1, 1)


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)


# 创建模型实例
model = LinearModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 用于记录训练过程中的w和loss
w_history = []
loss_history = []

# 训练模型
epochs = 100
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_tensor)

    # 计算损失
    loss = criterion(y_pred, y_tensor)

    # 记录当前参数和损失
    current_w = model.linear.weight.item()
    current_b = model.linear.bias.item()
    w_history.append(current_w)
    loss_history.append(loss.item())

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每10个epoch打印一次信息
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, w: {current_w:.4f}, b: {current_b:.4f}')

# 绘制w和loss的变化过程
plt.figure(figsize=(12, 5))

# 子图1：loss随epoch的变化
plt.subplot(1, 2, 1)
plt.plot(range(epochs), loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)

# 子图2：w随epoch的变化
plt.subplot(1, 2, 2)
plt.plot(range(epochs), w_history, 'r-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('Weight vs Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()

# 最终参数
final_w = model.linear.weight.item()
final_b = model.linear.bias.item()
final_loss = criterion(model(x_tensor), y_tensor).item()

print(f"\nFinal parameters: w = {final_w:.4f}, b = {final_b:.4f}")
print(f"Final loss: {final_loss:.4f}")

# 绘制原始数据和最佳拟合线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='Original data')

# 生成预测线
x_range = np.linspace(min(x_data), max(x_data), 100)
x_range_tensor = torch.FloatTensor(x_range).view(-1, 1)
y_pred_range = model(x_range_tensor).detach().numpy()

plt.plot(x_range, y_pred_range, color='red',
         label=f'Fitted line: y = {final_w:.2f}x + {final_b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Results of Linear Regression Fitting')
plt.grid(True)
plt.show()


# 预测新数据
def predict(x):
    x_tensor = torch.FloatTensor([x]).view(-1, 1)
    return model(x_tensor).item()


print("\nPrediction examples:")
print(f"predict (after training) x=4: y={predict(4):.2f}")
print(f"predict (after training) x=5: y={predict(5):.2f}")