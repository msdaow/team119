import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 读取数据，修改文件路径为本地实际路径
df = pd.read_csv('F:/MobileFile/train.csv')

# 处理缺失值
print("处理前数据形状:", df.shape)
print("缺失值统计:")
print(df.isnull().sum())

# 移除包含缺失值的行
df_clean = df.dropna(subset=['x', 'y'])
print("处理后数据形状:", df_clean.shape)

# 提取处理后的数据
x_data = df_clean['x'].values
y_data = df_clean['y'].values

# 数据预处理：归一化，避免梯度爆炸和NaN问题
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()

x_normalized = (x_data - x_mean) / x_std
y_normalized = (y_data - y_mean) / y_std

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x_normalized).view(-1, 1)
y_tensor = torch.FloatTensor(y_normalized).view(-1, 1)


# 创建模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1个输入特征，1个输出

    def forward(self, x):
        return self.linear(x)


# 初始化模型
model = LinearRegression()
criterion = nn.MSELoss()
# 使用更小的学习率，并添加梯度裁剪
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 存储训练过程中的值
w_values = []
b_values = []
loss_values = []

print("\n数据统计信息:")
print(f"x均值: {x_mean:.4f}, x标准差: {x_std:.4f}")
print(f"y均值: {y_mean:.4f}, y标准差: {y_std:.4f}")

# 训练模型
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # 记录参数和损失
    current_w = model.linear.weight.item()
    current_b = model.linear.bias.item()
    w_values.append(current_w)
    b_values.append(current_b)
    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}, w: {current_w:.6f}, b: {current_b:.6f}')

# 将归一化的参数转换回原始尺度
final_w_norm = model.linear.weight.item()
final_b_norm = model.linear.bias.item()

final_w_original = final_w_norm * y_std / x_std
final_b_original = final_b_norm * y_std + y_mean - final_w_norm * y_std * x_mean / x_std

print(f"\n归一化尺度参数: w = {final_w_norm:.6f}, b = {final_b_norm:.6f}")
print(f"原始尺度参数: w = {final_w_original:.6f}, b = {final_b_original:.6f}")
print(f"模型方程: y = {final_w_original:.6f}x + {final_b_original:.6f}")

# 绘制w和loss的变化
plt.figure(figsize=(15, 5))

# w的变化
plt.subplot(1, 3, 1)
plt.plot(range(epochs), w_values, 'b-', linewidth=2)
plt.title('Weight (w) Change During Training')
plt.xlabel('Epoch')
plt.ylabel('w value (normalized)')
plt.grid(True)

# loss的变化
plt.subplot(1, 3, 3)
plt.plot(range(epochs), loss_values, 'g-', linewidth=2)
plt.title('Loss Change During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 使用对数坐标更好地观察loss变化
plt.grid(True)

plt.tight_layout()
plt.show()

# 绘制最终拟合结果（原始尺度）
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.6, label='Original data')

# 生成预测线（原始尺度）
x_range = np.linspace(min(x_data), max(x_data), 100)
x_range_normalized = (x_range - x_mean) / x_std
x_tensor_range = torch.FloatTensor(x_range_normalized).view(-1, 1)

with torch.no_grad():
    y_pred_range_normalized = model(x_tensor_range).numpy()

# 将预测结果转换回原始尺度
y_pred_range = y_pred_range_normalized * y_std + y_mean

plt.plot(x_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# 测试预测
test_x = 5.0
test_x_normalized = (test_x - x_mean) / x_std
test_pred_normalized = model(torch.FloatTensor([test_x_normalized]))
test_pred = test_pred_normalized.item() * y_std + y_mean
print(f"\n测试预测: x={test_x}, y_pred={test_pred:.4f}")