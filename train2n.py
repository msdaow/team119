import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决OpenMP库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 读取CSV数据并处理异常值
def load_and_process_data(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径")
    # 读取CSV数据
    df = pd.read_csv(file_path)
    # 确保至少有两列数据
    if len(df.columns) < 2:
        raise ValueError("CSV文件至少需要包含两列数据（x和y）")
    # 取前两列作为x和y
    df = df.iloc[:, :2]
    df.columns = ['x', 'y']
    # 使用IQR方法处理异常值
    def remove_outliers(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return df[(df[column] >= lower) & (df[column] <= upper)]
    # 处理x和y列的异常值
    df_clean = remove_outliers(df, 'x')
    df_clean = remove_outliers(df_clean, 'y')
    print(f"数据处理完成：原始{len(df)}条，处理后{len(df_clean)}条（已移除异常值）")
    return df_clean['x'].values, df_clean['y'].values
# 替换为你的CSV文件路径
file_path = "train.csv"  # 请修改为实际的CSV文件路径
x_data, y_data = load_and_process_data(file_path)
# 数据归一化（防止梯度问题）
x_mean, x_std = np.mean(x_data), np.std(x_data)
y_mean, y_std = np.mean(y_data), np.std(y_data)
# 避免除以零
x_std = x_std if x_std != 0 else 1
y_std = y_std if y_std != 0 else 1
x_norm = (x_data - x_mean) / x_std
y_norm = (y_data - y_mean) / y_std
# 初始化权重w和偏置b
w = torch.Tensor([1.0])
b = torch.Tensor([0.0])  # 新增偏置项b
w.requires_grad = True
b.requires_grad = True  # 偏置项也需要计算梯度

# 前向传播 - 新增偏置项b
def forward(x):
    return x * w + b

# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) **2


# 记录训练过程 - 新增记录b的变化
w_history = []
b_history = []
loss_history = []
epochs_list = []

# 训练前的预测（转换回原始尺度）
x_test = (4 - x_mean) / x_std  # 对测试值进行归一化
y_pred_before = forward(torch.Tensor([x_test])).item() * y_std + y_mean
print(f"训练前的预测（x=4）: {y_pred_before:.4f}")

# 训练模型
epochs = 100
learning_rate = 0.001

for epoch in range(epochs):
    total_loss = 0.0
    for x, y in zip(x_norm, y_norm):
        x_tensor = torch.Tensor([x])
        y_tensor = torch.Tensor([y])

        l = loss(x_tensor, y_tensor)
        total_loss += l.item()

        # 反向传播计算梯度
        l.backward()

        # 更新权重和偏置 - 新增更新b的代码
        w.data = w.data - learning_rate * w.grad.data
        b.data = b.data - learning_rate * b.grad.data

        # 清空梯度
        w.grad.data.zero_()
        b.grad.data.zero_()  # 同时清空b的梯度

    # 记录每轮的平均损失、权重和偏置
    avg_loss = total_loss / len(x_norm)
    w_history.append(w.item())
    b_history.append(b.item())  # 记录b的变化
    loss_history.append(avg_loss)
    epochs_list.append(epoch)

    # 每10轮打印一次信息
    if (epoch + 1) % 10 == 0:
        print(f"轮次: {epoch + 1}, 平均损失: {avg_loss:.6f}, 权重w: {w.item():.6f}, 偏置b: {b.item():.6f}")

# 训练后的预测
y_pred_after = forward(torch.Tensor([x_test])).item() * y_std + y_mean
print(f"训练后的预测（x=4）: {y_pred_after:.4f}")

# 可视化w、b和loss的变化
plt.figure(figsize=(12, 15))

# 绘制w的变化曲线
plt.subplot(3, 1, 1)
plt.plot(epochs_list, w_history, 'b-', linewidth=2)
plt.title('权重w随训练轮次的变化')
plt.xlabel('训练轮次')
plt.ylabel('w')
plt.grid(True, linestyle='--', alpha=0.7)

# 绘制b的变化曲线 - 新增b的可视化
plt.subplot(3, 1, 2)
plt.plot(epochs_list, b_history, 'g-', linewidth=2)
plt.title('偏置b随训练轮次的变化')
plt.xlabel('训练轮次')
plt.ylabel('b')
plt.grid(True, linestyle='--', alpha=0.7)

# 绘制loss的变化曲线
plt.subplot(3, 1, 3)
plt.plot(epochs_list, loss_history, 'r-', linewidth=2)
plt.title('loss随训练轮次的变化')
plt.xlabel('训练轮次')
plt.ylabel('loss')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 绘制原始数据和拟合直线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='原始数据')

# 生成拟合直线
x_range = np.linspace(min(x_data), max(x_data), 100)
x_range_norm = (x_range - x_mean) / x_std
y_pred_norm = [forward(torch.Tensor([x])).item() for x in x_range_norm]
y_pred = np.array(y_pred_norm) * y_std + y_mean

# 显示包含偏置项的拟合方程
plt.plot(x_range, y_pred, 'r-', linewidth=2, label=f'拟合直线: y = {w.item():.4f}x + {b.item():.4f}')
plt.title('原始数据与拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
