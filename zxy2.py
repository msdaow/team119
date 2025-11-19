import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 解决可能的OpenMP冲突
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 读取数据
data = pd.read_csv('train.csv')

# 数据清洗：更严格的清洗
data = data.dropna()
data = data[(data['x'] > 0) & (data['y'] > 0)]
# 去除极端值
data = data[(data['x'] < data['x'].quantile(0.95)) &
            (data['y'] < data['y'].quantile(0.95))]

# 准备数据集
x_data = torch.tensor(data['x'].values, dtype=torch.float32)
y_data = torch.tensor(data['y'].values, dtype=torch.float32)

print("数据统计:")
print(f"x范围: {x_data.min().item()} ~ {x_data.max().item()}")
print(f"y范围: {y_data.min().item()} ~ {y_data.max().item()}")
print(f"样本数量: {len(x_data)}")

# 初始化权重参数 - 使用更小的初始值
w = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)  # 从0.1开始而不是0


# 正向传播
def forward(x):
    return x * w


# 损失函数 - 添加小的epsilon防止数值问题
def loss(x, y, epsilon=1e-8):
    y_pred = forward(x)
    return (y_pred - y) ** 2 + epsilon  # 添加小值防止除零或log(0)


# 记录历史
w_history = []
loss_history = []

print("预测（训练前）:", 4, forward(torch.tensor([4.0])).item())

# 训练循环 - 添加稳定性措施
learning_rate = 0.001  # 使用更小的学习率

for epoch in range(100):
    epoch_loss = 0
    count = 0

    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        # 检查梯度是否为nan
        if torch.isnan(w.grad).any():
            print(f"警告: 第{epoch}轮梯度为nan，跳过更新")
            w.grad.data.zero_()
            continue

        # 梯度裁剪
        grad_norm = torch.norm(w.grad)
        if grad_norm > 100:  # 如果梯度太大就裁剪
            w.grad.data = w.grad.data / grad_norm * 100

        # 更新参数
        w.data = w.data - learning_rate * w.grad.data

        # 检查参数是否为nan
        if torch.isnan(w).any():
            print(f"警告: 第{epoch}轮参数为nan，重新初始化")
            w.data = torch.tensor([0.1], dtype=torch.float32)
            w.grad.data.zero_()
            continue

        w.grad.data.zero_()

        epoch_loss += l.item()
        count += 1

    if count > 0:
        avg_loss = epoch_loss / count
    else:
        avg_loss = float('nan')

    # 记录历史
    w_history.append(w.item())
    loss_history.append(avg_loss)

    # 检查loss是否为nan
    if np.isnan(avg_loss):
        print(f"进度: {epoch}, loss=nan (训练出现问题)")
        break
    else:
        if epoch % 10 == 0:
            print(f"进度: {epoch}, loss={avg_loss:.4f}, w={w.item():.4f}")

# 绘制结果（只有在训练正常完成时）
if not any(np.isnan(loss_history)):
    # 绘制训练过程中w和loss的变化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
    plt.xlabel(' Epoch')
    plt.ylabel(' Loss')
    plt.title('loss change of Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(w_history)), w_history, 'r-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('w')
    plt.title('w  change of Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制拟合结果
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data.numpy(), y_data.numpy(), color='blue', label='original')

    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_pred = forward(torch.tensor(x_range, dtype=torch.float32)).detach().numpy()
    plt.plot(x_range, y_pred, 'red', label=f'拟合: y = {w.item():.3f}x')

    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n最终结果: w = {w.item():.4f}")