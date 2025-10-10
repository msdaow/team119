import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm

# 从train.csv加载数据
data = pd.read_csv('train.csv')

# 数据清洗
data_cleaned = data.drop_duplicates(subset=['x'], keep='first')

# 检查异常值（基于IQR方法）
Q1 = data_cleaned['y'].quantile(0.25)
Q3 = data_cleaned['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cleaned = data_cleaned[(data_cleaned['y'] >= lower_bound) & (data_cleaned['y'] <= upper_bound)]

# 数据标准化
x_mean, x_std = data_cleaned['x'].mean(), data_cleaned['x'].std()
y_mean, y_std = data_cleaned['y'].mean(), data_cleaned['y'].std()

data_cleaned['x_normalized'] = (data_cleaned['x'] - x_mean) / x_std
data_cleaned['y_normalized'] = (data_cleaned['y'] - y_mean) / y_std

# 准备训练数据
x_data = torch.Tensor(data_cleaned['x_normalized'].values.reshape(-1, 1))
y_data = torch.Tensor(data_cleaned['y_normalized'].values.reshape(-1, 1))


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)


# 定义不同的学习率和epoch组合
learning_rates = [0.0001, 0.001, 0.01, 0.1]
epochs_list = [500, 1000, 2000, 5000]

# 存储所有实验结果
results = {}

print("开始超参数调节实验...")

# 实验1：固定epoch=1000，比较不同学习率
print("\n=== 实验1: 固定epoch=1000，比较不同学习率 ===")
fixed_epoch = 1000
lr_results = {}

for lr in learning_rates:
    print(f"\n学习率 η = {lr}")
    model = LinearModel()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    w_history = []
    b_history = []

    for epoch in range(fixed_epoch):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        current_w = model.linear.weight.item() * (y_std / x_std)
        current_b = model.linear.bias.item() * y_std + y_mean - current_w * x_mean

        loss_history.append(loss.item())
        w_history.append(current_w)
        b_history.append(current_w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_results[lr] = {
        'loss': loss_history,
        'w': w_history,
        'b': b_history,
        'final_loss': loss_history[-1],
        'final_w': w_history[-1],
        'final_b': b_history[-1]
    }

    print(f"最终损失: {loss_history[-1]:.4f}, w: {w_history[-1]:.4f}, b: {b_history[-1]:.4f}")

results['learning_rate_comparison'] = lr_results

# 实验2：固定学习率=0.001，比较不同epoch数量
print("\n=== 实验2: 固定学习率=0.001，比较不同epoch数量 ===")
fixed_lr = 0.001
epoch_results = {}

for max_epochs in epochs_list:
    print(f"\n最大epoch = {max_epochs}")
    model = LinearModel()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr)

    loss_history = []
    w_history = []
    b_history = []

    for epoch in range(max_epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        current_w = model.linear.weight.item() * (y_std / x_std)
        current_b = model.linear.bias.item() * y_std + y_mean - current_w * x_mean

        loss_history.append(loss.item())
        w_history.append(current_w)
        b_history.append(current_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (max_epochs // 10) == 0 and epoch > 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    epoch_results[max_epochs] = {
        'loss': loss_history,
        'w': w_history,
        'b': b_history,
        'final_loss': loss_history[-1],
        'final_w': w_history[-1],
        'final_b': b_history[-1]
    }

    print(f"最终损失: {loss_history[-1]:.4f}")

results['epoch_comparison'] = epoch_results

# 实验3：学习率和epoch的组合网格搜索
print("\n=== 实验3: 学习率和epoch组合网格搜索 ===")
grid_results = {}

for lr in [0.0001, 0.001, 0.01]:
    for max_epochs in [500, 1000, 2000]:
        print(f"\n学习率: {lr}, Epoch: {max_epochs}")
        model = LinearModel()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        final_loss = 0
        for epoch in range(max_epochs):
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        grid_results[(lr, max_epochs)] = final_loss
        print(f"最终损失: {final_loss:.4f}")

results['grid_search'] = grid_results

# 创建可视化图表
print("\n生成可视化图表...")

# 图1：不同学习率的损失曲线对比
plt.figure(figsize=(15, 12))

# 1.1 不同学习率的损失曲线
plt.subplot(2, 2, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))
for i, (lr, data_dict) in enumerate(results['learning_rate_comparison'].items()):
    plt.plot(data_dict['loss'], label=f'η={lr}', color=colors[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('η  compare\n(固定epoch=1000)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标更好地观察变化

# 1.2 不同学习率的收敛速度
plt.subplot(2, 2, 2)
for i, (lr, data_dict) in enumerate(results['learning_rate_comparison'].items()):
    # 计算达到最终损失90%所需的epoch数
    target_loss = data_dict['final_loss'] * 1.1
    convergence_epoch = next((idx for idx, loss in enumerate(data_dict['loss']) if loss <= target_loss), fixed_epoch)
    plt.bar(f'η={lr}', convergence_epoch, color=colors[i], alpha=0.7)
plt.ylabel('need poch num')
plt.title('Convergence speed of different learning rates')
plt.grid(True, alpha=0.3)

# 1.3 不同epoch数量的损失曲线
plt.subplot(2, 2, 3)
colors_epoch = plt.cm.plasma(np.linspace(0, 1, len(epochs_list)))
for i, (max_epochs, data_dict) in enumerate(results['epoch_comparison'].items()):
    plt.plot(data_dict['loss'], label=f'Epoch={max_epochs}', color=colors_epoch[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(' Loss curve for the number of epochs\n(固定学习率η=0.001)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 1.4 超参数组合的热力图
plt.subplot(2, 2, 4)
lr_values = sorted(set(k[0] for k in results['grid_search'].keys()))
epoch_values = sorted(set(k[1] for k in results['grid_search'].keys()))
loss_matrix = np.zeros((len(lr_values), len(epoch_values)))

for (lr, epochs), loss in results['grid_search'].items():
    i = lr_values.index(lr)
    j = epoch_values.index(epochs)
    loss_matrix[i, j] = loss

im = plt.imshow(loss_matrix, cmap='viridis_r', aspect='auto')
plt.colorbar(im, label='Final Loss')
plt.xticks(range(len(epoch_values)), [f'{e}' for e in epoch_values])
plt.yticks(range(len(lr_values)), [f'{lr:.4f}' for lr in lr_values])
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Hyperparameter Combination Performance Heatmap\n(颜色越深性能越好)')

plt.tight_layout()
plt.savefig('hyperparameter_tuning_comprehensive.png', dpi=300, bbox_inches='tight')

# 图2：学习率对训练动态的影响
plt.figure(figsize=(15, 10))

# 2.1 学习率对损失收敛的影响（前200个epoch）
plt.subplot(2, 3, 1)
for lr, data_dict in results['learning_rate_comparison'].items():
    plt.plot(data_dict['loss'][:200], label=f'η={lr}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('The impact of η on early convergence\n(前200个epoch)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 2.2 学习率对参数w收敛的影响
plt.subplot(2, 3, 2)
for lr, data_dict in results['learning_rate_comparison'].items():
    plt.plot(data_dict['w'], label=f'η={lr}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('The effect of learning rate on the convergence of  w')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.3 学习率对参数b收敛的影响
plt.subplot(2, 3, 3)
for lr, data_dict in results['learning_rate_comparison'].items():
    plt.plot(data_dict['b'], label=f'η={lr}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title('The effect of learning rate on the convergence of  b')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.4 最终损失 vs 学习率
plt.subplot(2, 3, 4)
final_losses = [data_dict['final_loss'] for data_dict in results['learning_rate_comparison'].values()]
plt.semilogx(learning_rates, final_losses, 'o-', linewidth=2, markersize=8)
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Final Loss')
plt.title('loss vs η')
plt.grid(True, alpha=0.3)

# 2.5 训练时间效率（用epoch数代替）
plt.subplot(2, 3, 5)
convergence_speed = []
for lr, data_dict in results['learning_rate_comparison'].items():
    target = data_dict['final_loss'] * 1.05  # 达到最终损失的95%
    speed = next((idx for idx, loss in enumerate(data_dict['loss']) if loss <= target), fixed_epoch)
    convergence_speed.append(speed)
plt.semilogx(learning_rates, convergence_speed, 's-', linewidth=2, markersize=8)
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Epoch num')
plt.title('Convergence rate vs η')
plt.grid(True, alpha=0.3)

# 2.6 最优学习率推荐
plt.subplot(2, 3, 6)
optimal_idx = np.argmin(final_losses)
optimal_lr = learning_rates[optimal_idx]
optimal_loss = final_losses[optimal_idx]

plt.bar(range(len(learning_rates)), final_losses,
        color=['red' if i == optimal_idx else 'blue' for i in range(len(learning_rates))])
plt.xticks(range(len(learning_rates)), [f'η={lr}' for lr in learning_rates], rotation=45)
plt.ylabel('Final Loss')
plt.title(f'best: η={optimal_lr}\nloss： {optimal_loss:.4f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_analysis_detailed.png', dpi=300, bbox_inches='tight')

print("\n=== 超参数调节实验总结 ===")
print("1. 学习率比较 (固定epoch=1000):")
for lr, data_dict in results['learning_rate_comparison'].items():
    print(f"   学习率 η={lr}: 最终损失 = {data_dict['final_loss']:.4f}")

print("\n2. Epoch数量比较 (固定学习率=0.001):")
for epochs, data_dict in results['epoch_comparison'].items():
    print(f"   Epoch={epochs}: 最终损失 = {data_dict['final_loss']:.4f}")

print("\n3. 网格搜索最佳组合:")
best_combo = min(results['grid_search'], key=results['grid_search'].get)
best_loss = results['grid_search'][best_combo]
print(f"   最佳组合: 学习率={best_combo[0]}, Epoch={best_combo[1]}")
print(f"   最佳损失: {best_loss:.4f}")

print("\n可视化图表已保存:")
print("- hyperparameter_tuning_comprehensive.png")
print("- learning_rate_analysis_detailed.png")