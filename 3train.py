import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 读取数据
data = pd.read_csv('train.csv')

# 数据清洗
data_clean = data.dropna()
data_clean = data_clean[np.isfinite(data_clean['x'])]
data_clean = data_clean[np.isfinite(data_clean['y'])]

print(f"Original data size: {len(data)}")
print(f"Cleaned data size: {len(data_clean)}")
print(f"x range: [{data_clean['x'].min():.2f}, {data_clean['x'].max():.2f}]")
print(f"y range: [{data_clean['y'].min():.2f}, {data_clean['y'].max():.2f}]")

# 数据标准化
x_mean = data_clean['x'].mean()
x_std = data_clean['x'].std()
y_mean = data_clean['y'].mean()
y_std = data_clean['y'].std()

x_normalized = (data_clean['x'] - x_mean) / x_std
y_normalized = (data_clean['y'] - y_mean) / y_std

# 转换为Tensor
x_data = torch.FloatTensor(x_normalized.values).reshape(-1, 1)
y_data = torch.FloatTensor(y_normalized.values).reshape(-1, 1)


# 自定义数据集类
class CSVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 创建数据加载器
dataset = CSVDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 定义线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 使用正态分布初始化权重和偏置
        nn.init.normal_(self.linear.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear.bias, mean=0, std=1.0)

    def forward(self, x):
        return self.linear(x)


# 选择三种优化器
optimizers_list = [
    ('SGD', torch.optim.SGD),
    ('Adam', torch.optim.Adam),
    ('RMSprop', torch.optim.RMSprop)
]

# 存储训练结果
results = {}

# 存储参数调节过程
param_history = {'SGD': {'w': [], 'b': [], 'loss': []},
                 'Adam': {'w': [], 'b': [], 'loss': []},
                 'RMSprop': {'w': [], 'b': [], 'loss': []}}


# 训练函数
def train_model(optimizer_name, optimizer_class, lr=0.01, epochs=1000):
    print(f"\nTraining {optimizer_name} optimizer (lr={lr}, epochs={epochs})...")

    # 创建新模型
    model = LinearModel()
    criterion = nn.MSELoss()

    # 创建优化器
    if optimizer_name == 'SGD':
        optimizer = optimizer_class(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optimizer_class(model.parameters(), lr=lr)
    else:  # RMSprop
        optimizer = optimizer_class(model.parameters(), lr=lr)

    # 记录训练过程
    losses = []
    w_values = []
    b_values = []

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        for batch_x, batch_y in dataloader:
            # 前向传播
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
        else:
            avg_loss = float('inf')

        losses.append(avg_loss)

        # 记录参数
        current_w = model.linear.weight.item()
        current_b = model.linear.bias.item()
        w_values.append(current_w)
        b_values.append(current_b)

        # 记录到历史
        param_history[optimizer_name]['w'].append(current_w)
        param_history[optimizer_name]['b'].append(current_b)
        param_history[optimizer_name]['loss'].append(avg_loss)

        if epoch % 200 == 0 and epoch > 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.6f}, w: {current_w:.6f}, b: {current_b:.6f}')

    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()

    print(f'Final parameters - w: {final_w:.6f}, b: {final_b:.6f}, Final loss: {losses[-1]:.6f}')

    return {
        'model': model,
        'losses': losses,
        'w_values': w_values,
        'b_values': b_values,
        'final_w': final_w,
        'final_b': final_b
    }


# 1. 测试不同优化器
print("=" * 60)
print("1. Compare Different Optimizers")
print("=" * 60)

for opt_name, opt_class in optimizers_list:
    results[opt_name] = train_model(opt_name, opt_class, lr=0.01, epochs=1000)

# 2. 测试不同学习率
print("\n" + "=" * 60)
print("2. Compare Different Learning Rates")
print("=" * 60)

learning_rates = [0.001, 0.01, 0.1]
lr_results = {}

for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")
    lr_results[lr] = train_model('SGD', torch.optim.SGD, lr=lr, epochs=500)

# 3. 测试不同epoch数量
print("\n" + "=" * 60)
print("3. Compare Different Epoch Counts")
print("=" * 60)

epoch_counts = [100, 500, 1000]
epoch_results = {}

for epochs in epoch_counts:
    print(f"\nTesting epoch count: {epochs}")
    epoch_results[epochs] = train_model('SGD', torch.optim.SGD, lr=0.01, epochs=epochs)

# 创建可视化图表
fig = plt.figure(figsize=(20, 15))

# 1. 不同优化器性能比较
ax1 = plt.subplot(3, 3, 1)
colors = ['blue', 'red', 'green']
for i, opt_name in enumerate(results):
    if len(results[opt_name]['losses']) > 0:
        display_epochs = min(200, len(results[opt_name]['losses']))
        ax1.plot(range(display_epochs), results[opt_name]['losses'][:display_epochs],
                 label=opt_name, alpha=0.8, linewidth=2, color=colors[i])
ax1.set_title('Loss Curves of Different Optimizers', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. 不同优化器参数w调节过程
ax2 = plt.subplot(3, 3, 2)
for i, opt_name in enumerate(results):
    if len(param_history[opt_name]['w']) > 0:
        display_points = min(200, len(param_history[opt_name]['w']))
        ax2.plot(range(display_points), param_history[opt_name]['w'][:display_points],
                 label=f'{opt_name} w', alpha=0.8, linewidth=1.5, color=colors[i])
ax2.set_title('Weight (w) Adjustment Process', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Weight Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 不同优化器参数b调节过程
ax3 = plt.subplot(3, 3, 3)
for i, opt_name in enumerate(results):
    if len(param_history[opt_name]['b']) > 0:
        display_points = min(200, len(param_history[opt_name]['b']))
        ax3.plot(range(display_points), param_history[opt_name]['b'][:display_points],
                 label=f'{opt_name} b', alpha=0.8, linewidth=1.5, color=colors[i])
ax3.set_title('Bias (b) Adjustment Process', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Bias Value')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 参数空间轨迹 - SGD
ax4 = plt.subplot(3, 3, 4)
if len(param_history['SGD']['w']) > 10:
    indices = range(0, min(100, len(param_history['SGD']['w'])), 5)
    w_samples = [param_history['SGD']['w'][i] for i in indices]
    b_samples = [param_history['SGD']['b'][i] for i in indices]
    loss_samples = [param_history['SGD']['loss'][i] for i in indices]

    scatter = ax4.scatter(w_samples, b_samples, c=loss_samples,
                          cmap='viridis', alpha=0.7, s=30)
    ax4.plot(w_samples, b_samples, 'gray', alpha=0.5, linewidth=1)
    ax4.set_title('SGD: Optimization Trajectory', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Weight w')
    ax4.set_ylabel('Bias b')
    plt.colorbar(scatter, ax=ax4, label='Loss')
# 5. 参数空间轨迹 - Adam
ax5 = plt.subplot(3, 3, 5)
if len(param_history['Adam']['w']) > 10:
    indices = range(0, min(100, len(param_history['Adam']['w'])), 5)
    w_samples = [param_history['Adam']['w'][i] for i in indices]
    b_samples = [param_history['Adam']['b'][i] for i in indices]
    loss_samples = [param_history['Adam']['loss'][i] for i in indices]

    scatter = ax5.scatter(w_samples, b_samples, c=loss_samples,
                          cmap='viridis', alpha=0.7, s=30)
    ax5.plot(w_samples, b_samples, 'gray', alpha=0.5, linewidth=1)
    ax5.set_title('Adam: Optimization Trajectory', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Weight w')
    ax5.set_ylabel('Bias b')
    plt.colorbar(scatter, ax=ax5, label='Loss')
# 6. 参数空间轨迹 - RMSprop
ax6 = plt.subplot(3, 3, 6)
if len(param_history['RMSprop']['w']) > 10:
    indices = range(0, min(100, len(param_history['RMSprop']['w'])), 5)
    w_samples = [param_history['RMSprop']['w'][i] for i in indices]
    b_samples = [param_history['RMSprop']['b'][i] for i in indices]
    loss_samples = [param_history['RMSprop']['loss'][i] for i in indices]

    scatter = ax6.scatter(w_samples, b_samples, c=loss_samples,
                          cmap='viridis', alpha=0.7, s=30)
    ax6.plot(w_samples, b_samples, 'gray', alpha=0.5, linewidth=1)
    ax6.set_title('RMSprop: Optimization Trajectory', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Weight w')
    ax6.set_ylabel('Bias b')
    plt.colorbar(scatter, ax=ax6, label='Loss')

# 7. 不同学习率比较
ax7 = plt.subplot(3, 3, 7)
lr_colors = ['purple', 'orange', 'brown']
for i, (lr, result) in enumerate(lr_results.items()):
    if len(result['losses']) > 0:
        display_epochs = min(100, len(result['losses']))
        ax7.plot(range(display_epochs), result['losses'][:display_epochs],
                 label=f'LR={lr}', alpha=0.8, linewidth=2, color=lr_colors[i])
ax7.set_title('Loss Curves with Different Learning Rates', fontsize=12, fontweight='bold')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Loss')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_yscale('log')

# 8. 不同epoch数量比较
ax8 = plt.subplot(3, 3, 8)
epoch_colors = ['teal', 'magenta', 'darkred']
for i, (epochs, result) in enumerate(epoch_results.items()):
    if len(result['losses']) > 0:
        ax8.plot(range(len(result['losses'])), result['losses'],
                 label=f'Epochs={epochs}', alpha=0.8, linewidth=2, color=epoch_colors[i])
ax8.set_title('Training Results with Different Epoch Counts', fontsize=12, fontweight='bold')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Loss')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_yscale('log')

# 9. 最终模型拟合效果
ax9 = plt.subplot(3, 3, 9)
if 'SGD' in results:
    final_model = results['SGD']['model']
    x_range = torch.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = final_model(x_range)

    # 反标准化
    x_plot = x_range.numpy() * x_std + x_mean
    y_plot = y_pred.numpy() * y_std + y_mean
    x_original = x_data.numpy() * x_std + x_mean
    y_original = y_data.numpy() * y_std + y_mean

    ax9.scatter(x_original, y_original, alpha=0.6, label='True Data', s=10, color='blue')
    ax9.plot(x_plot, y_plot, 'r-', linewidth=2, label='Fitted Line')
    ax9.set_title('Final Model Fitting Result', fontsize=12, fontweight='bold')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出最终结果比较
print("\n" + "=" * 60)
print("Final Results Comparison")
print("=" * 60)

print(f"{'Optimizer':<10} {'Final w':<12} {'Final b':<12} {'Final Loss':<12}")
print("-" * 50)
for opt_name in results:
    if len(results[opt_name]['losses']) > 0:
        final_loss = results[opt_name]['losses'][-1]
        final_w = results[opt_name]['final_w']
        final_b = results[opt_name]['final_b']
        print(f"{opt_name:<10} {final_w:<12.6f} {final_b:<12.6f} {final_loss:<12.6f}")

# 测试预测
print("\n" + "=" * 60)
print("Prediction Test")
print("=" * 60)

if 'SGD' in results:
    final_model = results['SGD']['model']
    test_values = [50.0, 100.0, 150.0]

    print("Prediction Results:")
    for x_val in test_values:
        x_normalized = (x_val - x_mean) / x_std
        x_tensor = torch.Tensor([[x_normalized]])
        with torch.no_grad():
            y_pred_normalized = final_model(x_tensor)
            y_pred = y_pred_normalized.item() * y_std + y_mean
        print(f"x = {x_val:6.1f}, Predicted y = {y_pred:8.2f}")

print("\nTraining completed! Results saved to training_analysis.png")