import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns  # 用于更美观的可视化

# 设置字体（增加更多中文字体选项以提高兼容性）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示
sns.set_style("whitegrid")  # 设置seaborn样式

# 读取数据并处理异常值
try:
    df = pd.read_csv('F:/MobileFile/train.csv')
    print("Successfully read the dataset")

    # 检查并处理异常值 (x=3530.16可能是异常值)
    print("\nData outlier check:")
    x_max = df['x'].max()
    if x_max > 1000:  # 假设x值超过1000可能是异常值
        print(f"Found an abnormal maximum value in x: {x_max}, please check if it's valid data")
except Exception as e:
    print(f"Error reading dataset: {e}")
    # 使用示例数据进行演示
    df = pd.DataFrame({
        'x': [24, 50, 15, 38, 87, 36, 12, 81, 25, 5, 16, 16, 24],
        'y': [21.55, 47.46, 17.22, 36.59, 87.29, 32.46, 10.78, 80.76,
              24.61, 6.96, 11.24, 13.53, 24.60]
    })
    print("Using sample data for demonstration")

# 检查并处理缺失值
print("\nMissing value statistics:")
print(df.isnull().sum())

# 处理缺失值：使用列均值填充
if df.isnull().any().any():
    df = df.fillna(df.mean())
    print("Filled missing values with column means")
else:
    print("No missing values in the dataset")

print(f"\nDataset basic information:")
print(f"Dataset size: {len(df)} rows")
print(f"x column statistics: Min={df['x'].min():.2f}, Max={df['x'].max():.2f}, Mean={df['x'].mean():.2f}")
print(f"y column statistics: Min={df['y'].min():.2f}, Max={df['y'].max():.2f}, Mean={df['y'].mean():.2f}")

# 数据标准化（使用对数变换处理可能的异常值）
x_mean, x_std = df['x'].mean(), df['x'].std()
y_mean, y_std = df['y'].mean(), df['y'].std()

# 对x进行对数变换处理异常大的值
df['x_log'] = df['x'].apply(lambda x: np.log(x + 1) if x > 0 else 0)
x_mean_log, x_std_log = df['x_log'].mean(), df['x_log'].std()

# 标准化数据
x_normalized = (df['x_log'].values - x_mean_log) / x_std_log
y_normalized = (df['y'].values - y_mean) / y_std

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_normalized, y_normalized, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
x_train_tensor = torch.Tensor(x_train.reshape(-1, 1))
y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))
x_test_tensor = torch.Tensor(x_test.reshape(-1, 1))
y_test_tensor = torch.Tensor(y_test.reshape(-1, 1))

print(f"\nNormalized data range:")
print(f"x_train: [{x_train_tensor.min():.2f}, {x_train_tensor.max():.2f}]")
print(f"y_train: [{y_train_tensor.min():.2f}, {y_train_tensor.max():.2f}]")


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 更严格的正态分布初始化，避免梯度问题
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.01)

    def forward(self, x):
        return self.linear(x)


# 训练模型函数，增加早停机制（核心修改：解决LBFGS警告）
def train_model(optimizer_class, optimizer_name, lr=0.01, epochs=500, patience=50):
    """训练模型并返回训练历史和评估结果，包含早停机制"""
    model = LinearModel()
    criterion = nn.MSELoss()

    # 选择优化器
    if optimizer_class == torch.optim.LBFGS:
        # LBFGS需设置合适的history_size和max_iter（控制每次step的迭代次数）
        optimizer = optimizer_class(model.parameters(), lr=lr, history_size=10, max_iter=20)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    # 记录训练过程
    train_losses = []
    test_losses = []
    weights = []
    biases = []
    best_test_loss = float('inf')
    no_improvement = 0

    # 梯度裁剪阈值
    grad_clip = 1.0

    # -------------------------- 核心修改：解决LBFGS警告 --------------------------
    def closure():
        optimizer.zero_grad()  # 清空梯度
        y_pred = model(x_train_tensor)  # 前向传播
        loss = criterion(y_pred, y_train_tensor)  # 计算损失
        loss.backward()  # 反向传播（保留梯度计算）
        # 关键：返回分离梯度后的损失，避免后续标量转换冲突
        return loss.detach()
    # ---------------------------------------------------------------------------

    for epoch in range(epochs):
        if optimizer_class == torch.optim.LBFGS:
            try:
                # LBFGS通过closure完成一次优化步骤
                optimizer.step(closure)
                # 重新计算损失用于记录（用no_grad避免额外计算图）
                with torch.no_grad():
                    y_pred = model(x_train_tensor)
                    train_loss = criterion(y_pred, y_train_tensor).item()

                    # 计算测试集损失
                    y_pred_test = model(x_test_tensor)
                    test_loss = criterion(y_pred_test, y_test_tensor).item()
            except Exception as e:
                print(f"{optimizer_name} - Epoch {epoch} error: {e}")
                break
        else:
            # 其他优化器（Adamax/ASGD）的训练逻辑
            model.train()
            y_pred = model(x_train_tensor)
            train_loss = criterion(y_pred, y_train_tensor).item()

            optimizer.zero_grad()
            loss = criterion(y_pred, y_train_tensor)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # 评估测试集
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test_tensor)
                test_loss = criterion(y_pred_test, y_test_tensor).item()

        # 检查损失是否为NaN（防止训练崩溃）
        if np.isnan(train_loss) or np.isnan(test_loss):
            print(f"{optimizer_name} - Epoch {epoch} loss became NaN, stopping training")
            break

        # 记录训练过程
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

        # 早停机制（避免过拟合，提高训练效率）
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"{optimizer_name} - No improvement in test loss for {patience} consecutive epochs, early stopping")
                break

        # 每50轮打印一次训练状态
        if epoch % 50 == 0:
            print(f'{optimizer_name} - Epoch {epoch}, '
                  f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # 计算最终模型指标
    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()
    final_train_loss = train_losses[-1] if train_losses else float('inf')
    final_test_loss = test_losses[-1] if test_losses else float('inf')

    print(f'{optimizer_name} final results: '
          f'w={final_w:.4f}, b={final_b:.4f}, '
          f'Train Loss={final_train_loss:.4f}, Test Loss={final_test_loss:.4f}')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'weights': weights,
        'biases': biases,
        'final_w': final_w,
        'final_b': final_b,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'model': model
    }


# 第一部分：比较三种优化器（Adamax、ASGD、LBFGS）
print("\n" + "=" * 50)
print("Part 1: Comparison of Three Optimizers")
print("=" * 50)

# 为不同优化器设置适配的学习率（关键：不同优化器对学习率敏感度不同）
optimizers = [
    (torch.optim.Adamax, 'Adamax', 0.01),    # Adamax对学习率不敏感，用0.01
    (torch.optim.ASGD, 'ASGD', 0.001),      # ASGD收敛慢，用更小的学习率避免震荡
    (torch.optim.LBFGS, 'LBFGS', 0.01)      # LBFGS是二阶优化，学习率可与Adamax一致
]

results = {}
for optimizer_class, name, lr in optimizers:
    print(f"\nTraining {name} optimizer (Learning Rate: {lr})...")
    results[name] = train_model(optimizer_class, name, lr=lr, epochs=500)

# 可视化三种优化器的性能对比
plt.figure(figsize=(18, 12))

# 1. 训练损失对比（对数坐标更易观察收敛趋势）
plt.subplot(2, 2, 1)
for name in results:
    if results[name]['train_losses']:
        plt.plot(results[name]['train_losses'], label=f'{name} (Train)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison of Three Optimizers')
plt.legend()
plt.grid(True)
plt.yscale('log')

# 2. 测试损失对比（评估泛化能力）
plt.subplot(2, 2, 2)
for name in results:
    if results[name]['test_losses']:
        plt.plot(results[name]['test_losses'], label=f'{name} (Test)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss Comparison of Three Optimizers')
plt.legend()
plt.grid(True)
plt.yscale('log')

# 3. 权重参数w的变化过程（观察参数收敛稳定性）
plt.subplot(2, 2, 3)
for name in results:
    if results[name]['weights']:
        plt.plot(results[name]['weights'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('Change Process of Weight Parameter w')
plt.legend()
plt.grid(True)

# 4. 偏置参数b的变化过程（观察偏置收敛趋势）
plt.subplot(2, 2, 4)
for name in results:
    if results[name]['biases']:
        plt.plot(results[name]['biases'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title('Change Process of Bias Parameter b')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('optimizers_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 第二部分：调节学习率对模型的影响（以Adamax为例）
print("\n" + "=" * 50)
print("Part 2: Influence of Learning Rate")
print("=" * 50)

learning_rates = [0.001, 0.01, 0.05, 0.1]  # 覆盖小、中、大三种学习率范围
lr_results = {}

plt.figure(figsize=(18, 12))

for i, lr in enumerate(learning_rates):
    print(f"\nTraining Adamax optimizer with learning rate {lr}...")
    result = train_model(torch.optim.Adamax, 'Adamax', lr=lr, epochs=300)
    lr_results[lr] = result

    # 绘制单学习率的损失曲线
    plt.subplot(2, 2, i + 1)
    if result['train_losses']:
        plt.plot(result['train_losses'], label=f'LR={lr} (Train)')
        if result['test_losses']:
            plt.plot(result['test_losses'], label=f'LR={lr} (Test)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Influence of Learning Rate {lr} on Adamax Optimizer')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

plt.tight_layout()
plt.savefig('learning_rate_effect.png', dpi=300, bbox_inches='tight')
plt.show()

# 第三部分：调节训练轮数（Epoch）对模型的影响（以Adamax为例）
print("\n" + "=" * 50)
print("Part 3: Influence of Epoch Number")
print("=" * 50)

epochs_list = [50, 100, 200, 300]  # 覆盖短、中、长期训练场景
epoch_results = {}

plt.figure(figsize=(18, 12))

for i, epochs in enumerate(epochs_list):
    print(f"\nTraining Adamax optimizer for {epochs} epochs...")
    result = train_model(torch.optim.Adamax, 'Adamax', lr=0.01, epochs=epochs)
    epoch_results[epochs] = result

    # 绘制单轮数的损失曲线
    plt.subplot(2, 2, i + 1)
    if result['train_losses']:
        plt.plot(result['train_losses'], label=f'Epochs={epochs} (Train)')
        if result['test_losses']:
            plt.plot(result['test_losses'], label=f'Epochs={epochs} (Test)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Influence of Epoch Number {epochs} on Adamax Optimizer')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

plt.tight_layout()
plt.savefig('epochs_effect.png', dpi=300, bbox_inches='tight')
plt.show()

# 找到性能最优的模型（以测试损失为核心指标，避免过拟合）
best_result = None
best_name = None
min_test_loss = float('inf')

for name in results:
    if (results[name]['final_test_loss'] < min_test_loss and
            not np.isnan(results[name]['final_test_loss']) and
            results[name]['final_test_loss'] != float('inf')):
        min_test_loss = results[name]['final_test_loss']
        best_result = results[name]
        best_name = name

if best_result is not None:
    print("\n" + "=" * 50)
    print("Best-Performing Model:")
    print("=" * 50)
    print(f"Optimizer: {best_name}")
    print(f"Final Weight w: {best_result['final_w']:.6f}")
    print(f"Final Bias b: {best_result['final_b']:.6f}")
    print(f"Train Loss: {best_result['final_train_loss']:.6f}")
    print(f"Test Loss: {best_result['final_test_loss']:.6f}")

    # 保存最优模型（便于后续复用）
    torch.save(best_result['model'].state_dict(), 'best_linear_model.pth')
    print("Best model saved as 'best_linear_model.pth'")

    # 最终可视化：最优模型的完整性能
    plt.figure(figsize=(18, 12))

    # 1. 原始数据分布
    plt.subplot(2, 2, 1)
    x_original = df['x'].values
    y_original = df['y'].values
    plt.scatter(x_original, y_original, alpha=0.6, color='blue', label='Original Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Data Distribution')
    plt.grid(True)

    # 2. 最优模型拟合效果（反推原始数据尺度）
    plt.subplot(2, 2, 2)
    plt.scatter(x_original, y_original, alpha=0.6, color='blue', label='Original Data')

    # 生成预测线（覆盖原始x的全部范围）
    x_range = torch.linspace(torch.tensor(x_original).min(), torch.tensor(x_original).max(), 100).reshape(-1, 1)
    x_range_log = torch.log(x_range + 1)  # 应用与训练一致的对数变换
    x_range_normalized = (x_range_log.numpy() - x_mean_log) / x_std_log  # 标准化
    x_range_tensor = torch.Tensor(x_range_normalized)

    # 无梯度预测（避免额外计算）
    with torch.no_grad():
        y_pred_normalized = best_result['model'](x_range_tensor)
        y_pred_original = y_pred_normalized.numpy() * y_std + y_mean  # 反标准化回原始尺度

    plt.plot(x_range.numpy(), y_pred_original, 'r-', linewidth=2,
             label=f'Best Fit: y = {best_result["final_w"]:.3f}x + {best_result["final_b"]:.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting Effect of the Best Model')
    plt.legend()
    plt.grid(True)

    # 3. 损失函数收敛过程
    plt.subplot(2, 2, 3)
    if best_result['train_losses']:
        plt.plot(best_result['train_losses'], label='Train Loss')
        if best_result['test_losses']:
            plt.plot(best_result['test_losses'], label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Process of Loss Function for the Best Model')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # 4. 参数收敛过程（权重w和偏置b）
    plt.subplot(2, 2, 4)
    if best_result['weights'] and best_result['biases']:
        plt.plot(best_result['weights'], label='Weight (w)')
        plt.plot(best_result['biases'], label='Bias (b)')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Value')
    plt.title('Convergence Process of Parameters for the Best Model')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('best_model_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No valid model results found")

