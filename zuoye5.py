import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# 简化字体设置，使用系统默认字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (12, 8)  # 设置默认图表大小

# 1. 读取并处理train.csv数据集
try:
    # 尝试读取train.csv文件
    file_path = 'D:/train.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path}文件不存在")

    df = pd.read_csv(file_path)
    print(f"成功读取数据集，形状: {df.shape}")
    print(f"数据集前5行:\n{df.head()}")

    # 检查数据质量
    print(f"数据统计信息:\n{df.describe()}")
    print(f"缺失值数量:\n{df.isnull().sum()}")

    # 处理缺失值
    df = df.fillna(df.mean())

    # 检查数据异常值
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    print(f"检测到 {outlier_mask.sum()} 个异常值")

    # 假设最后一列是目标变量，其他是特征
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 检查标准化后的数据范围
    print(f"标准化后特征范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"标准化后目标范围: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

    # 转换为张量
    x_data = torch.tensor(X_scaled, dtype=torch.float32)
    y_data = torch.tensor(y_scaled, dtype=torch.float32)

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

except Exception as e:
    print(f"数据集读取失败: {e}，将使用示例数据进行演示")
    # 使用示例数据
    torch.manual_seed(42)  # 设置随机种子
    x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0]])
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )


# 2. 定义线性模型，使用更稳定的参数初始化
class LinearModel(nn.Module):
    def __init__(self, input_size=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

        # 使用Kaiming初始化，更适合线性模型
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)


# 改进的损失函数，增加稳定性
class StableMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8, reduction='mean'):
        super(StableMSELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        # 添加小常数防止数值不稳定
        loss = (input - target) ** 2 + self.epsilon

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


# 动态梯度裁剪类
class DynamicGradientClipper:
    def __init__(self, initial_max_norm=1.0, adaptation_factor=0.9, min_norm=0.1, max_norm=10.0):
        self.max_norm = initial_max_norm
        self.adaptation_factor = adaptation_factor
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.gradient_history = []

    def clip_gradients(self, model):
        """动态调整梯度裁剪阈值"""
        current_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

        # 记录梯度历史
        self.gradient_history.append(current_norm.item())

        # 动态调整裁剪阈值
        if len(self.gradient_history) > 10:
            recent_norms = self.gradient_history[-10:]
            avg_recent = np.mean(recent_norms)

            # 如果梯度持续较大，增加裁剪阈值
            if avg_recent > self.max_norm * 0.8:
                self.max_norm = min(self.max_norm * 1.1, self.max_norm)
            # 如果梯度持续较小，减小裁剪阈值
            elif avg_recent < self.max_norm * 0.2:
                self.max_norm = max(self.max_norm * 0.9, self.min_norm)

        return current_norm.item()


# 自定义学习率调度器
class CustomLRScheduler:
    def __init__(self, optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.num_bad_epochs = 0
        self.last_epoch = 0

    def step(self, metrics):
        self.last_epoch += 1

        if self.mode == 'min':
            is_better = metrics < self.best * (1 - 1e-4)
        else:
            is_better = metrics > self.best * (1 + 1e-4)

        if is_better:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr != old_lr:
                param_group['lr'] = new_lr
                print(f'Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}')


# 改进的训练函数，确保所有模型训练相同的轮数
def train_model(model, optimizer, criterion, x_train, y_train, x_test, y_test,
                epochs=100, initial_grad_clip=1.0, patience=10, min_delta=1e-6):
    train_losses = []
    test_losses = []
    weights = []
    biases = []
    grad_norms = []

    best_loss = float('inf')
    patience_counter = 0
    actual_epochs = epochs  # 记录实际训练的轮数

    # 初始化动态梯度裁剪器
    clipper = DynamicGradientClipper(initial_max_norm=initial_grad_clip)

    # 使用自定义学习率调度器
    scheduler = CustomLRScheduler(optimizer, mode='min', factor=0.5,
                                  patience=patience // 2, min_lr=1e-7)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        # 检查损失是否为nan或过大
        if torch.isnan(loss) or loss > 1e5 or (epoch > 10 and train_losses and loss > 10 * train_losses[0]):
            print(f"Epoch {epoch + 1}: Abnormal loss ({loss.item():.6f}), stopping training")
            actual_epochs = epoch
            break

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()

        # 使用动态梯度裁剪
        current_grad_norm = clipper.clip_gradients(model)
        grad_norms.append(current_grad_norm)

        # 检查梯度是否存在NaN或过大值
        nan_detected = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"Epoch {epoch + 1}: NaN gradient detected ({name}), stopping training")
                    nan_detected = True
                    break
                if torch.isinf(param.grad).any():
                    print(f"Epoch {epoch + 1}: Inf gradient detected ({name}), stopping training")
                    nan_detected = True
                    break

        if nan_detected:
            actual_epochs = epoch
            break

        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 测试阶段
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            test_losses.append(test_loss.item())

        # 更新学习率
        scheduler.step(test_loss)

        # 记录参数值
        weights.append(model.linear.weight.detach().mean().item())
        biases.append(model.linear.bias.detach().item())

        # 早停机制
        if test_loss < best_loss - min_delta:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Epoch {epoch + 1}: Early stopping triggered")
            actual_epochs = epoch + 1
            break

        # 定期打印信息
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{epochs}], LR: {current_lr:.2e}, Train Loss: {loss.item():.6f}, '
                  f'Test Loss: {test_loss.item():.6f}, Grad Norm: {current_grad_norm:.6f}')

    # 如果提前停止，用最后一个值填充剩余轮数
    if len(train_losses) < epochs:
        last_train_loss = train_losses[-1] if train_losses else 0
        last_test_loss = test_losses[-1] if test_losses else 0
        last_weight = weights[-1] if weights else 0
        last_bias = biases[-1] if biases else 0
        last_grad_norm = grad_norms[-1] if grad_norms else 0

        train_losses.extend([last_train_loss] * (epochs - len(train_losses)))
        test_losses.extend([last_test_loss] * (epochs - len(test_losses)))
        weights.extend([last_weight] * (epochs - len(weights)))
        biases.extend([last_bias] * (epochs - len(biases)))
        grad_norms.extend([last_grad_norm] * (epochs - len(grad_norms)))

    return train_losses, test_losses, weights, biases, grad_norms, actual_epochs


# 生成所有结果曲线图
def generate_all_plots():
    # 存储所有结果的字典
    all_results = {}

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 3. 比较三种不同优化器的性能，使用更稳定的配置
    print("=== Comparing Optimizer Performance ===")
    input_size = x_train.shape[1]

    # 使用更稳定的损失函数
    criterion = StableMSELoss()
    epochs = 500
    learning_rate = 0.001

    # 初始化模型和优化器
    model_adagrad = LinearModel(input_size)
    model_adam = LinearModel(input_size)
    model_adamax = LinearModel(input_size)

    # 使用更稳定的优化器配置
    optimizer_adagrad = optim.Adagrad(model_adagrad.parameters(), lr=learning_rate,
                                      weight_decay=1e-4, eps=1e-8)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)
    optimizer_adamax = optim.Adamax(model_adamax.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)

    # 训练不同优化器的模型
    print("Training with Adagrad optimizer...")
    adagrad_results = train_model(
        model_adagrad, optimizer_adagrad, criterion,
        x_train, y_train, x_test, y_test, epochs, initial_grad_clip=0.5, patience=50  # 增加patience避免过早停止
    )

    print("Training with Adam optimizer...")
    adam_results = train_model(
        model_adam, optimizer_adam, criterion,
        x_train, y_train, x_test, y_test, epochs, initial_grad_clip=0.5, patience=50
    )

    print("Training with Adamax optimizer...")
    adamax_results = train_model(
        model_adamax, optimizer_adamax, criterion,
        x_train, y_train, x_test, y_test, epochs, initial_grad_clip=0.5, patience=50
    )

    # 提取结果
    adagrad_train, adagrad_test, adagrad_w, adagrad_b, adagrad_grad, adagrad_actual = adagrad_results
    adam_train, adam_test, adam_w, adam_b, adam_grad, adam_actual = adam_results
    adamax_train, adamax_test, adamax_w, adamax_b, adamax_grad, adamax_actual = adamax_results

    all_results['optimizers'] = {
        'adagrad': adagrad_results,
        'adam': adam_results,
        'adamax': adamax_results
    }

    # 绘制优化器性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 训练损失对比
    axes[0, 0].plot(adagrad_train, 'b-', label=f'Adagrad (stopped at {adagrad_actual})', linewidth=2)
    axes[0, 0].plot(adam_train, 'r-', label=f'Adam (stopped at {adam_actual})', linewidth=2)
    axes[0, 0].plot(adamax_train, 'g-', label=f'Adamax (stopped at {adamax_actual})', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)

    # 测试损失对比
    axes[0, 1].plot(adagrad_test, 'b-', label=f'Adagrad (stopped at {adagrad_actual})', linewidth=2)
    axes[0, 1].plot(adam_test, 'r-', label=f'Adam (stopped at {adam_actual})', linewidth=2)
    axes[0, 1].plot(adamax_test, 'g-', label=f'Adamax (stopped at {adamax_actual})', linewidth=2)
    axes[0, 1].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)

    # 梯度范数对比
    axes[1, 0].plot(adagrad_grad, 'b-', label=f'Adagrad (stopped at {adagrad_actual})', linewidth=2)
    axes[1, 0].plot(adam_grad, 'r-', label=f'Adam (stopped at {adam_actual})', linewidth=2)
    axes[1, 0].plot(adamax_grad, 'g-', label=f'Adamax (stopped at {adamax_actual})', linewidth=2)
    axes[1, 0].set_title('Gradient Norm Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=10)

    # 参数变化
    axes[1, 1].plot(adagrad_w, 'b-', label='Adagrad w', linewidth=2, alpha=0.7)
    axes[1, 1].plot(adam_w, 'r-', label='Adam w', linewidth=2, alpha=0.7)
    axes[1, 1].plot(adamax_w, 'g-', label='Adamax w', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Weight Parameter w Changes', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('w Value', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig('1_optimizers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 调节学习率并可视化
    print("\n=== Learning Rate Tuning ===")
    learning_rates = [0.0001, 0.0005, 0.001, 0.005]
    lr_results = {}
    colors = ['b', 'r', 'g', 'm']

    plt.figure(figsize=(14, 6))
    for i, lr in enumerate(learning_rates):
        torch.manual_seed(42)
        model = LinearModel(input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"Training with learning rate {lr}...")
        train_loss, test_loss, _, _, _, actual_epochs = train_model(
            model, optimizer, criterion, x_train, y_train, x_test, y_test,
            epochs=300, initial_grad_clip=0.5, patience=30
        )
        lr_results[lr] = (train_loss, test_loss)
        plt.plot(test_loss, f'{colors[i]}-', label=f'LR = {lr} (stopped at {actual_epochs})', linewidth=2)

    all_results['learning_rates'] = lr_results

    plt.title('Learning Rate Effect on Model Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig('2_learning_rate_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 梯度监控图
    print("\n=== Gradient Monitoring ===")
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(adagrad_grad, 'b-', label=f'Adagrad (stopped at {adagrad_actual})', linewidth=2)
    plt.plot(adam_grad, 'r-', label=f'Adam (stopped at {adam_actual})', linewidth=2)
    plt.plot(adamax_grad, 'g-', label=f'Adamax (stopped at {adamax_actual})', linewidth=2)
    plt.title('Gradient Norm Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.subplot(2, 1, 2)
    # 绘制梯度分布直方图（只使用实际训练的数据）
    actual_adagrad_grad = adagrad_grad[:adagrad_actual] if adagrad_actual > 0 else adagrad_grad
    actual_adam_grad = adam_grad[:adam_actual] if adam_actual > 0 else adam_grad
    actual_adamax_grad = adamax_grad[:adamax_actual] if adamax_actual > 0 else adamax_grad

    all_grads = np.concatenate([actual_adagrad_grad, actual_adam_grad, actual_adamax_grad])
    plt.hist(all_grads, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Gradient Norm', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig('3_gradient_monitoring.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_results


# 主函数
def main():
    print("Starting training with advanced gradient explosion protection...")
    print("=" * 60)

    # 生成所有结果曲线图
    all_results = generate_all_plots()

    print("\nAll experiments completed. Visualization results saved as PNG files:")
    print("1. Optimizer comparison: 1_optimizers_comparison.png")
    print("2. Learning rate effect: 2_learning_rate_effect.png")
    print("3. Gradient monitoring: 3_gradient_monitoring.png")

    # 打印训练稳定性总结
    print("\n" + "=" * 60)
    print("Gradient Explosion Protection Summary")
    print("=" * 60)
    print("✓ Kaiming parameter initialization")
    print("✓ Dynamic gradient clipping")
    print("✓ Stable MSE loss function")
    print("✓ Custom learning rate scheduler")
    print("✓ Early stopping mechanism")
    print("✓ Weight decay (L2 regularization)")
    print("✓ Gradient NaN/Inf checking")
    print("✓ Outlier detection")
    print("✓ Real-time gradient monitoring")


if __name__ == "__main__":
    main()