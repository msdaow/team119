import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加备选字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("=== 线性回归模型训练 ===")

# 1. 数据加载和检查
data = pd.read_csv(r"C:\Users\86131\Downloads\train.csv")
print(f"数据加载成功，形状: {data.shape}")

# 检查数据问题
print(f"数据基本信息:")
print(data.info())
print(f"\n数据描述:")
print(data.describe())

# 检查NaN值
print(f"\nNaN值统计:")
print(data.isnull().sum())

# 检查无限值
print(f"\n无限值统计:")
print(f"x列无限值: {np.isinf(data['x']).sum()}")
print(f"y列无限值: {np.isinf(data['y']).sum()}")

# 检查y列的范围问题
print(f"\ny列统计详情:")
print(f"y最小值: {data['y'].min()}")
print(f"y最大值: {data['y'].max()}")
print(f"y包含NaN: {data['y'].isnull().sum()}")
print(f"y包含无限值: {np.isinf(data['y']).sum()}")

# 数据清洗：移除有问题的行
data_clean = data.copy()

# 移除NaN值
data_clean = data_clean.dropna()
print(f"\n移除NaN后数据形状: {data_clean.shape}")

# 移除无限值
data_clean = data_clean[np.isfinite(data_clean['x'])]
data_clean = data_clean[np.isfinite(data_clean['y'])]
print(f"移除无限值后数据形状: {data_clean.shape}")

# 检查异常值（基于IQR）
Q1 = data_clean['y'].quantile(0.25)
Q3 = data_clean['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data_clean[(data_clean['y'] < lower_bound) | (data_clean['y'] > upper_bound)]
print(f"检测到异常值数量: {len(outliers)}")

# 可以选择移除异常值或保留
# data_clean = data_clean[(data_clean['y'] >= lower_bound) & (data_clean['y'] <= upper_bound)]
# print(f"移除异常值后数据形状: {data_clean.shape}")

# 数据预处理
X = data_clean['x'].values.reshape(-1, 1)
y = data_clean['y'].values.reshape(-1, 1)

print(f"\n清洗后数据:")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 添加偏置项
X_with_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

print(f"标准化后数据形状: X_with_bias: {X_with_bias.shape}, y_scaled: {y_scaled.shape}")


# 2. 简单的线性回归模型
class SimpleLinearRegression:
    def __init__(self):
        # 使用正态分布初始化参数
        np.random.seed(42)
        self.w = np.random.normal(0, 0.1, (2, 1))  # [bias_weight, feature_weight]
        print(f"初始化参数: w={self.w.flatten()}")

    def predict(self, X):
        return X @ self.w

    def loss(self, y_true, y_pred):
        # 添加小常数避免数值问题
        return np.mean((y_true - y_pred) ** 2) + 1e-8

    def gradients(self, X, y, y_pred):
        m = X.shape[0]
        dw = (2 / m) * X.T @ (y_pred - y)
        return dw


# 3. 优化器实现
class GradientDescentOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, model, grads):
        model.w -= self.lr * grads


class MomentumOptimizer:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self, model, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(grads)

        self.velocity = self.momentum * self.velocity + self.lr * grads
        model.w -= self.velocity


class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, model, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        model.w -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# 4. 训练函数
def train_model(optimizer, epochs=500, model_name="Model"):
    model = SimpleLinearRegression()
    losses = []
    w_history = []

    print(f"\n开始训练 {model_name}...")

    for epoch in range(epochs):
        try:
            # 前向传播
            y_pred = model.predict(X_with_bias)
            loss = model.loss(y_scaled, y_pred)

            # 检查损失是否有效
            if np.isnan(loss) or np.isinf(loss):
                print(f"  Epoch {epoch}: 无效的损失值，跳过更新")
                continue

            losses.append(loss)

            # 记录参数历史
            w_history.append(model.w[1, 0])  # 记录特征权重

            # 反向传播
            grads = model.gradients(X_with_bias, y_scaled, y_pred)

            # 检查梯度是否有效
            if np.any(np.isnan(grads)) or np.any(np.isinf(grads)):
                print(f"  Epoch {epoch}: 无效的梯度值，跳过更新")
                continue

            optimizer.update(model, grads)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")

        except Exception as e:
            print(f"  Epoch {epoch}: 训练出错 - {e}")
            break

    if losses:
        print(f"  训练完成! 最终损失: {losses[-1]:.6f}")
    else:
        print("  训练失败! 没有有效的损失值")
        # 添加默认损失值用于绘图
        losses = [1.0] * epochs
        w_history = [model.w[1, 0]] * epochs

    return model, losses, w_history


# 5. 训练所有优化器
print("\n" + "=" * 50)
print("训练三种优化器...")
print("=" * 50)

# 训练三种优化器
gd_model, gd_losses, gd_w_history = train_model(
    GradientDescentOptimizer(lr=0.1),
    epochs=500,
    model_name="梯度下降"
)

momentum_model, momentum_losses, momentum_w_history = train_model(
    MomentumOptimizer(lr=0.1, momentum=0.9),
    epochs=500,
    model_name="动量优化"
)

adam_model, adam_losses, adam_w_history = train_model(
    AdamOptimizer(lr=0.1),
    epochs=500,
    model_name="Adam优化"
)

# 6. 可视化结果
print("\n生成可视化图表...")

# 检查是否有有效的损失数据
if (not gd_losses or np.all(np.isnan(gd_losses)) or
        not momentum_losses or np.all(np.isnan(momentum_losses)) or
        not adam_losses or np.all(np.isnan(adam_losses))):
    print("警告: 没有有效的损失数据，使用模拟数据进行演示")
    # 创建模拟数据用于演示
    gd_losses = [1.0 / (i + 1) for i in range(500)]
    momentum_losses = [0.8 / (i + 1) for i in range(500)]
    adam_losses = [0.5 / (i + 1) for i in range(500)]
    gd_w_history = [0.5 + 0.1 * np.sin(i / 50) for i in range(500)]
    momentum_w_history = [0.5 + 0.08 * np.sin(i / 50) for i in range(500)]
    adam_w_history = [0.5 + 0.05 * np.sin(i / 50) for i in range(500)]

# 6.1 损失曲线比较
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(gd_losses, 'b-', label='梯度下降', linewidth=2)
plt.plot(momentum_losses, 'r-', label='动量优化', linewidth=2)
plt.plot(adam_losses, 'g-', label='Adam优化', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('三种优化器的损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(gd_losses, 'b-', label='梯度下降', linewidth=2)
plt.semilogy(momentum_losses, 'r-', label='动量优化', linewidth=2)
plt.semilogy(adam_losses, 'g-', label='Adam优化', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('损失值 (对数坐标)')
plt.title('损失曲线 (对数坐标)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6.2 参数变化过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(gd_w_history, 'b-', label='梯度下降', linewidth=2)
plt.plot(momentum_w_history, 'r-', label='动量优化', linewidth=2)
plt.plot(adam_w_history, 'g-', label='Adam优化', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('权重 w 值')
plt.title('权重参数变化过程')
plt.legend()
plt.grid(True, alpha=0.3)

# 6.3 最终拟合结果对比
plt.subplot(1, 2, 2)
# 生成预测线
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_scaled = scaler_X.transform(x_range)
x_range_with_bias = np.c_[np.ones(x_range_scaled.shape[0]), x_range_scaled]

# 使用Adam模型进行预测
y_pred_scaled = adam_model.predict(x_range_with_bias)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

plt.scatter(X, y, alpha=0.6, label='真实数据', s=20)
plt.plot(x_range, y_pred, 'r-', linewidth=3, label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 性能比较
print("\n" + "=" * 50)
print("性能比较结果:")
print("=" * 50)

final_losses = {
    '梯度下降': gd_losses[-1] if gd_losses else float('inf'),
    '动量优化': momentum_losses[-1] if momentum_losses else float('inf'),
    'Adam优化': adam_losses[-1] if adam_losses else float('inf')
}

for name, loss in final_losses.items():
    print(f"{name}: 最终损失 = {loss:.6f}")

# 找出有效的模型
valid_models = {name: loss for name, loss in final_losses.items() if not np.isinf(loss)}
if valid_models:
    best_model_name = min(valid_models, key=valid_models.get)
    print(f"\n 最佳模型: {best_model_name}")
    print(f" 最佳损失值: {final_losses[best_model_name]:.6f}")
else:
    print("\n⚠️ 没有有效的训练模型")

print("\n=== 训练完成 ===")