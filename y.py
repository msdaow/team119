import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 用pandas读取数据
df = pd.read_csv(r"C:\Users\86131\Downloads\train.csv")

# 清理数据 - 删除包含NaN值的行
df_clean = df.dropna()
print(f"清理前数据形状: {df.shape}")
print(f"清理后数据形状: {df_clean.shape}")

# 2. 训练y=wx+b模型
X = df_clean[['x']].values
y = df_clean['y'].values

# 使用线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取训练结果
w = model.coef_[0]
b = model.intercept_

print(f"\n训练结果:")
print(f"斜率 w = {w:.4f}")
print(f"截距 b = {b:.4f}")
print(f"R²分数 = {model.score(X, y):.4f}")

# 计算预测值和损失
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"均方误差(MSE) = {mse:.4f}")

# 3. 绘制w和loss之间的关系、b和loss之间的关系
def calculate_loss(w_val, b_val, X, y):
    """计算给定w和b时的均方误差"""
    y_pred = w_val * X.flatten() + b_val
    return mean_squared_error(y, y_pred)

# 生成w和b的取值范围
w_range = np.linspace(w - 0.5, w + 0.5, 100)
b_range = np.linspace(b - 10, b + 10, 100)

# 计算不同w值时的loss（固定b为最优值）
loss_w = [calculate_loss(w_val, b, X, y) for w_val in w_range]

# 计算不同b值时的loss（固定w为最优值）
loss_b = [calculate_loss(w, b_val, X, y) for b_val in b_range]

# 创建图表
plt.figure(figsize=(15, 5))

# 子图1: w和loss的关系
plt.subplot(1, 2, 1)
plt.plot(w_range, loss_w, 'b-', linewidth=2, label='Loss曲线')
plt.axvline(x=w, color='r', linestyle='--', label=f'最优w = {w:.4f}')
plt.xlabel('权重 w')
plt.ylabel('均方误差 Loss')
plt.title('权重w与损失函数关系')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: b和loss的关系
plt.subplot(1, 2, 2)
plt.plot(b_range, loss_b, 'g-', linewidth=2, label='Loss曲线')
plt.axvline(x=b, color='r', linestyle='--', label=f'最优b = {b:.4f}')
plt.xlabel('偏置 b')
plt.ylabel('均方误差 Loss')
plt.title('偏置b与损失函数关系')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出统计信息
print(f"\n数据统计信息:")
print(f"x的均值: {X.mean():.4f}")
print(f"y的均值: {y.mean():.4f}")
print(f"x的标准差: {X.std():.4f}")
print(f"y的标准差: {y.std():.4f}")
print(f"x和y的相关系数: {np.corrcoef(X.flatten(), y)[0,1]:.4f}")

# 模型评估
print(f"\n模型评估:")
print(f"平均绝对误差: {np.mean(np.abs(y - y_pred)):.4f}")
print(f"均方根误差: {np.sqrt(mse):.4f}")