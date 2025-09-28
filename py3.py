import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取数据并处理缺失值
df = pd.read_csv('D:/train.csv')

# 查看缺失值情况
print("原始数据缺失值情况：")
print(df.isnull().sum())

# 处理缺失值：使用列的平均值填充
df['x'] = df['x'].fillna(df['x'].mean())
df['y'] = df['y'].fillna(df['y'].mean())

# 确认缺失值已处理
print("\n处理后缺失值情况：")
print(df.isnull().sum())

# 提取处理后的x和y数据
x_data = df['x'].values
y_data = df['y'].values


# 2. 定义模型和损失函数
def forward(x, w, b):
    """前向传播：计算预测值 y = wx + b"""
    return x * w + b


def loss(x, y, w, b):
    """计算均方误差损失"""
    y_pred = forward(x, w, b)
    return np.mean((y_pred - y) ** 2)


# 3. 搜索最优参数w和b
# 定义参数搜索范围
w_range = np.arange(0.0, 4.1, 0.1)  # w的搜索范围
b_range = np.arange(-2.0, 2.1, 0.1)  # b的搜索范围

# 存储参数和对应的损失值
w_list = []
b_list = []
loss_list = []

# 遍历所有可能的参数组合
for w in w_range:
    for b in b_range:
        current_loss = loss(x_data, y_data, w, b)
        w_list.append(w)
        b_list.append(b)
        loss_list.append(current_loss)

# 找到最优参数（损失最小的参数组合）
min_loss_index = np.argmin(loss_list)
best_w = w_list[min_loss_index]
best_b = b_list[min_loss_index]
min_loss = loss_list[min_loss_index]

print(f"\n最优参数: w={best_w:.1f}, b={best_b:.1f}, 最小损失={min_loss:.4f}")

# 4. 准备可视化数据
# 提取最优b对应的w和损失值（用于绘制w与loss关系）
optimal_b_mask = np.array(b_list) == best_b
w_values = np.array(w_list)[optimal_b_mask]
loss_w_values = np.array(loss_list)[optimal_b_mask]

# 提取最优w对应的b和损失值（用于绘制b与loss关系）
optimal_w_mask = np.array(w_list) == best_w
b_values = np.array(b_list)[optimal_w_mask]
loss_b_values = np.array(loss_list)[optimal_w_mask]

# 5. 绘制可视化图表 - w和b与损失的关系
plt.figure(figsize=(14, 6))

# 子图1：w与loss的关系（固定b为最优值）
plt.subplot(1, 2, 1)
plt.plot(w_values, loss_w_values, 'b-')
plt.axvline(x=best_w, color='r', linestyle='--', label=f'最优w: {best_w:.1f}')
plt.title('w and loss values', fontsize=12, pad=20)  # 增加标题间距，确保可见
plt.xlabel('w', fontsize=10)
plt.ylabel('loss (MSE)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()  # 自动调整布局

# 子图2：b与loss的关系（固定w为最优值）
plt.subplot(1, 2, 2)
plt.plot(b_values, loss_b_values, 'g-')
plt.axvline(x=best_b, color='r', linestyle='--', label=f'最优b: {best_b:.1f}')
plt.title('b and loss values', fontsize=12, pad=20)  # 增加标题间距，确保可见
plt.xlabel('b', fontsize=10)
plt.ylabel('loss (MSE)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()  # 自动调整布局

plt.suptitle('参数与损失值关系图', fontsize=14, y=1.02)  # 总标题
plt.show()

# 6. 绘制数据点和拟合直线
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='blue', label='数据点')
plt.plot(x_data, forward(x_data, best_w, best_b), 'r-', label=f'拟合直线: y={best_w:.2f}x + {best_b:.2f}')
plt.title('数据点与拟合直线', fontsize=12, pad=20)  # 增加标题间距，确保可见
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()  # 自动调整布局
plt.show()
