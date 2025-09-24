import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据并处理缺失值
try:
    # 读取数据
    data = pd.read_csv('train.csv')

    # 显示原始数据缺失情况
    print("原始数据缺失值统计：")
    print(data.isnull().sum())

    # 处理缺失值：优先用均值填充，保留更多数据
    if data.isnull().values.any():
        print("\n正在处理缺失值...")
        # 对x列缺失值用均值填充
        if data['x'].isnull().any():
            x_mean = data['x'].mean()
            data['x'].fillna(x_mean, inplace=True)
            print(f"x列缺失值已用均值({x_mean:.2f})填充")

        # 对y列缺失值用均值填充
        if data['y'].isnull().any():
            y_mean = data['y'].mean()
            data['y'].fillna(y_mean, inplace=True)
            print(f"y列缺失值已用均值({y_mean:.2f})填充")

    # 提取处理后的训练数据
    x_data = data['x'].values
    y_data = data['y'].values
    print(f"\n处理后的数据量：{len(x_data)}条")

except FileNotFoundError:
    print("错误：未找到train.csv文件，请检查文件路径是否正确")
    exit()
except KeyError as e:
    print(f"错误：数据中缺少必要的列 {e}")
    exit()


# 前向传播函数
def forward(x, w, b):
    return x * w + b


# 损失计算函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 存储参数和对应的损失值
w_list = []
b_list = []
mse_list = []

# 定义参数搜索范围
w_range = np.arange(0.0, 4.1, 0.1)  # w从0.0到4.0，步长0.1
b_range = np.arange(-2.0, 2.1, 0.1)  # b从-2.0到2.0，步长0.1

# 遍历所有可能的参数组合
for w in w_range:
    for b in b_range:
        total_loss = 0
        for x_val, y_val in zip(x_data, y_data):
            total_loss += loss(x_val, y_val, w, b)
        # 计算平均损失
        mse = total_loss / len(x_data)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 找到最优参数（损失最小的参数组合）
min_index = np.argmin(mse_list)
best_w = w_list[min_index]
best_b = b_list[min_index]
min_mse = mse_list[min_index]

# 创建图形
plt.figure(figsize=(14, 6))

# 绘制w与loss的关系图（在最佳b值附近）
plt.subplot(1, 2, 1)
# 筛选b值接近最佳b的点
filtered_w = []
filtered_loss_w = []
for w, b, mse in zip(w_list, b_list, mse_list):
    if abs(b - best_b) < 0.1:  # 适当放宽条件确保有足够数据点
        filtered_w.append(w)
        filtered_loss_w.append(mse)

# 排序确保绘图顺序正确
sorted_pairs = sorted(zip(filtered_w, filtered_loss_w))
sorted_w, sorted_loss_w = zip(*sorted_pairs)

plt.plot(sorted_w, sorted_loss_w, 'b-', linewidth=2)
plt.scatter(best_w, min_mse, c='red', s=100, marker='*', label=f'optimal w: {best_w:.1f}')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Relationship between w and Loss (with optimal b)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 绘制b与loss的关系图（在最佳w值附近）
plt.subplot(1, 2, 2)
# 筛选w值接近最佳w的点
filtered_b = []
filtered_loss_b = []
for w, b, mse in zip(w_list, b_list, mse_list):
    if abs(w - best_w) < 0.1:  # 适当放宽条件确保有足够数据点
        filtered_b.append(b)
        filtered_loss_b.append(mse)

# 排序确保绘图顺序正确
sorted_pairs_b = sorted(zip(filtered_b, filtered_loss_b))
sorted_b, sorted_loss_b = zip(*sorted_pairs_b)

plt.plot(sorted_b, sorted_loss_b, 'g-', linewidth=2)
plt.scatter(best_b, min_mse, c='red', s=100, marker='*', label=f'optimal b: {best_b:.1f}')
plt.xlabel('b')
plt.ylabel('Loss')
plt.title('Relationship between b and Loss (with optimal w)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# 输出最优参数
print(f"\n最优参数: w={best_w:.2f}, b={best_b:.2f}")
print(f"最小损失值: {min_mse:.4f}")
