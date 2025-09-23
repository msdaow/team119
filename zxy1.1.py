# 导入所需要的工具包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('train.csv')

# 数据清洗：去除空值和异常值
data = data.dropna()  # 删除空值
data = data[(data['x'] > 0) & (data['y'] > 0)]

# 准备数据集
x_data = data['x'].values
y_data = data['y'].values

print("after clean:")
for i in range(len(x_data)):
    print(f"x={x_data[i]:.1f}, y={y_data[i]:.1f}")


# 计算正向传播结果 y_pred = x * w + b
def forward(x, w, b):
    return x * w + b


# 计算损失值 loss = (y_pred - y)^2
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) * (y_pred - y)


# 将权重系数、偏置项和损失值分别存放为画图做准备
w_list = []
b_list = []
mse_list = []

# 计算w和b取不同值时的损失值
# 固定b=1，只变化w，以便于可视化

b_fixed = 1.0  # 固定偏置项

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w, 'b=', b_fixed)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w, b_fixed)
        loss_val = loss(x_val, y_val, w, b_fixed)
        l_sum += loss_val
        print('\t', f'x={x_val:.1f}, y={y_val:.1f}, y_pred={y_pred_val:.2f}, loss={loss_val:.2f}')

    mse = l_sum / len(x_data)
    print('MSE=', mse)
    w_list.append(w)
    b_list.append(b_fixed)
    mse_list.append(mse)

# 绘制损失函数随w变化的图像
plt.figure(figsize=(10, 6))
plt.plot(w_list, mse_list)
plt.ylabel('Loss (MSE)')
plt.xlabel('Weight (w)')
plt.title(f'Loss vs Weight (with fixed b={b_fixed})')
plt.grid(True)
plt.show()

# 找到最小损失对应的w值
min_loss_index = np.argmin(mse_list)
best_w = w_list[min_loss_index]
best_loss = mse_list[min_loss_index]

print(f"\nbest w: w = {best_w:.2f}, b = {b_fixed}")
print(f"mini loss: {best_loss:.4f}")

# 绘制原始数据和最佳拟合线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='begain data')
x_range = np.linspace(min(x_data), max(x_data), 100)
y_pred_range = forward(x_range, best_w, b_fixed)
plt.plot(x_range, y_pred_range, color='red', label=f'fitted line: y = {best_w:.2f}x + {b_fixed}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('results of linear regression fitting')
plt.grid(True)
plt.show()