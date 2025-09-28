import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_model(x_data, y_data, w_space, b_space):
    """
    训练 y = wx + b 模型，计算不同 w 和 b 组合下的损失，找到最优 w 和 b。

    参数:
    x_data -- 自变量数据
    y_data -- 因变量数据
    w_space -- w 的取值范围
    b_space -- b 的取值范围

    返回:
    best_w -- 最优的 w 值
    best_b -- 最优的 b 值
    loss_matrix -- 不同 w 和 b 组合下的损失矩阵
    """
    # 初始化损失矩阵
    loss_matrix = np.zeros((len(w_space), len(b_space)))

    # 计算不同 w 和 b 组合下的损失
    for i, w in enumerate(w_space):
        for j, b in enumerate(b_space):
            y_pred = w * x_data + b
            loss = np.mean((y_pred - y_data) ** 2)
            loss_matrix[i, j] = loss

    # 找到最小损失对应的 w 和 b
    min_loss_index = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)
    best_w = w_space[min_loss_index[0]]
    best_b = b_space[min_loss_index[1]]

    return best_w, best_b, loss_matrix


def plot_loss_relationship(w_space, b_space, loss_matrix, best_w, best_b):
    """
    绘制 w 和 loss 之间的关系图线、b 和 loss 之间的关系图线，并标注最优解。

    参数:
    w_space -- w 的取值范围
    b_space -- b 的取值范围
    loss_matrix -- 不同 w 和 b 组合下的损失矩阵
    best_w -- 最优的 w 值
    best_b -- 最优的 b 值
    """
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 100

    # 设置 matplotlib 支持中文
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制 w 和 loss 的关系图（固定 b 为最优值）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    best_b_index = np.argmin(np.abs(b_space - best_b))
    plt.plot(w_space, loss_matrix[:, best_b_index], label='Loss')
    # 标注最优 w 值，将 marker 改为 'o'
    plt.scatter(best_w, loss_matrix[np.argmin(np.abs(w_space - best_w)), best_b_index],
                marker='o', color='red', s=150, label='Best w')
    plt.xlabel('w')
    plt.xticks(rotation=45)
    plt.ylabel('Loss')
    plt.title('Relationship between w and Loss')
    plt.legend()

    # 绘制 b 和 loss 的关系图（固定 w 为最优值）
    plt.subplot(1, 2, 2)
    best_w_index = np.argmin(np.abs(w_space - best_w))
    plt.plot(b_space, loss_matrix[best_w_index, :], label='Loss')
    # 标注最优 b 值，将 marker 改为 'o'
    plt.scatter(best_b, loss_matrix[best_w_index, np.argmin(np.abs(b_space - best_b))],
                marker='o', color='red', s=150, label='Best b')
    plt.xlabel('b')
    plt.xticks(rotation=45)
    plt.ylabel('Loss')
    plt.title('Relationship between b and Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 读取数据集
# 注意：如果在平台运行，需要修改为 '/mnt/train.csv'
df = pd.read_csv('F:/MobileFile/train.csv')

# 使用均值填充 y 列的缺失值
df['y'] = df['y'].fillna(df['y'].mean())

# 提取特征和目标变量
x_data = df['x'].values
y_data = df['y'].values

# 设置参数范围
w_space = np.arange(0.0, 2.0, 0.01)
b_space = np.arange(-10.0, 10.0, 0.1)

# 训练模型并获取最优 w 和 b 以及损失矩阵
best_w, best_b, loss_matrix = train_model(x_data, y_data, w_space, b_space)

print(f"Best w: {best_w}")
print(f"Best b: {best_b}")

# 绘制关系图线并标注最优解
plot_loss_relationship(w_space, b_space, loss_matrix, best_w, best_b)