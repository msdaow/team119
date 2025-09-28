# 解决OpenMP运行时冲突
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# -------------------------- 1. 图表样式配置 --------------------------
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.markersize"] = 4
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False


# -------------------------- 2. 数据集加载（增强容错） --------------------------
def load_train_data(file_path="D:/train.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在：{file_path}，请检查路径！")

    df = pd.read_csv(file_path)
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("数据集必须包含'x'（特征）和'y'（标签）列！")

    # 1. 强制转换为float，避免非数值类型
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # 2. 过滤NaN/Inf和极端值（保留95%置信区间内的数据，更严格容错）
    df = df[(np.isfinite(df['x'])) & (np.isfinite(df['y']))]
    x_mean, x_std = df['x'].mean(), df['x'].std()
    y_mean, y_std = df['y'].mean(), df['y'].std()
    df = df[(df['x'] >= x_mean - 3 * x_std) & (df['x'] <= x_mean + 3 * x_std) &
            (df['y'] >= y_mean - 3 * y_std) & (df['y'] <= y_mean + 3 * y_std)]

    # 3. 确保数据量足够（至少5个样本，避免训练无意义）
    if len(df) < 5:
        raise ValueError(f"有效样本数过少（仅{len(df)}个），请检查数据集质量！")

    x_data = df['x'].values.tolist()
    y_data = df['y'].values.tolist()

    print(f"数据集加载成功！有效样本数：{len(x_data)}")
    print(f"x范围：{min(x_data):.2f} ~ {max(x_data):.2f}")
    print(f"y范围：{min(y_data):.2f} ~ {max(y_data):.2f}")
    return x_data, y_data


# 加载数据（捕获所有可能异常）
try:
    x_data, y_data = load_train_data(file_path="D:/train.csv")  # 若路径不同，修改此处！
except Exception as e:
    print(f"数据加载失败：{str(e)}")
    exit()

# -------------------------- 3. 模型初始化与训练（核心容错优化） --------------------------
# 初始化权重：用数据粗略估计初始值（避免初始偏差过大导致NaN）
x_mean = np.mean(x_data)
y_mean = np.mean(y_data)
init_w = y_mean / x_mean if x_mean != 0 else 0.1  # 基于数据均值估计初始w
w = torch.tensor([init_w], dtype=torch.float32, requires_grad=True)

learning_rate = 0.001  # 降低学习率，避免权重震荡
epochs = 150
# 训练记录（新增有效性标记）
epoch_history = []
w_history = []
loss_history = []


# -------------------------- 4. 模型与损失函数 --------------------------
def forward(x):
    x_tensor = torch.tensor([x], dtype=torch.float32)
    return x_tensor * w


def compute_loss(x, y):
    y_pred = forward(x)
    y_tensor = torch.tensor([y], dtype=torch.float32)
    return (y_pred - y_tensor) ** 2


# -------------------------- 5. 训练过程（新增无效值检测与处理） --------------------------
init_pred = forward(4.0).item()
print(f"\n训练前预测（x=4）：{init_pred:.4f}")
print("\n开始训练...")

for epoch in range(epochs):
    total_loss = 0.0
    valid_sample = 0  # 记录有效样本数（避免空样本累加）

    for x, y in zip(x_data, y_data):
        # 跳过极端值样本（双重保险）
        if abs(x) > 1e6 or abs(y) > 1e6:
            continue

        try:
            # 前向传播与反向传播
            loss = compute_loss(x, y)
            loss.backward()

            # 梯度裁剪：强制限制梯度范围（核心！防止梯度爆炸导致w为NaN）
            torch.nn.utils.clip_grad_value_(w, clip_value=0.5)

            # 更新权重（用no_grad()确保不追踪梯度）
            with torch.no_grad():
                w.data -= learning_rate * w.grad.data

                # 检测权重是否为无效值，若无效则重置
                if torch.isnan(w.data).any() or torch.isinf(w.data).any():
                    print(f"警告：轮次{epoch + 1}权重无效，重置为初始值！")
                    w.data = torch.tensor([init_w], dtype=torch.float32)

            # 清空梯度
            w.grad.zero_()

            # 累加有效损失
            total_loss += loss.item()
            valid_sample += 1

        except Exception as e:
            print(f"轮次{epoch + 1}样本({x},{y})训练失败：{str(e)}，跳过该样本！")
            continue

    # 计算平均损失（确保有效样本数>0）
    if valid_sample == 0:
        print(f"轮次{epoch + 1}无有效样本，跳过记录！")
        continue
    avg_loss = total_loss / valid_sample

    # 提取当前w的数值（确保为有效值才记录）
    current_w = w.item()
    if not (np.isnan(current_w) or np.isinf(current_w)) and not (np.isnan(avg_loss) or np.isinf(avg_loss)):
        epoch_history.append(epoch)
        w_history.append(current_w)
        loss_history.append(avg_loss)
    else:
        print(f"轮次{epoch + 1}数据无效（w={current_w:.2f}, loss={avg_loss:.2f}），跳过记录！")

    # 打印进度
    if (epoch + 1) % 15 == 0:
        current_pred = forward(4.0).item() if not np.isnan(current_w) else 0.0
        print(f"轮次：{epoch + 1:3d} | 平均损失：{avg_loss:.6f} | 权重w：{current_w:.6f} | 预测（x=4）：{current_pred:.4f}")

print("训练完成！")


# -------------------------- 6. 可视化（新增轴范围容错计算） --------------------------
# 第一步：过滤训练记录中的无效值（双重保险）
def filter_invalid(data):
    return [d for d in data if not (np.isnan(d) or np.isinf(d))]


w_history = filter_invalid(w_history)
loss_history = filter_invalid(loss_history)
epoch_history = epoch_history[:len(w_history)]  # 保持长度一致

# 检查是否有有效训练数据
if len(w_history) < 2 or len(loss_history) < 2:
    print("有效训练数据不足，无法生成图表！")
    exit()

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Linear Model Training Process (y = w*x)', fontsize=14, fontweight='bold')

# 子图1：w随迭代次数变化（轴范围容错）
ax1 = axes[0, 0]
ax1.plot(epoch_history, w_history, color='#2E86AB', marker='o',
         markerfacecolor='white', markeredgecolor='#2E86AB', label='Weight w')
final_w = w_history[-1]
ax1.axhline(y=final_w, color='#A23B72', linestyle='--', linewidth=2,
            label=f'Converged w: {final_w:.4f}')

# 核心：轴范围容错计算（避免w_min == w_max或值为0的情况）
w_min, w_max = min(w_history), max(w_history)
if w_max - w_min < 1e-6:  # 若w几乎无变化（收敛到固定值）
    w_margin = 0.1  # 固定余量
else:
    w_margin = (w_max - w_min) * 0.1  # 10%余量
ax1.set_ylim(w_min - w_margin, w_max + w_margin)

ax1.set_xlabel('Epoch (Iteration)')
ax1.set_ylabel('Value of Weight w')
ax1.set_title('Weight w Changes During Training')
ax1.legend()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# 子图2：loss随迭代次数变化（对数刻度+容错）
ax2 = axes[0, 1]
# 过滤loss中的极小值（避免对数刻度报错）
loss_valid = [l if l > 1e-10 else 1e-10 for l in loss_history]
ax2.plot(epoch_history, loss_valid, color='#F18F01', marker='s',
         markerfacecolor='white', markeredgecolor='#F18F01', label='Average Loss')
ax2.set_yscale('log')
ax2.set_xlabel('Epoch (Iteration)')
ax2.set_ylabel('Average Loss (Log Scale)')
ax2.set_title('Average Loss Changes During Training')
ax2.legend()
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# 子图3：w与loss相关性（容错拟合）
ax3 = axes[1, 0]
scatter = ax3.scatter(w_history, loss_valid, c=epoch_history, cmap='viridis',
                      alpha=0.8, s=50, edgecolors='white')
# 趋势线拟合容错（至少5个点才拟合）
if len(w_history) >= 5:
    try:
        z = np.polyfit(w_history, loss_valid, 3)
        p = np.poly1d(z)
        w_smooth = np.linspace(w_min, w_max, 200)
        ax3.plot(w_smooth, p(w_smooth), color='#E63946', linestyle='--',
                 linewidth=2.5, label='Trend Line (3rd-order Fit)')
    except Exception as e:
        print(f"趋势线拟合失败：{str(e)}，跳过趋势线！")
ax3.set_xlabel('Value of Weight w')
ax3.set_ylabel('Average Loss')
ax3.set_title('Correlation Between Weight w and Average Loss')
ax3.legend()
cbar = plt.colorbar(scatter, ax=ax3, shrink=0.85)
cbar.set_label('Epoch (Iteration)', rotation=270, labelpad=15)

# 子图4：拟合曲线与实际数据（数据排序+范围容错）
ax4 = axes[1, 1]
ax4.scatter(x_data, y_data, color='#06D6A0', alpha=0.6, s=60,
            edgecolors='white', label='Actual Data')
# 按x排序，确保拟合线平滑
x_sorted = sorted(x_data)
y_pred_sorted = [forward(x).item() for x in x_sorted]
ax4.plot(x_sorted, y_pred_sorted, color='#EF476F', linewidth=3,
         label=f'Fitted Line: y = {final_w:.4f}x')
# x/y轴范围容错
x_min, x_max = min(x_data), max(x_data)
y_min, y_max = min(y_data), max(y_data)
x_margin = 0.1 if x_max - x_min < 1e-6 else (x_max - x_min) * 0.1
y_margin = 0.1 if y_max - y_min < 1e-6 else (y_max - y_min) * 0.1
ax4.set_xlim(x_min - x_margin, x_max + x_margin)
ax4.set_ylim(y_min - y_margin, y_max + y_margin)
ax4.set_xlabel('x (Feature)')
ax4.set_ylabel('y (Label/Prediction)')
ax4.set_title('Fitted Line vs Actual Data')
ax4.legend()

# 调整布局并保存
plt.tight_layout()
try:
    plt.savefig('linear_model_training_history.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n图表已保存为：linear_model_training_history.png")
except Exception as e:
    print(f"图表保存失败：{str(e)}")
plt.show()

# -------------------------- 7. 输出最终结果 --------------------------
final_pred = forward(4.0).item()
print("\n" + "=" * 60)
print("训练结果汇总")
print("=" * 60)
print(f"最终权重 w：{final_w:.6f}")
print(f"最终平均损失：{loss_history[-1]:.6f}")
print(f"训练后预测（x=4）：{final_pred:.4f}")
print("=" * 60)