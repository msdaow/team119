import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可重现
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据预处理类
class BreastCancerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)  # 分类问题使用LongTensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型定义（保持原有复杂度）
class BreastCancerNet(nn.Module):
    def __init__(self, input_size):
        super(BreastCancerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),  # 输入层到第一隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(7, 6),  # 第一隐藏层到第二隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(6, 5),  # 第二隐藏层到第三隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(5, 2)  # 输出层改为2个神经元（二分类）
        )

    def forward(self, x):
        return self.network(x)


# 3. 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"数据集形状: {df.shape}")
    print(f"所有列名: {df.columns.tolist()}")

    # 显示前几行数据
    print(f"\n前5行数据:")
    print(df.head())

    # 检查数据基本信息
    print(f"\n数据基本信息:")
    print(df.info())

    # 检查目标变量
    if 'diagnosis' in df.columns:
        target_column = 'diagnosis'
        print(f"找到目标列: {target_column}")
        print(f"目标列唯一值: {df[target_column].unique()}")
    else:
        # 显示所有列的数据类型和前几个值
        print("\n各列信息:")
        for col in df.columns[:5]:  # 只显示前5列避免输出过多
            print(f"{col}: {df[col].dtype}, 前3个值: {df[col].head(3).tolist()}")

        # 尝试找到目标列 - 通常是第二列
        target_column = df.columns[1]
        print(f"\n使用默认目标列: {target_column}")
        print(f"目标列唯一值: {df[target_column].unique()}")

    print(f"\n目标变量 '{target_column}' 的分布:")
    print(df[target_column].value_counts())

    # 将目标变量转换为数值（M=恶性=1, B=良性=0）
    df_clean = df.copy()
    if df_clean[target_column].dtype == 'object':
        # 创建映射字典
        mapping_dict = {'M': 1, 'B': 0, 'malignant': 1, 'benign': 0}
        # 只映射存在的值
        unique_vals = df_clean[target_column].unique()
        actual_mapping = {}
        for val in unique_vals:
            if val in mapping_dict:
                actual_mapping[val] = mapping_dict[val]
            elif str(val).upper() in mapping_dict:
                actual_mapping[val] = mapping_dict[str(val).upper()]
            else:
                # 如果是数值字符串，直接转换
                try:
                    actual_mapping[val] = int(float(val))
                except:
                    actual_mapping[val] = val

        df_clean[target_column] = df_clean[target_column].map(actual_mapping)
        print(f"目标列转换后分布: {df_clean[target_column].value_counts()}")
    else:
        print(f"目标列已经是数值类型")

    # 选择特征列 - 排除ID列和目标列
    exclude_columns = ['id', 'diagnosis', 'Unnamed: 32']  # 添加常见的排除列
    # 移除不存在的排除列
    exclude_columns = [col for col in exclude_columns if col in df_clean.columns]
    exclude_columns.append(target_column)  # 确保目标列被排除

    feature_columns = []

    for col in df_clean.columns:
        if col not in exclude_columns and df_clean[col].dtype in ['int64', 'float64']:
            feature_columns.append(col)

    print(f"\n使用的特征列数量: {len(feature_columns)}")
    print(f"特征列: {feature_columns}")

    # 检查缺失值
    print(f"\n缺失值统计:")
    missing_info = df_clean[feature_columns + [target_column]].isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("没有缺失值")

    # 处理缺失值 - 使用填充而不是删除
    for col in feature_columns:
        if df_clean[col].isnull().any():
            fill_value = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(fill_value)
            print(f"用均值 {fill_value:.4f} 填充列 {col} 的缺失值")

    # 提取特征和目标
    X = df_clean[feature_columns].values
    y = df_clean[target_column].values

    print(f"\n最终数据形状 - X: {X.shape}, y: {y.shape}")
    print(f"处理后数据分布:")
    print(f"恶性(1): {np.sum(y == 1)} 样本")
    print(f"良性(0): {np.sum(y == 0)} 样本")

    # 检查数据是否为空
    if len(X) == 0:
        raise ValueError("错误: 特征矩阵为空!")

    if len(y) == 0:
        raise ValueError("错误: 目标变量为空!")

    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    print(f"标准化后数据形状: {X_scaled.shape}")

    return X_scaled, y, scaler_X, feature_columns, target_column


# 4. 训练函数（包含准确率计算）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_targets.size(0)
            train_correct += (predicted == batch_targets).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()

        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 保存最佳模型（基于验证准确率）
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_state


# 5. 可视化函数
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # 最终准确率对比
    plt.subplot(1, 3, 3)
    categories = ['Training', 'Validation']
    accuracies = [train_accuracies[-1], val_accuracies[-1]]
    colors = ['lightblue', 'lightcoral']

    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7)
    plt.title('Final Accuracy Comparison')
    plt.ylabel('Accuracy (%)')

    # 在柱状图上显示数值
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('breast_cancer_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 6. 绘制混淆矩阵
def plot_confusion_matrix(true_labels, predictions, save_path='confusion_matrix.png'):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['良性 (B)', '恶性 (M)'],
                yticklabels=['良性 (B)', '恶性 (M)'])
    plt.title('混淆矩阵 - 乳腺癌分类')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 7. 主函数
def main():
    # 参数设置
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.001

    print("开始处理威斯康星乳腺癌数据集...")
    print("=" * 50)

    # 加载和预处理数据
    try:
        X, y, scaler_X, feature_columns, target_column = load_and_preprocess_data('data.csv')
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    print(f"\n处理后数据形状 - 特征: {X.shape}, 目标: {y.shape}")
    print(f"输入特征维度: {X.shape[1]}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 创建数据集
    train_dataset = BreastCancerDataset(X_train, y_train)
    val_dataset = BreastCancerDataset(X_val, y_val)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n数据加载器配置:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  训练批次数量: {len(train_loader)}")
    print(f"  验证批次数量: {len(val_loader)}")

    # 初始化模型
    input_size = X.shape[1]
    model = BreastCancerNet(input_size)

    # 定义损失函数和优化器（分类问题使用CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f"\n模型结构:")
    print(model)
    print(f"\n开始训练...")
    print("=" * 50)

    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )

    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model_state,
        'scaler_X': scaler_X,
        'feature_columns': feature_columns,
        'input_size': input_size,
        'target_column': target_column
    }, 'best_breast_cancer_model.pt')

    print(f"\n最佳模型已保存为 'best_breast_cancer_model.pt'")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # 最终评估
    model.load_state_dict(best_model_state)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        # 训练集评估
        train_predictions = []
        train_targets = []
        for batch_features, batch_targets in train_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.numpy())
            train_targets.extend(batch_targets.numpy())

        # 验证集评估
        val_predictions = []
        val_targets = []
        for batch_features, batch_targets in val_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.numpy())
            val_targets.extend(batch_targets.numpy())
            all_predictions.extend(predicted.numpy())
            all_targets.extend(batch_targets.numpy())

    # 计算准确率
    train_accuracy = accuracy_score(train_targets, train_predictions)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    final_accuracy = accuracy_score(all_targets, all_predictions)

    print(f"\n" + "=" * 50)
    print(f"=== 模型性能评估 ===")
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"验证集准确率: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print(f"总体准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")

    # 分类报告
    print(f"\n=== 详细分类报告 ===")
    print(classification_report(all_targets, all_predictions,
                                target_names=['良性 (B)', '恶性 (M)']))

    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_predictions)

    # 保存评估结果
    results_df = pd.DataFrame({
        'epoch': range(1, NUM_EPOCHS + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    results_df.to_csv('training_results.csv', index=False)
    print(f"\n训练结果已保存为 'training_results.csv'")


# 8. 模型加载和预测函数
def load_and_predict(model_path, new_data=None):
    """加载保存的模型并进行预测"""
    checkpoint = torch.load(model_path)

    # 重建模型
    model = BreastCancerNet(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("模型加载成功!")
    print(f"输入特征数量: {checkpoint['input_size']}")

    if new_data is not None:
        # 对新数据进行预测
        new_data_scaled = checkpoint['scaler_X'].transform(new_data)
        new_data_tensor = torch.FloatTensor(new_data_scaled)

        with torch.no_grad():
            outputs = model(new_data_tensor)
            _, predictions = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)

        return predictions.numpy(), probabilities.numpy()

    return model


if __name__ == "__main__":
    main()