import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import pickle
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib字体，解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# -------------------------- 1. 基础配置（保证可复现） --------------------------
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True


# -------------------------- 2. 自定义Dataset类 --------------------------
class HealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# -------------------------- 3. 数据加载与预处理 --------------------------
# 加载数据集
file_path = r'F:\health_lifestyle_dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"请确保数据集文件'{file_path}'存在")
df = pd.read_csv(file_path)

print("=" * 50)
print("1. 数据集基础信息与预处理")
print("=" * 50)
print(f"原始数据集形状: {df.shape}")
print("\n【缺失值统计】")
missing = df.isnull().sum()
print(missing[missing > 0])


def handle_missing_values(df):
    df_clean = df.copy()
    # 数值型特征：用中位数填充
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            print(f"数值列'{col}'：用中位数填充缺失值")

    # 类别型特征：用众数填充
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            print(f"类别列'{col}'：用众数填充缺失值")
    return df_clean


df = handle_missing_values(df)
print(f"\n缺失值处理后形状: {df.shape}")


# 3.2 异常值处理（IQR方法）
def remove_outliers(df):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns

    # 识别目标列（假设最后一列是目标变量）
    target_col = df_clean.columns[-1]

    for col in numeric_cols:
        # 排除目标变量
        if col == target_col:
            continue

        # IQR计算
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 统计并移除异常值
        outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            df_clean = df_clean[~outlier_mask]
            print(f"数值列'{col}'：移除{outlier_count}个异常值")

    print(f"\n异常值处理后形状: {df_clean.shape}")
    return df_clean


df = remove_outliers(df)

# 3.3 特征与标签分离
target_col = df.columns[-1]
features = df.drop(columns=[target_col]).copy()
labels = df[target_col].copy()

# 查看类别分布
print("\n【类别分布统计】")
label_counts = Counter(labels)
print(label_counts)

# 3.4 特征编码与标准化
# 对类别型特征编码
categorical_cols = features.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le
    print(f"类别列'{col}'编码完成")

# 数值特征标准化
numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
print("数值特征标准化完成")

# 标签编码
label_le = LabelEncoder()
if labels.dtype == 'object':
    labels_encoded = label_le.fit_transform(labels)
    print(f"\n标签编码映射: {dict(zip(label_le.classes_, label_le.transform(label_le.classes_)))}")
else:
    labels_encoded = labels.values
    label_le.classes_ = np.unique(labels_encoded)

# 转换为numpy数组
X = features.values
y = labels_encoded

print(f"\n预处理后特征形状: {X.shape}，标签形状: {y.shape}")

# 3.5 处理数据不平衡（可选，根据实际情况决定是否使用SMOTE）
try:
    from imblearn.over_sampling import SMOTE

    use_smote = True

    print("\n【处理数据不平衡】")
    print(f"处理前类别分布: {Counter(y)}")

    # 检查是否需要SMOTE（如果类别分布不平衡）
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    # 如果最大类别数量是最小类别的2倍以上，使用SMOTE
    if max_count / min_count > 2:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"处理后类别分布: {Counter(y_resampled)}")
        X_final, y_final = X_resampled, y_resampled
    else:
        print("类别分布相对平衡，跳过SMOTE处理")
        X_final, y_final = X, y

except ImportError:
    print("未安装imblearn，跳过SMOTE处理")
    X_final, y_final = X, y
    use_smote = False

print(f"\n最终特征形状: {X_final.shape}，标签形状: {y_final.shape}")

# -------------------------- 4. 数据集划分 --------------------------
dataset = HealthDataset(X_final, y_final)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("\n" + "=" * 50)
print("2. 数据集划分结果")
print("=" * 50)
print(f"训练集样本数: {len(train_dataset)}，测试集样本数: {len(test_dataset)}")

# -------------------------- 5. DataLoader配置 --------------------------
batch_size = min(64, len(train_dataset))  # 动态调整batch_size

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True  # 丢弃最后一个不完整的batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("\n" + "=" * 50)
print("3. DataLoader参数配置")
print("=" * 50)
print(f"批次大小: {batch_size}")
print(f"训练集批次: {len(train_loader)}，测试集批次: {len(test_loader)}")


# -------------------------- 6. 搭建神经网络 --------------------------
class HealthClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HealthClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# 获取输入维度和类别数
input_dim = X_final.shape[1]
num_classes = len(np.unique(y_final))
model = HealthClassifier(input_dim, num_classes)

print("\n" + "=" * 50)
print("4. 神经网络结构")
print("=" * 50)
print(f"输入维度: {input_dim}")
print(f"输出类别数: {num_classes}")
print(model)

# -------------------------- 7. 训练配置与训练过程 --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 计算类别权重
class_counts = Counter(y_final)
total = len(y_final)
class_weights = torch.FloatTensor([total / class_counts[i] for i in range(num_classes)]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
num_epochs = 150

# 记录训练指标
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0.0
best_model_path = 'best_health_model.pt'

print("\n" + "=" * 50)
print("5. 训练配置")
print("=" * 50)
print(f"训练设备: {device}")
print(f"类别权重: {class_weights.cpu().numpy()}")
print(f"优化器: Adam (lr=0.001)")
print(f"训练轮次: {num_epochs}")

# 保存预处理信息
preprocessing_info = {
    'label_encoders': label_encoders,
    'scaler': scaler,
    'label_le': label_le,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'feature_columns': features.columns.tolist(),
    'target_col': target_col
}

with open('preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)

# 训练循环
print("\n" + "=" * 50)
print("6. 开始训练")
print("=" * 50)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += target.size(0)
        train_correct += (preds == target).sum().item()

    # 计算训练集指标
    avg_train_loss = train_loss / train_total
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # 测试阶段
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)
            test_total += target.size(0)
            test_correct += (preds == target).sum().item()

    avg_test_loss = test_loss / test_total
    test_acc = 100 * test_correct / test_total
    test_losses.append(avg_test_loss)
    test_accs.append(test_acc)

    # 学习率调整
    scheduler.step(avg_test_loss)

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_acc,
            'epoch': epoch
        }, best_model_path)

    # 每10轮打印日志
    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
              f"训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:6.2f}% | "
              f"测试损失: {avg_test_loss:.4f} | 测试准确率: {test_acc:6.2f}% | "
              f"学习率: {current_lr:.6f}")

print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")

# -------------------------- 8. 训练过程可视化 --------------------------
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2, label='训练损失')
plt.plot(test_losses, 'r-', linewidth=2, label='测试损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('训练与测试损失变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# 准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accs, 'b-', linewidth=2, label='训练准确率')
plt.plot(test_accs, 'r-', linewidth=2, label='测试准确率')
plt.axhline(y=best_acc, color='g', linestyle='--', label=f'最佳准确率: {best_acc:.2f}%')
plt.xlabel('训练轮次')
plt.ylabel('准确率(%)')
plt.title('训练与测试准确率变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# 类别分布图
plt.subplot(1, 3, 3)
if use_smote:
    original_counts = list(Counter(y).values())
    resampled_counts = list(Counter(y_final).values())
    x = np.arange(len(original_counts))
    width = 0.35

    plt.bar(x - width / 2, original_counts, width, label='原始分布', alpha=0.7)
    plt.bar(x + width / 2, resampled_counts, width, label='SMOTE后', alpha=0.7)
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('SMOTE前后类别分布对比')
    plt.legend()
else:
    counts = list(Counter(y_final).values())
    plt.bar(range(len(counts)), counts, alpha=0.7)
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('类别分布')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\n训练可视化图已保存为: training_metrics.png")

# -------------------------- 9. 加载最优模型并测试 --------------------------
print("\n" + "=" * 50)
print("7. 加载最优模型并测试")
print("=" * 50)

# 加载模型
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 在测试集上评估
all_preds = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算准确率
test_accuracy = accuracy_score(all_targets, all_preds) * 100
print(f"最优模型在测试集上的准确率: {test_accuracy:.2f}%")

# 详细分类报告
print("\n测试集分类报告:")
print(classification_report(all_targets, all_preds, target_names=[str(cls) for cls in label_le.classes_]))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_le.classes_,
            yticklabels=label_le.classes_)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n【预测类别分布】")
print(Counter(all_preds))


# -------------------------- 10. 模型预测函数 --------------------------
def predict_health_status(new_data):
    """
    使用训练好的模型预测健康状态
    """
    # 加载预处理信息
    with open('preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)

    # 加载模型
    model = HealthClassifier(preprocessing_info['input_dim'], preprocessing_info['num_classes'])
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 预处理新数据
    new_data_processed = new_data.copy()

    # 对类别特征编码
    for col, encoder in preprocessing_info['label_encoders'].items():
        if col in new_data_processed.columns:
            # 处理未知类别
            unknown_mask = ~new_data_processed[col].isin(encoder.classes_)
            if unknown_mask.any():
                print(f"警告: 列 '{col}' 中有未知类别，用众数替换")
                new_data_processed.loc[unknown_mask, col] = encoder.classes_[0]
            new_data_processed[col] = encoder.transform(new_data_processed[col])

    # 数值特征标准化
    numeric_cols = new_data_processed.select_dtypes(include=['float64', 'int64']).columns
    new_data_processed[numeric_cols] = preprocessing_info['scaler'].transform(new_data_processed[numeric_cols])

    # 转换为tensor并预测
    with torch.no_grad():
        input_tensor = torch.FloatTensor(new_data_processed.values)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

        # 转换为原始标签
        predicted_labels = preprocessing_info['label_le'].inverse_transform(predictions.numpy())

    return predicted_labels, probabilities.numpy()


# -------------------------- 11. 保存完整模型信息 --------------------------
print("\n" + "=" * 50)
print("8. 模型保存信息")
print("=" * 50)
print("已保存的文件:")
print(f"- {best_model_path} (最佳模型)")
print(f"- preprocessing_info.pkl (预处理信息)")
print(f"- training_metrics.png (训练指标图)")
print(f"- confusion_matrix.png (混淆矩阵)")

print(f"\n模型训练完成！最佳测试准确率: {best_acc:.2f}%")