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
import pickle
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据加载和预处理
class EarthquakeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 修复文件路径问题 - 使用原始斜杠或原始字符串
df = pd.read_csv(r'F:\earthquake_alert_balanced_dataset.csv')  # 使用原始字符串

print("数据集基本信息:")
print(f"数据集形状: {df.shape}")
print(f"特征列: {df.columns.tolist()}")
print("\n前5行数据:")
print(df.head())

print("\n数据信息:")
print(df.info())

print("\n目标变量分布:")
print(df['alert'].value_counts())

# 数据质量检查
print("\n数据质量检查:")
print("缺失值统计:")
print(df.isnull().sum())

print("\n数据描述:")
print(df.describe())


# 处理异常值 - 使用IQR方法
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"{col} 列的异常值数量: {len(outliers)}")

        # 移除异常值
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    print(f"原始数据量: {len(df)}")
    print(f"清理后数据量: {len(df_clean)}")
    print(f"移除的样本数: {len(df) - len(df_clean)}")
    return df_clean


# 选择数值列进行异常值检测
numeric_columns = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
df_clean = remove_outliers(df, numeric_columns)

# 编码目标变量
label_encoder = LabelEncoder()
df_clean['alert_encoded'] = label_encoder.fit_transform(df_clean['alert'])

print("\n目标变量编码映射:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")

# 准备特征和目标
features = df_clean[numeric_columns].values
labels = df_clean['alert_encoded'].values

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print(f"\n标准化后的特征形状: {features_scaled.shape}")
print(f"标签形状: {labels.shape}")

# 2. 创建数据集和数据加载器
dataset = EarthquakeDataset(features_scaled, labels)

# 划分训练集和测试集 (80% 训练, 20% 测试)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"\n训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 创建数据加载器
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"\n数据加载器参数:")
print(f"批次大小: {batch_size}")
print(f"训练批次数量: {len(train_loader)}")
print(f"测试批次数量: {len(test_loader)}")


# 3. 定义神经网络模型
class EarthquakeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, hidden_size3=32, num_classes=4):
        super(EarthquakeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Dropout(0.3),

            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Dropout(0.3),

            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size3),
            nn.Dropout(0.2),

            nn.Linear(hidden_size3, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# 初始化模型
input_size = features_scaled.shape[1]
num_classes = len(label_encoder.classes_)
model = EarthquakeClassifier(input_size=input_size, num_classes=num_classes)

print(f"\n模型结构:")
print(model)
print(f"输入特征数: {input_size}")
print(f"输出类别数: {num_classes}")

# 4. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\n训练配置:")
print(f"设备: {device}")
print(f"学习率: 0.001")
print(f"epochs: {num_epochs}")
print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam")

# 用于记录训练过程
train_losses = []
train_accuracies = []
test_accuracies = []
best_accuracy = 0.0
best_model_state = None

# 5. 训练循环
print("\n开始训练...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = 100 * test_correct / test_total

    # 记录指标
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 保存最佳模型 - 只保存模型权重，不包含sklearn对象
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_state = model.state_dict().copy()
        # 只保存模型权重
        torch.save(best_model_state, 'best_earthquake_model_weights.pt')

    # 更新学习率
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

print(f"\n训练完成! 最佳测试准确率: {best_accuracy:.2f}%")

# 保存预处理对象到单独的文件
preprocessing_data = {
    'label_encoder_classes': label_encoder.classes_,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'input_size': input_size,
    'num_classes': num_classes
}

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing_data, f)

print("预处理对象已保存到 preprocessing.pkl")

# 6. 可视化训练过程
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# 最终测试准确率
plt.subplot(1, 3, 3)
final_train_acc = train_accuracies[-1]
final_test_acc = test_accuracies[-1]
categories = ['Training', 'Test']
accuracies = [final_train_acc, final_test_acc]
bars = plt.bar(categories, accuracies, color=['blue', 'orange'])
plt.title('Final Accuracy')
plt.ylabel('Accuracy (%)')

# 在柱状图上显示数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 加载最佳模型进行最终测试
print("\n加载最佳模型进行最终测试...")

# 加载预处理数据
with open('preprocessing.pkl', 'rb') as f:
    preprocessing_data = pickle.load(f)

# 重建预处理对象
label_encoder_rebuilt = LabelEncoder()
label_encoder_rebuilt.classes_ = preprocessing_data['label_encoder_classes']

scaler_rebuilt = StandardScaler()
scaler_rebuilt.mean_ = preprocessing_data['scaler_mean']
scaler_rebuilt.scale_ = preprocessing_data['scaler_scale']

# 加载模型权重
model_weights = torch.load('best_earthquake_model_weights.pt')
model = EarthquakeClassifier(input_size=preprocessing_data['input_size'],
                             num_classes=preprocessing_data['num_classes'])
model.load_state_dict(model_weights)
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算最终指标
final_accuracy = accuracy_score(all_targets, all_predictions) * 100
print(f"最终测试准确率: {final_accuracy:.2f}%")

# 分类报告
print("\n详细分类报告:")
print(classification_report(all_targets, all_predictions,
                            target_names=label_encoder_rebuilt.classes_))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_targets, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder_rebuilt.classes_,
            yticklabels=label_encoder_rebuilt.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n模型和预处理工具已保存!")
print("保存的文件:")
print("- best_earthquake_model_weights.pt (最佳模型权重)")
print("- preprocessing.pkl (预处理对象)")
print("- training_metrics.png (训练指标图)")
print("- confusion_matrix.png (混淆矩阵)")


# 8. 模型推理示例
def predict_earthquake_alert(magnitude, depth, cdi, mmi, sig):
    """
    使用训练好的模型预测地震预警级别
    """
    # 加载预处理数据
    with open('preprocessing.pkl', 'rb') as f:
        preprocessing_data = pickle.load(f)

    # 重建预处理对象
    label_encoder_rebuilt = LabelEncoder()
    label_encoder_rebuilt.classes_ = preprocessing_data['label_encoder_classes']

    scaler_rebuilt = StandardScaler()
    scaler_rebuilt.mean_ = preprocessing_data['scaler_mean']
    scaler_rebuilt.scale_ = preprocessing_data['scaler_scale']

    # 加载模型
    model = EarthquakeClassifier(input_size=preprocessing_data['input_size'],
                                 num_classes=preprocessing_data['num_classes'])
    model.load_state_dict(torch.load('best_earthquake_model_weights.pt'))
    model.eval()

    # 准备输入数据
    input_data = np.array([[magnitude, depth, cdi, mmi, sig]])
    input_scaled = scaler_rebuilt.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        alert_level = label_encoder_rebuilt.inverse_transform(predicted.numpy())[0]
        probabilities = torch.softmax(output, dim=1).numpy()[0]

    print(f"\n地震参数预测:")
    print(f"震级: {magnitude}, 深度: {depth}, CDI: {cdi}, MMI: {mmi}, SIG: {sig}")
    print(f"预测预警级别: {alert_level}")
    print("各类别概率:")
    for i, class_name in enumerate(label_encoder_rebuilt.classes_):
        print(f"  {class_name}: {probabilities[i]:.4f}")

    return alert_level


# 测试预测函数
print("\n预测示例:")
predict_earthquake_alert(7.5, 50, 8, 7, 100)
predict_earthquake_alert(6.0, 10, 5, 5, 50)