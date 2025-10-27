import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 1. 自定义数据集类 - 修复版
class CustomDataset(Dataset):
    def __init__(self, data_path):
        # 使用pandas读取数据，支持混合数据类型
        data = pd.read_csv(data_path)

        # 分离特征和标签（假设最后一列是标签）
        self.raw_labels = data.iloc[:, -1].values
        features = data.iloc[:, :-1]

        # 处理分类变量
        self.features, self.feature_encoders = self._preprocess_features(features)

        # 处理标签 - 统一编码
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.raw_labels)
        self.num_classes = len(self.label_encoder.classes_)

        print(f"标签类别: {self.label_encoder.classes_}")
        print(f"标签编码映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 数据标准化
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        print(f"数据集形状: 特征 {self.features.shape}, 标签 {self.labels.shape}")
        print(f"类别数量: {self.num_classes}")

    def _preprocess_features(self, features):
        """预处理特征，将分类变量转换为数值"""
        processed_features = []
        feature_encoders = {}

        for column in features.columns:
            if features[column].dtype == 'object':  # 分类变量
                le = LabelEncoder()
                encoded = le.fit_transform(features[column].values)
                processed_features.append(encoded)
                feature_encoders[column] = le
                print(f"编码分类特征 '{column}': {len(le.classes_)} 个类别")
            else:  # 数值变量
                # 处理缺失值
                if features[column].isnull().any():
                    features[column] = features[column].fillna(features[column].mean())
                processed_features.append(features[column].values)

        # 合并所有特征
        return np.column_stack(processed_features), feature_encoders

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        labels = torch.LongTensor([self.labels[idx]])  # 使用编码后的标签
        return features, labels.squeeze()


# 2. 神经网络模型定义 - 修复版
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=[7, 6, 5], output_size=2):
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        # 只有在分类问题且输出维度>1时才添加Softmax
        # 注意：CrossEntropyLoss已经包含Softmax，所以这里不需要
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 3. 训练函数 - 修复版
def train_model(model, train_loader, val_loader, num_epochs=100, problem_type='classification', num_classes=2):
    # 根据问题类型选择损失函数
    if problem_type == 'classification':
        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()  # 二分类
        else:
            criterion = nn.CrossEntropyLoss()  # 多分类
    else:  # regression
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)

            # 根据问题类型处理损失计算
            if problem_type == 'classification':
                if num_classes == 2:
                    # 二分类，需要将标签转换为float并调整形状
                    loss = criterion(outputs.squeeze(), batch_labels.float())
                else:
                    # 多分类
                    loss = criterion(outputs, batch_labels)
            else:
                # 回归
                loss = criterion(outputs.squeeze(), batch_labels.float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)

                # 根据问题类型处理损失计算
                if problem_type == 'classification':
                    if num_classes == 2:
                        loss = criterion(outputs.squeeze(), batch_labels.float())
                        # 计算准确率
                        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                        correct += (predicted == batch_labels).sum().item()
                    else:
                        loss = criterion(outputs, batch_labels)
                        # 计算准确率
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == batch_labels).sum().item()
                    total += batch_labels.size(0)
                else:
                    loss = criterion(outputs.squeeze(), batch_labels.float())

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 计算准确率
        accuracy = correct / total if problem_type == 'classification' else 0

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_layers': [7, 6, 5],
                'output_size': output_size,
                'problem_type': problem_type,
                'num_classes': num_classes,
                'label_encoder': dataset.label_encoder,
                'scaler': dataset.scaler
            }, 'best_model.pt')
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], 保存最佳模型，验证损失: {avg_val_loss:.4f}, 准确率: {accuracy:.4f}')
        elif (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}, LR: {current_lr:.6f}')

    return train_losses, val_losses


# 4. 可视化函数
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss (log)')
    plt.semilogy(val_losses, label='Validation Loss (log)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. 数据探索函数
def explore_data(data_path):
    """探索数据集的基本信息"""
    data = pd.read_csv(data_path)
    print("=" * 50)
    print("数据集探索")
    print("=" * 50)
    print(f"数据集形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print("\n前5行数据:")
    print(data.head())
    print("\n数据类型:")
    print(data.dtypes)
    print("\n缺失值统计:")
    print(data.isnull().sum())
    print("\n标签分布:")
    label_counts = data.iloc[:, -1].value_counts()
    print(label_counts)
    print(f"唯一标签数量: {len(label_counts)}")
    print("=" * 50)


# 主程序
if __name__ == "__main__":
    # 数据集路径 - 请修改为您的实际路径
    data_path = "D:/countries.csv"  # 替换为您的数据集路径

    # 探索数据
    explore_data(data_path)

    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    dataset = CustomDataset(data_path)

    # 数据集分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 数据加载器配置
    batch_size = 32
    num_workers = 0  # 在Windows上建议设为0避免问题

    print(f"Batch size: {batch_size}, Num workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    # 自动检测问题类型
    num_classes = dataset.num_classes

    if num_classes == 2:
        problem_type = 'classification'
        output_size = 1  # 二分类输出1个值
        print(f"检测为二分类问题")
    elif num_classes > 2:
        problem_type = 'classification'
        output_size = num_classes  # 多分类输出类别数
        print(f"检测为多分类问题, 类别数: {num_classes}")
    else:
        problem_type = 'regression'
        output_size = 1
        print(f"检测为回归问题")

    # 创建模型
    input_size = dataset[0][0].shape[0]
    model = NeuralNetwork(input_size=input_size,
                          hidden_layers=[7, 6, 5],
                          output_size=output_size).to(device)

    print(f"输入维度: {input_size}")
    print(f"输出维度: {output_size}")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           num_epochs=100,
                                           problem_type=problem_type,
                                           num_classes=num_classes)

    # 可视化训练过程
    plot_training_curves(train_losses, val_losses)

    print("训练完成！")
    print("最佳模型已保存为 'best_model.pt'")
    print("训练曲线已保存为 'training_curves.png'")