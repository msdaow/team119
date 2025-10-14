import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据预处理
class CountryDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型
class EcologicalNetwork(nn.Module):
    def __init__(self, input_size):
        super(EcologicalNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),  # 输入层到第一隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(7, 6),  # 第一隐藏层到第二隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(6, 5),  # 第二隐藏层到第三隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(5, 1)  # 第三隐藏层到输出层
        )

    def forward(self, x):
        return self.network(x)


# 3. 数据加载和预处理函数
def load_and_preprocess_data():
    # 读取数据
    df = pd.read_csv('F:/countries.csv')

    # 选择特征和目标变量
    # 特征：人口、HDI、GDP、各种足迹
    # 目标：总生态足迹
    feature_columns = [
        'Population (millions)', 'HDI', 'GDP per Capita',
        'Cropland Footprint', 'Grazing Footprint', 'Forest Footprint',
        'Carbon Footprint', 'Fish Footprint'
    ]

    target_column = 'Total Ecological Footprint'

    # 数据清洗
    df_clean = df.copy()

    # 处理GDP列（移除$和逗号）
    df_clean['GDP per Capita'] = df_clean['GDP per Capita'].astype(str)
    df_clean['GDP per Capita'] = df_clean['GDP per Capita'].str.replace('$', '').str.replace(',', '')
    df_clean['GDP per Capita'] = pd.to_numeric(df_clean['GDP per Capita'], errors='coerce')

    # 处理其他数值列
    for col in feature_columns + [target_column]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 移除包含NaN的行
    df_clean = df_clean[feature_columns + [target_column]].dropna()

    # 分离特征和目标
    X = df_clean[feature_columns].values
    y = df_clean[target_column].values.reshape(-1, 1)

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y, df_clean


# 4. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses, best_model_state


# 5. 主程序
def main():
    print("Starting data preprocessing...")
    X, y, scaler_X, scaler_y, df_clean = load_and_preprocess_data()

    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )

    print(f"Training set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}")

    # 创建数据集和数据加载器
    train_dataset = CountryDataset(X_train, y_train)
    val_dataset = CountryDataset(X_val, y_val)
    test_dataset = CountryDataset(X_test, y_test)

    # 设置batch_size和num_workers
    # batch_size: 根据数据集大小选择，通常为16-64
    # num_workers: 根据CPU核心数选择，通常为0-8
    batch_size = 16
    num_workers = 2  # 大多数电脑配置适用

    print(f"Batch size: {batch_size}, Num workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 创建模型
    input_size = X.shape[1]
    model = EcologicalNetwork(input_size)

    print(f"Model structure: Input layer({input_size}) -> Hidden layer(7) -> Hidden layer(6) -> Hidden layer(5) -> Output layer(1)")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Starting model training...")
    train_losses, val_losses, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=200
    )

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_size': input_size
    }, 'best_ecological_model.pt')

    print("Best model saved as 'best_ecological_model.pt'")

    # 测试模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test loss: {test_loss:.4f}")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 预测可视化
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        test_predictions = []
        test_actual = []
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_predictions.extend(outputs.numpy())
            test_actual.extend(batch_y.numpy())

        test_predictions = np.array(test_predictions)
        test_actual = np.array(test_actual)

        # 反标准化
        test_predictions_original = scaler_y.inverse_transform(test_predictions)
        test_actual_original = scaler_y.inverse_transform(test_actual)

        plt.scatter(test_actual_original, test_predictions_original, alpha=0.6)
        plt.plot([test_actual_original.min(), test_actual_original.max()],
                 [test_actual_original.min(), test_actual_original.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印模型总结
    print("\nModel Summary:")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"- Best validation loss: {min(val_losses):.4f}")
    print(f"- Final test loss: {test_loss:.4f}")

    # 特征重要性分析（基于梯度）
    print("\nFeature Importance Analysis:")
    feature_names = ['Population', 'HDI', 'GDP', 'Cropland', 'Grazing', 'Forest', 'Carbon', 'Fish']
    model.eval()

    # 计算特征梯度
    X_tensor = torch.FloatTensor(X_train[:10])  # 使用少量样本
    X_tensor.requires_grad = True

    outputs = model(X_tensor)
    outputs.mean().backward()

    gradients = X_tensor.grad.abs().mean(dim=0).numpy()

    # 归一化并显示重要性
    importance = gradients / gradients.sum()
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.3f}")


if __name__ == "__main__":
    main()