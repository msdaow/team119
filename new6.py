import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
class CountriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型
class CountriesNet(nn.Module):
    def __init__(self, input_size):
        super(CountriesNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),
            nn.ReLU(),
            nn.Linear(7, 6),
            nn.ReLU(),
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.network(x)


# 3. 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 选择数值型特征和目标变量
    # 使用HDI作为目标变量，选择其他数值特征作为输入
    numeric_columns = ['Population (millions)', 'GDP per Capita', 'Cropland Footprint',
                       'Grazing Footprint', 'Forest Footprint', 'Carbon Footprint',
                       'Fish Footprint', 'Total Ecological Footprint', 'Cropland',
                       'Grazing Land', 'Forest Land', 'Fishing Water', 'Urban Land',
                       'Total Biocapacity', 'Biocapacity Deficit or Reserve',
                       'Earths Required', 'Countries Required']

    # 清理数据：移除非数值字符
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # 处理目标变量
    df = df.dropna(subset=['HDI'])
    target = df['HDI'].values

    # 选择特征
    feature_columns = [col for col in numeric_columns if col in df.columns]
    features = df[feature_columns].fillna(0).values  # 用0填充缺失值

    return features, target, feature_columns


# 4. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses, best_model_state


# 5. 可视化函数
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    # 加载数据
    print("Loading and preprocessing data...")
    features, target, feature_columns = load_and_preprocess_data('countries.csv')

    print(f"Dataset shape: {features.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")

    # 数据标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 创建数据集
    train_dataset = CountriesDataset(X_train, y_train)
    val_dataset = CountriesDataset(X_val, y_val)
    test_dataset = CountriesDataset(X_test, y_test)

    # 创建数据加载器
    # Batch size选择依据：考虑到数据集大小和内存限制
    # 对于中等规模数据集，通常选择16-64的batch size
    # num_workers选择依据：根据CPU核心数，通常设置为CPU核心数或略少
    batch_size = 32
    num_workers = 2  # 对于大多数个人电脑，2-4个workers是安全的

    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # 创建模型
    input_size = len(feature_columns)
    model = CountriesNet(input_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    print("Starting training...")
    num_epochs = 100
    train_losses, val_losses, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最佳模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler,
        'feature_columns': feature_columns,
        'best_val_loss': min(val_losses)
    }, 'best_countries_model.pt')

    print("Best model saved as 'best_countries_model.pt'")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses)

    # 在测试集上评估
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Final Test Loss: {test_loss:.4f}')

    # 打印模型结构
    print("\nModel Architecture:")
    print(model)


if __name__ == "__main__":
    main()