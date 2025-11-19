import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt


# 1. 定义数据集类
class CountriesDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# 2. 定义5层神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),  # 第一隐藏层：7个神经元
            nn.ReLU(),
            nn.Linear(7, 6),  # 第二隐藏层：6个神经元
            nn.ReLU(),
            nn.Linear(6, 5),  # 第三隐藏层：5个神经元
            nn.ReLU(),
            nn.Linear(5, 1)  # 输出层
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    # 3. 加载数据
    df = pd.read_csv('countries.csv')

    # 4. 选择数据最完整的特征列
    # 根据数据完整度分析，选择以下特征（缺失值最少）：
    feature_columns = [
        'Population (millions)',  # 人口数据最完整
        'Carbon Footprint',  # 碳足迹数据相对完整
        'Total Biocapacity'  # 生物承载力数据相对完整
    ]
    target_column = 'Total Ecological Footprint'  # 总生态足迹

    # 数据清洗
    data_for_training = df[feature_columns + [target_column]].copy()

    # 转换为数值类型
    for col in feature_columns + [target_column]:
        data_for_training[col] = pd.to_numeric(data_for_training[col], errors='coerce')

    # 删除包含NaN的行
    clean_data = data_for_training.dropna()

    # 准备数据
    x_data = clean_data[feature_columns].values
    y_data = clean_data[target_column].values.reshape(-1, 1)

    print(f"数据形状: 特征 {x_data.shape}, 目标 {y_data.shape}")
    print(f"使用的特征: {feature_columns}")
    print(f"目标变量: {target_column}")

    # 5. 创建数据集和数据加载器
    dataset = CountriesDataset(x_data, y_data)

    # DataLoader配置依据：
    # batch_size=16 - 基于31.7GB大内存，平衡训练速度和内存使用
    # num_workers=0 - 避免多进程问题，保证稳定性
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # 6. 初始化模型
    input_size = x_data.shape[1]
    model = NeuralNetwork(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"模型输入维度: {input_size}")

    # 7. 训练循环
    num_epochs = 1000
    losses = []
    best_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            # 前向传播
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict().copy()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')

    # 8. 保存最佳模型
    torch.save(best_model, 'best_model.pt')
    print("最佳模型已保存为 best_model.pt")

    # 9. 保存训练过程可视化图像
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss: Multiple Features -> Total Ecological Footprint')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    print("训练完成！可视化图像已保存为 training_loss.png")