import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 配置参数与数据加载 --------------------------
# 数据集路径（替换为你的实际路径）
DATA_PATH = r"D:"  # 包含四个.gz文件的文件夹
BATCH_SIZE = 64
EPOCHS = 10  # 训练轮数（与PPT一致）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择GPU/CPU

# 数据预处理：转换为张量+标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集统计均值和方差
])

# 加载训练集（含验证集，这里直接用训练集训练，测试集评估）
train_dataset = datasets.MNIST(
    root=DATA_PATH, train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root=DATA_PATH, train=False, download=False, transform=transform
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------- 2. 定义FC（全连接）模型 --------------------------
class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net, self).__init__()
        # 全连接层：784(28x28) → 512 → 256 → 128 → 64 → 10（与PPT一致）
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平：(batch,1,28,28) → (batch,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# -------------------------- 3. 定义CNN模型 --------------------------
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        # 卷积层1：1→10通道，5x5卷积核（与PPT一致）
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层2：10→20通道，5x5卷积核（与PPT一致）
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 池化层：2x2最大池化（与PPT一致）
        self.pool = nn.MaxPool2d(2)
        # 全连接层：20*4*4=320 → 10（与PPT一致）
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # 第一层：卷积→ReLU→池化
        x = self.pool(F.relu(self.conv1(x)))  # (batch,1,28,28)→(batch,10,12,12)
        # 第二层：卷积→ReLU→池化
        x = self.pool(F.relu(self.conv2(x)))  # (batch,10,12,12)→(batch,20,4,4)
        # 展平：(batch,20,4,4)→(batch,320)
        x = x.view(-1, 320)
        # 全连接层输出
        return self.fc(x)


# -------------------------- 4. 训练与测试函数 --------------------------
def train_model(model, train_loader, criterion, optimizer, epoch):
    """训练单个epoch"""
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        running_loss += loss.item()

    # 打印每轮平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, test_loader):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)  # 获取预测类别
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# -------------------------- 5. 模型训练与性能记录 --------------------------
# 初始化模型、损失函数、优化器（两个模型配置一致，保证公平对比）
fc_model = FC_Net().to(DEVICE)
cnn_model = CNN_Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适合分类任务）
fc_optimizer = torch.optim.SGD(fc_model.parameters(), lr=0.01, momentum=0.9)
cnn_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# 记录准确率曲线
fc_acc_list = []
cnn_acc_list = []
epochs_list = list(range(1, EPOCHS + 1))

# 训练与测试
for epoch in range(EPOCHS):
    print("\n=== Training FC Model ===")
    train_model(fc_model, train_loader, criterion, fc_optimizer, epoch)
    fc_acc = test_model(fc_model, test_loader)
    fc_acc_list.append(fc_acc)

    print("\n=== Training CNN Model ===")
    train_model(cnn_model, train_loader, criterion, cnn_optimizer, epoch)
    cnn_acc = test_model(cnn_model, test_loader)
    cnn_acc_list.append(cnn_acc)

# -------------------------- 6. 绘制准确率曲线 --------------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, fc_acc_list, label="FC (Full Connected)", marker='o', linewidth=2)
plt.plot(epochs_list, cnn_acc_list, label="CNN (Convolutional)", marker='s', linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title("MNIST Classification: CNN vs FC Accuracy Curve", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(epochs_list)
plt.ylim(80, 100)  # 限定y轴范围，更清晰展示差异
plt.savefig("mnist_cnn_vs_fc_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# 输出最终性能对比
print("\n=== Final Performance Comparison ===")
print(f"FC Model Final Accuracy: {fc_acc_list[-1]:.2f}%")
print(f"CNN Model Final Accuracy: {cnn_acc_list[-1]:.2f}%")