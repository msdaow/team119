import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
print("正在加载MNIST数据集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义简单的CNN模型（修复维度问题）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 输入:1, 输出:32, 卷积核:3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 输入:32, 输出:64, 卷积核:3x3
        self.pool = nn.MaxPool2d(2)  # 最大池化 2x2

        # 计算全连接层的输入维度
        # 输入: 28x28
        # conv1后: (28-3+1)=26x26 -> pool后: 13x13
        # conv2后: (13-3+1)=11x11 -> pool后: 5x5
        # 所以最终特征图大小: 64 * 5 * 5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 修正维度
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层1
        x = self.relu(self.conv1(x))  # [batch, 32, 26, 26]
        x = self.pool(x)  # [batch, 32, 13, 13]

        # 卷积层2
        x = self.relu(self.conv2(x))  # [batch, 64, 11, 11]
        x = self.pool(x)  # [batch, 64, 5, 5]

        # 展平
        x = x.view(-1, 64 * 5 * 5)  # [batch, 1600]

        # 全连接层
        x = self.relu(self.fc1(x))  # [batch, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 10]
        return x


# 创建模型
model = SimpleCNN().to(device)
print("模型创建完成")

# 打印模型结构
print("\n模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练过程
train_losses = []
train_accuracies = []
test_accuracies = []


# 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f'Epoch {epoch}: 训练损失: {avg_loss:.4f}, 训练准确率: {accuracy:.2f}%')


# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    test_accuracies.append(accuracy)

    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy


# 开始训练
print("\n开始训练...")
epochs = 5

for epoch in range(1, epochs + 1):
    print(f"\n=== Epoch {epoch}/{epochs} ===")
    train(epoch)
    test_accuracy = test()
    print('-' * 50)

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='训练损失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('训练损失曲线')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'b-', label='训练准确率', linewidth=2)
plt.plot(test_accuracies, 'r-', label='测试准确率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.title('准确率曲线')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n最终结果:")
print(f"最终训练准确率: {train_accuracies[-1]:.2f}%")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存为: mnist_cnn_model.pth")