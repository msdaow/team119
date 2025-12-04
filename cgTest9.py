import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import struct
import time
from torch.utils.data import Dataset, DataLoader


# ==================== 1. 加载MNIST数据 ====================
def load_mnist_images(filename):
    """读取MNIST图像文件"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
        images = images.astype(np.float32) / 255.0  # 归一化
    return torch.from_numpy(images)


def load_mnist_labels(filename):
    """读取MNIST标签文件"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return torch.from_numpy(labels).long()


class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = load_mnist_images(images_path)
        self.labels = load_mnist_labels(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # MNIST标准化 (均值0.1307, 标准差0.3081)
        image = (self.images[idx] - 0.1307) / 0.3081
        return image, self.labels[idx]


# ==================== 2. GoogLeNet 模型 ====================
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 4个分支
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.inception1 = InceptionBlock(10)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88 = 16+24+24+24
        self.inception2 = InceptionBlock(20)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)  # 88*4*4=1408

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))  # [B,10,12,12]
        x = self.inception1(x)  # [B,88,12,12]
        x = F.relu(self.pool(self.conv2(x)))  # [B,20,4,4]
        x = self.inception2(x)  # [B,88,4,4]
        x = x.view(x.size(0), -1)  # 展平
        return F.log_softmax(self.fc(x), dim=1)


# ==================== 3. ResNet 模型 ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)  # 残差连接


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.res1 = ResidualBlock(16)
        self.res2 = ResidualBlock(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res3 = ResidualBlock(32)
        self.res4 = ResidualBlock(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)  # 32*4*4=512

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))  # [B,16,12,12]
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.pool(self.conv2(x)))  # [B,32,4,4]
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)  # 展平
        return F.log_softmax(self.fc(x), dim=1)


# ==================== 4. 训练和测试函数 ====================
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    return total_loss / len(train_loader), 100. * correct / len(train_loader.dataset)


def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return total_loss, accuracy


# ==================== 5. 主程序 ====================
def main():
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 加载数据
    print("加载MNIST数据...")
    train_dataset = MNISTDataset(
        r"C:\Users\86131\Downloads\train-images.idx3-ubyte",
        r"C:\Users\86131\Downloads\train-labels.idx1-ubyte"
    )
    test_dataset = MNISTDataset(
        r"C:\Users\86131\Downloads\t10k-images.idx3-ubyte",
        r"C:\Users\86131\Downloads\t10k-labels.idx1-ubyte"
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")

    # 创建模型
    googlenet = GoogleNet().to(device)
    resnet = ResNet().to(device)

    # 优化器
    googlenet_opt = optim.Adam(googlenet.parameters(), lr=0.001)
    resnet_opt = optim.Adam(resnet.parameters(), lr=0.001)

    # 训练记录
    results = {
        'googlenet': {'train_loss': [], 'train_acc': [], 'test_acc': []},
        'resnet': {'train_loss': [], 'train_acc': [], 'test_acc': []}
    }

    # 训练
    epochs = 5
    print(f"\n开始训练，共{epochs}轮...")

    for epoch in range(1, epochs + 1):
        print(f"\n轮次 {epoch}/{epochs}")

        # 训练GoogLeNet
        loss, acc = train_model(googlenet, train_loader, googlenet_opt, device)
        test_loss, test_acc = test_model(googlenet, test_loader, device)
        results['googlenet']['train_loss'].append(loss)
        results['googlenet']['train_acc'].append(acc)
        results['googlenet']['test_acc'].append(test_acc)
        print(f"GoogLeNet - 训练损失: {loss:.4f}, 测试准确率: {test_acc:.2f}%")

        # 训练ResNet
        loss, acc = train_model(resnet, train_loader, resnet_opt, device)
        test_loss, test_acc = test_model(resnet, test_loader, device)
        results['resnet']['train_loss'].append(loss)
        results['resnet']['train_acc'].append(acc)
        results['resnet']['test_acc'].append(test_acc)
        print(f"ResNet    - 训练损失: {loss:.4f}, 测试准确率: {test_acc:.2f}%")

    # ==================== 6. 结果可视化 ====================
    plt.figure(figsize=(15, 5))

    # 测试准确率对比
    plt.subplot(1, 3, 1)
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, results['googlenet']['test_acc'], 'b-o', label='GoogLeNet')
    plt.plot(epochs_range, results['resnet']['test_acc'], 'r-s', label='ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('测试准确率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 训练损失对比
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, results['googlenet']['train_loss'], 'b-o', label='GoogLeNet')
    plt.plot(epochs_range, results['resnet']['train_loss'], 'r-s', label='ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最终性能对比
    plt.subplot(1, 3, 3)
    models = ['GoogLeNet', 'ResNet']
    final_acc = [results['googlenet']['test_acc'][-1], results['resnet']['test_acc'][-1]]
    colors = ['blue', 'red']
    plt.bar(models, final_acc, color=colors)
    plt.ylabel('Final Accuracy (%)')
    plt.title('最终性能对比')
    for i, acc in enumerate(final_acc):
        plt.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('cnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== 7. 打印总结 ====================
    print("\n" + "=" * 50)
    print("实验总结")
    print("=" * 50)

    print(f"\nGoogLeNet 最终准确率: {results['googlenet']['test_acc'][-1]:.2f}%")
    print(f"ResNet 最终准确率:    {results['resnet']['test_acc'][-1]:.2f}%")

    # 保存模型
    torch.save(googlenet.state_dict(), 'googlenet_simple.pth')
    torch.save(resnet.state_dict(), 'resnet_simple.pth')
    print("\n模型已保存: googlenet_simple.pth, resnet_simple.pth")


if __name__ == "__main__":
    main()