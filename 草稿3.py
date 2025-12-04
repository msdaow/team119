import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import matplotlib

# 设置中文字体（解决警告问题）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用系统默认字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 自定义数据集类来读取MNIST二进制文件
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform

        # 读取图像数据
        with open(images_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')

            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            self.images = images.astype(np.float32) / 255.0  # 归一化到[0,1]

        # 读取标签数据
        with open(labels_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            self.labels = labels

        print(f"加载数据集: {num_images} 张图像, {num_labels} 个标签")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # 形状: (28, 28)
        label = self.labels[idx]

        if self.transform:
            # 直接使用numpy数组，不添加通道维度
            image = self.transform(image)
        else:
            # 如果没有transform，手动添加通道维度并转换为tensor
            image = torch.from_numpy(image).unsqueeze(0).float()

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 这会自动将(28,28)转换为(1,28,28)并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))
])

# 文件路径
train_images_path = r"C:\Users\86131\Downloads\train-images.idx3-ubyte"
train_labels_path = r"C:\Users\86131\Downloads\train-labels.idx1-ubyte"
test_images_path = r"C:\Users\86131\Downloads\t10k-images.idx3-ubyte"
test_labels_path = r"C:\Users\86131\Downloads\t10k-labels.idx1-ubyte"

# 检查文件是否存在
use_backup = False
for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
    if not os.path.exists(path):
        print(f"警告: 文件不存在 - {path}")
        use_backup = True

# 加载数据集
print("正在加载数据集...")
if not use_backup:
    try:
        train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
        test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)
        print("成功加载本地MNIST数据集")
    except Exception as e:
        print(f"加载本地数据集失败: {e}")
        print("将使用torchvision的MNIST数据集作为备选...")
        use_backup = True
else:
    print("将使用torchvision的MNIST数据集作为备选...")

if use_backup:
    # 使用torchvision的MNIST数据集作为备选
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 打印一个样本的形状进行检查
sample_image, sample_label = train_dataset[0]
print(f"样本图像形状: {sample_image.shape}")
print(f"样本标签: {sample_label}")


# 定义LeNet网络
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 输入: 1x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 6x28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 16x10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 -> 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义AlexNet网络（适配28x28输入）
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 32x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x7x7
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, model_name="Model"):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"\n开始训练 {model_name}...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 检查数据形状
            if batch_idx == 0 and epoch == 0:
                print(f"输入数据形状: {data.shape}")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 每100个batch打印一次
            if batch_idx % 100 == 0:
                print(
                    f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 测试准确率
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(
            f'Epoch {epoch + 1}: 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

    training_time = time.time() - start_time
    print(f"{model_name} 训练完成! 总训练时间: {training_time:.2f}秒")

    return train_losses, train_accuracies, test_accuracies, training_time


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy


# 模型参数统计函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 初始化模型
lenet = LeNet().to(device)
alexnet = AlexNet().to(device)

print("\n模型参数统计:")
print(f"LeNet 参数数量: {count_parameters(lenet):,}")
print(f"AlexNet 参数数量: {count_parameters(alexnet):,}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
lenet_optimizer = optim.Adam(lenet.parameters(), lr=0.001)
alexnet_optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

# 测试模型前向传播
print("\n测试模型前向传播...")
with torch.no_grad():
    test_input = torch.randn(2, 1, 28, 28).to(device)  # batch_size=2, channels=1, height=28, width=28
    lenet_output = lenet(test_input)
    alexnet_output = alexnet(test_input)
    print(f"LeNet 输出形状: {lenet_output.shape}")
    print(f"AlexNet 输出形状: {alexnet_output.shape}")

# 训练两个模型
print("\n" + "=" * 60)
lenet_losses, lenet_train_acc, lenet_test_acc, lenet_time = train_model(
    lenet, train_loader, test_loader, criterion, lenet_optimizer,
    num_epochs=10, model_name="LeNet"
)

print("=" * 60)
alexnet_losses, alexnet_train_acc, alexnet_test_acc, alexnet_time = train_model(
    alexnet, train_loader, test_loader, criterion, alexnet_optimizer,
    num_epochs=10, model_name="AlexNet"
)

# 最终评估
print("\n" + "=" * 60)
print("最终性能对比:")
lenet_final_acc = evaluate_model(lenet, test_loader)
alexnet_final_acc = evaluate_model(alexnet, test_loader)

print(f"LeNet 最终测试准确率: {lenet_final_acc:.2f}%")
print(f"AlexNet 最终测试准确率: {alexnet_final_acc:.2f}%")
print(f"LeNet 训练时间: {lenet_time:.2f}秒")
print(f"AlexNet 训练时间: {alexnet_time:.2f}秒")


# 绘制训练曲线（使用英文标签避免字体问题）
def plot_comparison(lenet_losses, alexnet_losses,
                    lenet_train_acc, alexnet_train_acc,
                    lenet_test_acc, alexnet_test_acc):
    plt.figure(figsize=(15, 5))

    # Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(lenet_losses, label='LeNet', linewidth=2)
    plt.plot(alexnet_losses, label='AlexNet', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Training Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(lenet_train_acc, label='LeNet', linewidth=2)
    plt.plot(alexnet_train_acc, label='AlexNet', linewidth=2)
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Test Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(lenet_test_acc, label='LeNet', linewidth=2)
    plt.plot(alexnet_test_acc, label='AlexNet', linewidth=2)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 绘制对比图
plot_comparison(lenet_losses, alexnet_losses,
                lenet_train_acc, alexnet_train_acc,
                lenet_test_acc, alexnet_test_acc)

# 模型对比总结
print("\n" + "=" * 70)
print("模型对比总结:")
print(f"{'Metric':<15} {'LeNet':<12} {'AlexNet':<12}")
print(f"{'-' * 45}")
print(f"{'Parameters':<15} {count_parameters(lenet):<12,} {count_parameters(alexnet):<12,}")
print(f"{'Final Accuracy':<15} {lenet_final_acc:<12.2f}% {alexnet_final_acc:<12.2f}%")
print(f"{'Training Time':<15} {lenet_time:<12.2f}s {alexnet_time:<12.2f}s")
print(f"{'Best Epoch':<15} {np.argmax(lenet_test_acc) + 1:<12} {np.argmax(alexnet_test_acc) + 1:<12}")
print(f"{'Best Accuracy':<15} {np.max(lenet_test_acc):<12.2f}% {np.max(alexnet_test_acc):<12.2f}%")

# 详细性能分析
print("\n" + "=" * 70)
print("详细性能分析报告")
print("=" * 70)

print(f"\n1. 准确率对比:")
print(f"   LeNet:  {lenet_final_acc:.2f}%")
print(f"   AlexNet: {alexnet_final_acc:.2f}%")
print(f"   差异:    +{alexnet_final_acc - lenet_final_acc:.2f}% (AlexNet更高)")

print(f"\n2. 训练效率对比:")
print(f"   LeNet训练时间:  {lenet_time:.2f}秒")
print(f"   AlexNet训练时间: {alexnet_time:.2f}秒")
print(f"   时间比:         {alexnet_time / lenet_time:.1f}x (AlexNet更慢)")

print(f"\n3. 模型复杂度对比:")
print(f"   LeNet参数数量:  {count_parameters(lenet):,}")
print(f"   AlexNet参数数量: {count_parameters(alexnet):,}")
print(f"   参数比:         {count_parameters(alexnet) / count_parameters(lenet):.1f}x")

print(f"\n4. 性能总结:")
print(f"   • LeNet: 轻量高效，适合资源受限环境")
print(f"   • AlexNet: 精度更高，但计算成本大")
print(f"   • 对于MNIST任务，LeNet的性价比更好")

print(f"\n5. 推荐使用场景:")
print(f"   ✓ LeNet: 嵌入式设备、移动端、实时应用")
print(f"   ✓ AlexNet: 对精度要求高的场景、研究学习")


# 可视化一些测试样本的预测结果
def visualize_predictions(model, test_loader, model_name, num_samples=12):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        color = 'green' if labels[i] == predicted[i] else 'red'
        plt.title(f'True: {labels[i]}, Pred: {predicted[i]}', color=color)
        plt.axis('off')
    plt.suptitle(f'{model_name} Predictions (Green:Correct, Red:Wrong)')
    plt.tight_layout()
    plt.show()


# 显示两个模型的预测示例
print("\n显示预测示例...")
visualize_predictions(lenet, test_loader, "LeNet")
visualize_predictions(alexnet, test_loader, "AlexNet")

# 保存模型
torch.save(lenet.state_dict(), 'lenet_mnist.pth')
torch.save(alexnet.state_dict(), 'alexnet_mnist.pth')
print("\n模型已保存为 'lenet_mnist.pth' 和 'alexnet_mnist.pth'")

print("\n" + "=" * 70)
print("项目完成！LeNet和AlexNet手写数字识别对比实验成功运行")
print("=" * 70)