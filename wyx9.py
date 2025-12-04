#GoogleNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# 修正版GoogleNet - 修复维度问题
class SimpleGoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleGoogleNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Inception模块 - 修复通道数
        self.inception1 = InceptionModule(64, 32, 32, 32, 16, 16)  # 输出: 32+32+16+16=96
        self.inception2 = InceptionModule(96, 64, 48, 48, 24, 24)  # 输出: 64+48+24+24=160

        # 自适应池化层，确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层 - 修正输入维度
        self.fc = nn.Linear(160 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 64x16x16

        x = self.inception1(x)  # 96x16x16
        x = self.pool(x)  # 96x8x8

        x = self.inception2(x)  # 160x8x8
        x = self.adaptive_pool(x)  # 160x4x4

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5):
        super(InceptionModule, self).__init__()

        # 1x1卷积分支
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3池化 -> 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, ch5x5, kernel_size=1)
        )

    def forward(self, x):
        branch1 = torch.relu(self.branch1(x))
        branch2 = torch.relu(self.branch2(x))
        branch3 = torch.relu(self.branch3(x))
        branch4 = torch.relu(self.branch4(x))

        return torch.cat([branch1, branch2, branch3, branch4], 1)


# 初始化模型
model = SimpleGoogleNet(num_classes=10).to(device)
print("修正版GoogleNet模型已创建")

# 打印模型结构
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
train_losses = []
train_accuracies = []

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

print("训练完成!")

# 测试模型
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'\n测试集准确率: {test_accuracy:.2f}%')

# 计算评估指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print("\n=== 评估指标 ===")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确度 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 绘制loss图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', linewidth=2, label='训练损失')
plt.title('GoogleNet训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'r-s', linewidth=2, label='训练准确率')
plt.title('GoogleNet训练准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# 随机预测一张图片
def predict_random_image():
    model.eval()
    with torch.no_grad():
        # 随机选择一张测试图片
        data_iter = iter(test_loader)
        images, labels = next(data_iter)

        idx = random.randint(0, len(images) - 1)
        image = images[idx:idx + 1].to(device)
        true_label = labels[idx].item()

        # 预测
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

        probs = probabilities[0].cpu().numpy()

        # 显示结果
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.title(f'测试图像\n真实数字: {true_label}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        colors = ['red' if i != predicted_label else 'green' for i in range(10)]
        plt.bar(range(10), probs, color=colors, alpha=0.7)
        plt.title(f'预测结果: {predicted_label}\n概率: {probs[predicted_label]:.4f}')
        plt.xlabel('数字')
        plt.ylabel('概率')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印结果
        print(f"\n随机预测结果:")
        print(f"真实数字: {true_label}, 预测数字: {predicted_label}")
        print(f"预测概率: {probs[predicted_label]:.4f}")
        print(f"预测{'正确' if true_label == predicted_label else '错误'}")

        return true_label, predicted_label


print("\n随机预测一张测试图片...")
true_label, pred_label = predict_random_image()

# 保存模型
torch.save(model.state_dict(), 'mnist_googlenet.pth')
print("\n模型已保存为 'mnist_googlenet.pth'")

# 打印统计信息
print(f"\n=== 训练统计 ===")
print(f"最终训练损失: {train_losses[-1]:.4f}")
print(f"最终训练准确率: {train_accuracies[-1]:.2f}%")
print(f"测试集准确率: {test_accuracy:.2f}%")

#ResNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# 定义简化版ResNet
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差块
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)

        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 初始化模型
model = SimpleResNet(num_classes=10).to(device)
print("简化版ResNet模型已创建")

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
train_losses = []
train_accuracies = []

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

print("训练完成!")

# 测试模型
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'\n测试集准确率: {test_accuracy:.2f}%')

# 计算评估指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print("\n=== 评估指标 ===")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确度 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 绘制loss图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', linewidth=2, label='训练损失')
plt.title('ResNet训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'r-s', linewidth=2, label='训练准确率')
plt.title('ResNet训练准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# 随机预测一张图片
def predict_random_image():
    model.eval()
    with torch.no_grad():
        # 随机选择一张测试图片
        data_iter = iter(test_loader)
        images, labels = next(data_iter)

        idx = random.randint(0, len(images) - 1)
        image = images[idx:idx + 1].to(device)
        true_label = labels[idx].item()

        # 预测
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

        probs = probabilities[0].cpu().numpy()

        # 显示结果
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.title(f'测试图像\n真实数字: {true_label}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        colors = ['red' if i != predicted_label else 'green' for i in range(10)]
        plt.bar(range(10), probs, color=colors, alpha=0.7)
        plt.title(f'预测结果: {predicted_label}\n概率: {probs[predicted_label]:.4f}')
        plt.xlabel('数字')
        plt.ylabel('概率')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印结果
        print(f"\n随机预测结果:")
        print(f"真实数字: {true_label}, 预测数字: {predicted_label}")
        print(f"预测概率: {probs[predicted_label]:.4f}")
        print(f"预测{'正确' if true_label == predicted_label else '错误'}")

        return true_label, predicted_label


print("\n随机预测一张测试图片...")
true_label, pred_label = predict_random_image()

# 保存模型
torch.save(model.state_dict(), 'mnist_resnet.pth')
print("\n模型已保存为 'mnist_resnet.pth'")

# 打印统计信息
print(f"\n=== 训练统计 ===")
print(f"最终训练损失: {train_losses[-1]:.4f}")
print(f"最终训练准确率: {train_accuracies[-1]:.2f}%")
print(f"测试集准确率: {test_accuracy:.2f}%")