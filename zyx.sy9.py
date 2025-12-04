import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from PIL import Image
import os
import platform

# --------------------------
# 1. 跨系统中文显示配置（保留但简化）
# --------------------------
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 强制指定CPU（关键！避免设备匹配问题）
device = torch.device("cpu")

# --------------------------
# 2. 数据加载与预处理（轻量化batch_size）
# --------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST官方归一化参数
])

# 加载MNIST（无数据集则设download=True）
train_dataset = datasets.MNIST(
    root=r"F:\MobileFile\images\images",
    train=True,
    transform=transform,
    download=False
)
test_dataset = datasets.MNIST(
    root=r"F:\MobileFile\images\images",
    train=False,
    transform=transform,
    download=False
)

# 极小batch_size适配CPU（64→8）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)


# --------------------------
# 3. 轻量化ResNet（核心修改）
# --------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class LightResNet_MNIST(nn.Module):
    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1], num_classes=10):
        super().__init__()
        self.in_channels = 16

        # 初始卷积层（1→16通道，适配单通道输入）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 残差层（通道数从64/128/256/512→16/32/64/128）
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)  # 32x32→32x32
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 32x32→16x16
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 16x16→8x8
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 8x8→4x4

        # 分类头（适配最终通道数128）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# --------------------------
# 4. 模型初始化
# --------------------------
model = LightResNet_MNIST(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# --------------------------
# 5. 训练模型（简化+进度反馈）
# --------------------------
num_epochs = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    # 打印轮次进度
    print(f"\n===== Epoch {epoch + 1}/{num_epochs} 训练开始 =====")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # 训练指标计算
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # 测试阶段
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_true_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())  # 转回CPU用于计算指标
            all_true_labels.extend(labels.cpu().numpy())
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # 测试指标计算
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    # 计算多分类指标
    accuracy = accuracy_score(all_true_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_preds, average='weighted'
    )

    # 打印轮次结果
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# --------------------------
# 6. 绘制训练曲线（简化版）
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss曲线
ax1.plot(range(1, num_epochs + 1), train_losses, label='训练损失', color='blue')
ax1.plot(range(1, num_epochs + 1), test_losses, label='测试损失', color='red')
ax1.set_xlabel('训练轮次')
ax1.set_ylabel('损失值')
ax1.set_title('ResNet训练/测试损失曲线 (MNIST)')
ax1.legend()
ax1.grid(True)

# Accuracy曲线
ax2.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率', color='blue')
ax2.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率', color='red')
ax2.set_xlabel('训练轮次')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('ResNet训练/测试准确率曲线 (MNIST)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves_resnet_mnist.png')
plt.show()


# --------------------------
# 7. 手写数字预测（适配CPU）
# --------------------------
def predict_and_display_handwritten_digit(image_path, model):
    if not os.path.exists(image_path):
        print(f"错误：图片文件 '{image_path}' 不存在。")
        return None

    try:
        # 图片预处理
        img_pil = Image.open(image_path).convert('L')
        img_resized = img_pil.resize((32, 32))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(device)  # 绑定CPU

        # 预测
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()

        # 显示结果
        plt.figure(figsize=(6, 6))
        plt.imshow(img_pil, cmap='gray')
        plt.axis('off')
        plt.title(f'预测结果: {predicted_digit}', fontsize=16)
        plt.show()

        return predicted_digit

    except Exception as e:
        print(f"处理图片或预测时发生错误: {str(e)}")
        return None


# 调用预测函数
handwritten_image_path = r"F:\MobileFile\images\images\handwritten_digit1.jpg"
predicted_digit = predict_and_display_handwritten_digit(handwritten_image_path, model)

if predicted_digit is not None:
    print(f"\n手写数字的预测结果为: {predicted_digit}")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # 补全指标导入
import matplotlib.pyplot as plt
from PIL import Image
import os

# Windows中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------
# 轻量化Inception模块（通道数严格匹配）
# --------------------------
class LightInception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, ch1x1, 1), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, ch3x3red, 1), nn.ReLU(True),
                                     nn.Conv2d(ch3x3red, ch3x3, 3, padding=1), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, ch5x5red, 1), nn.ReLU(True),
                                     nn.Conv2d(ch5x5red, ch5x5, 5, padding=2), nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1),
                                     nn.Conv2d(in_channels, pool_proj, 1), nn.ReLU(True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# --------------------------
# 轻量化GoogLeNet（通道数全匹配）
# --------------------------
class LightGoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=False):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32→16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16→8
        )

        self.inception3a = LightInception(32, 16, 16, 16, 8, 8, 8)  # 输出48
        self.inception3b = LightInception(48, 24, 24, 24, 12, 12, 12)  # 输出72

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(72, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# --------------------------
# 数据预处理
# --------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root=r"F:\MobileFile\images\images",
    train=True,
    transform=transform,
    download=False
)
test_dataset = datasets.MNIST(
    root=r"F:\MobileFile\images\images",
    train=False,
    transform=transform,
    download=False
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# --------------------------
# 模型初始化
# --------------------------
device = torch.device("cpu")
model = LightGoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练
num_epochs = 10
train_losses, test_losses = [], []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if i % 500 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i}/{len(train_loader)}")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试阶段（计算完整指标）
    model.eval()
    total_test_loss = 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total_test_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_true.extend(labels.numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # 计算所有指标（匹配目标格式）
    acc = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='weighted'
    )

    # 按目标格式打印结果
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
          f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# --------------------------
# 绘制损失曲线
# --------------------------
plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
plt.plot(range(1, num_epochs + 1), test_losses, label='测试损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('轻量化GoogLeNet损失曲线（MNIST）')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_fixed.png')
plt.show()


# --------------------------
# 手写数字预测
# --------------------------
def predict_digit(image_path):
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return None
    try:
        img = Image.open(image_path).convert('L').resize((32, 32))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, 1).item()

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'预测结果：{pred}')
        plt.show()
        return pred
    except Exception as e:
        print(f"预测出错：{str(e)}")
        return None


# 调用预测
pred = predict_digit(r"F:\MobileFile\images\images\handwritten_digit2.jpg")
if pred is not None:
    print(f"\n最终预测结果：{pred}")