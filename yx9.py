import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib as mpl

# -------------------------- 核心配置（修复维度错误） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = True if device.type == 'cuda' else False
print(f"使用设备：{device} | pin_memory：{pin_memory}")

# 配置参数（CPU优化）
DATA_ROOT = r'D:/'
MODEL_SAVE_PATH = r'D:/mnist_googlenet_best.pth'
os.makedirs(DATA_ROOT, exist_ok=True)
batch_size, lr, epochs = 64, 3e-4, 10
num_classes = 10

# -------------------------- 数据预处理 --------------------------
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])

infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: 1 - x)
])

# -------------------------- 数据加载 --------------------------
train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=train_transform)
test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)


# -------------------------- Inception块（维度精确计算） --------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3, out5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out1x1, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out3x3, 3, padding=1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, out5x5, 5, padding=2), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1), nn.Conv2d(in_channels, out_pool, 1),
                                     nn.ReLU(inplace=True))
        # 计算输出通道数（供调试）
        self.output_channels = out1x1 + out3x3 + out5x5 + out_pool

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# -------------------------- GooLeNet（维度严格匹配） --------------------------
class GoogLeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 初始卷积：1x28x28 → 64x7x7
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )

        # Inception块序列（严格计算通道数）
        self.inception1 = InceptionBlock(64, 24, 32, 12, 12)  # 24+32+12+12=80
        self.inception2 = InceptionBlock(80, 32, 48, 16, 16)  # 32+48+16+16=112
        self.inception3 = InceptionBlock(112, 48, 64, 24, 24)  # 48+64+24+24=160

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 160x7x7 → 160x1x1
        # 分类器：输入160维（与上面输出一致）
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(160, 96),  # 输入=160（关键修复！）
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.avg_pool(x).flatten(1)  # 展平为(batch_size, 160)
        x = self.classifier(x)
        return x


# -------------------------- 模型初始化 --------------------------
model = GoogLeNetMNIST(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


# -------------------------- 训练测试函数 --------------------------
def train_epoch():
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate():
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


# -------------------------- 可视化函数 --------------------------
def plot_metrics(train_losses, test_losses, accs):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(test_losses, 'r-', label='测试损失')
    plt.xlabel('轮次'), plt.ylabel('损失'), plt.legend(), plt.grid(alpha=0.3)
    plt.subplot(122)
    plt.plot(accs, 'g-', label='测试准确率')
    plt.xlabel('轮次'), plt.ylabel('准确率'), plt.legend(), plt.grid(alpha=0.3)
    plt.tight_layout(), plt.show()


# -------------------------- 推理函数 --------------------------
def infer_digit(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = infer_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"图像读取失败：{e}")
        return None

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    plt.figure(figsize=(9, 3))
    plt.subplot(131), plt.imshow(image), plt.title('原始图像'), plt.axis('off')
    plt.subplot(132), plt.imshow(img_tensor.squeeze().cpu().numpy(), cmap='gray'), plt.title('预处理后'), plt.axis(
        'off')
    plt.subplot(133), plt.axis('off'), plt.gca().set_facecolor('#f0f0f0')
    plt.text(0.5, 0.6, f'预测：{pred}', fontsize=30, ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.4, f'置信度：{confidence:.3f}', fontsize=16, ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('GooLeNet识别结果'), plt.tight_layout(), plt.show()
    return pred


# -------------------------- 主执行流程 --------------------------
if __name__ == "__main__":
    TRAIN_MODEL = True
    best_acc = 0.0

    if TRAIN_MODEL:
        print(f"开始训练（{epochs}轮）...")
        train_losses, test_losses, accs = [], [], []
        for epoch in range(epochs):
            train_loss = train_epoch()
            test_loss, acc, f1 = evaluate()
            scheduler.step()
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accs.append(acc)
            print(
                f'Epoch [{epoch + 1}/{epochs}] | 训练损失：{train_loss:.4f} | 测试损失：{test_loss:.4f} | 准确率：{acc:.4f} | F1：{f1:.4f}')

        plot_metrics(train_losses, test_losses, accs)
        print(f"\n训练完成！最佳准确率：{best_acc:.4f} | 模型保存路径：{MODEL_SAVE_PATH}")
    else:
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print(f"加载模型成功！")
            _, acc, _ = evaluate()
            print(f"加载模型的测试准确率：{acc:.4f}")
        else:
            print("未找到模型文件，请先训练！")
            exit()

    # 推理
    IMAGE_PATH = r'D:/images/sz5.png'
    if os.path.exists(IMAGE_PATH):
        pred = infer_digit(IMAGE_PATH)
        print(f"\n最终识别结果：{pred}")
    else:
        print(f"图像文件不存在：{IMAGE_PATH}")
        print("建议：将手写数字图像保存到桌面，路径改为：r'C:/Users/池棠/Desktop/test.png'")
















        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        import matplotlib.pyplot as plt
        import os
        from PIL import Image
        import matplotlib as mpl

        # -------------------------- 核心配置 --------------------------
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = True if device.type == 'cuda' else False
        print(f"使用设备：{device} | pin_memory：{pin_memory}")

        # 配置参数
        DATA_ROOT = r'D:/'
        MODEL_SAVE_PATH = r'D:/mnist_resnet_best.pth'
        os.makedirs(DATA_ROOT, exist_ok=True)
        batch_size, lr, epochs = 64, 3e-4, 10
        num_classes = 10

        # -------------------------- 数据预处理 --------------------------
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])

        infer_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: 1 - x)
        ])

        # -------------------------- 数据加载 --------------------------
        train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=train_transform)
        test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)


        # -------------------------- ResNet核心模块 --------------------------
        class BasicBlock(nn.Module):
            """ResNet基础残差块（适用于ResNet18/34）"""
            expansion = 1

            def __init__(self, in_channels, out_channels, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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


        # -------------------------- 适配MNIST的ResNet --------------------------
        class ResNetMNIST(nn.Module):
            def __init__(self, block=BasicBlock, layers=[2, 2, 2], num_classes=10):
                super().__init__()
                self.in_channels = 16

                # 初始卷积层：适配MNIST 1x28x28输入
                self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(self.in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16x28x28 → 16x14x14

                # 残差层
                self.layer1 = self._make_layer(block, 16, layers[0], stride=1)  # 16x14x14 → 16x14x14
                self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 16x14x14 → 32x7x7
                self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 32x7x7 → 64x4x4

                # 全局平均池化 + 分类器
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 64x4x4 → 64x1x1
                self.fc = nn.Linear(64 * block.expansion, num_classes)

                # 初始化权重
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
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return x


        # -------------------------- 模型初始化 --------------------------
        # 创建简化版ResNet（3层残差块，适配MNIST小尺寸）
        model = ResNetMNIST(BasicBlock, [2, 2, 2], num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


        # -------------------------- 训练测试函数 --------------------------
        def train_epoch():
            model.train()
            total_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.size(0)
            return total_loss / len(train_loader.dataset)


        def evaluate():
            model.eval()
            total_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    total_loss += criterion(output, target).item() * data.size(0)
                    all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            avg_loss = total_loss / len(test_loader.dataset)
            acc = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            return avg_loss, acc, f1


        # -------------------------- 可视化函数 --------------------------
        def plot_metrics(train_losses, test_losses, accs):
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.plot(train_losses, 'b-', label='训练损失')
            plt.plot(test_losses, 'r-', label='测试损失')
            plt.xlabel('轮次'), plt.ylabel('损失'), plt.legend(), plt.grid(alpha=0.3)
            plt.subplot(122)
            plt.plot(accs, 'g-', label='测试准确率')
            plt.xlabel('轮次'), plt.ylabel('准确率'), plt.legend(), plt.grid(alpha=0.3)
            plt.tight_layout(), plt.show()


        # -------------------------- 推理函数 --------------------------
        def infer_digit(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                img_tensor = infer_transform(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"图像读取失败：{e}")
                return None

            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred].item()

            plt.figure(figsize=(9, 3))
            plt.subplot(131), plt.imshow(image), plt.title('原始图像'), plt.axis('off')
            plt.subplot(132), plt.imshow(img_tensor.squeeze().cpu().numpy(), cmap='gray'), plt.title(
                '预处理后'), plt.axis(
                'off')
            plt.subplot(133), plt.axis('off'), plt.gca().set_facecolor('#f0f0f0')
            plt.text(0.5, 0.6, f'预测：{pred}', fontsize=30, ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.4, f'置信度：{confidence:.3f}', fontsize=16, ha='center', va='center',
                     transform=plt.gca().transAxes)
            plt.title('ResNet识别结果'), plt.tight_layout(), plt.show()
            return pred


        # -------------------------- 主执行流程 --------------------------
        if __name__ == "__main__":
            TRAIN_MODEL = True
            best_acc = 0.0

            if TRAIN_MODEL:
                print(f"开始训练ResNet（{epochs}轮）...")
                train_losses, test_losses, accs = [], [], []
                for epoch in range(epochs):
                    train_loss = train_epoch()
                    test_loss, acc, f1 = evaluate()
                    scheduler.step()
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    accs.append(acc)
                    print(
                        f'Epoch [{epoch + 1}/{epochs}] | 训练损失：{train_loss:.4f} | 测试损失：{test_loss:.4f} | 准确率：{acc:.4f} | F1：{f1:.4f}')

                plot_metrics(train_losses, test_losses, accs)
                print(f"\n训练完成！最佳准确率：{best_acc:.4f} | 模型保存路径：{MODEL_SAVE_PATH}")
            else:
                if os.path.exists(MODEL_SAVE_PATH):
                    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
                    print(f"加载ResNet模型成功！")
                    _, acc, _ = evaluate()
                    print(f"加载模型的测试准确率：{acc:.4f}")
                else:
                    print("未找到模型文件，请先训练！")
                    exit()

            # 推理
            IMAGE_PATH = r'D:/images/sz5.png'
            if os.path.exists(IMAGE_PATH):
                pred = infer_digit(IMAGE_PATH)
                print(f"\n最终识别结果：{pred}")
            else:
                print(f"图像文件不存在：{IMAGE_PATH}")
                print("建议：将手写数字图像保存到桌面，路径改为：r'C:/Users/池棠/Desktop/test.png'")