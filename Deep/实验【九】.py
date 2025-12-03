import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 全局配置 --------------------------
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子（保证实验可复现）
torch.manual_seed(42)
# 训练超参数（所有模型共用）
batch_size = 128
epochs = 10
lr = 0.001

# -------------------------- 数据预处理与加载 --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("正在下载MNIST数据集...")
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("数据加载完成！")

# -------------------------- 指标计算函数 --------------------------
def calculate_metrics(true_labels, predictions, num_classes=10):
    accuracy = np.mean(true_labels == predictions)

    precision_per_class = []
    recall_per_class = []

    for i in range(num_classes):
        true_positive = np.sum((predictions == i) & (true_labels == i))
        false_positive = np.sum((predictions == i) & (true_labels != i))
        false_negative = np.sum((predictions != i) & (true_labels == i))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (
        precision_macro + recall_macro) > 0 else 0

    return accuracy, precision_macro, recall_macro, f1_macro

# -------------------------- 模型定义 --------------------------
# 1. GoogleNet
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 路线1，单1×1卷积层（对应c1）
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路线2，1×1卷积层+3×3的卷积（c2是列表：[ch3x3red, ch3x3]）
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 路线3，1×1卷积层+3×3的卷积（适配小尺寸，替代5×5；c3是列表：[ch5x5red, ch5x5]）
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=3, padding=1)

        # 路线4，3×3的最大池化+1×1的卷积（对应c4，就是你原来的pool_proj）
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

# ========== GoogleNet主体 ==========
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        # 初始卷积层（对应老师的b1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28→14
        )
                                         #8=out_channels=c2[0]
        self.inception3a = Inception(16, 8, (8, 16), (4, 8), 8)  # out: 8+16+8+8=40
        self.inception3b = Inception(40, 16, (8, 16), (4, 8), 8)  # out: 16+16+8+8=48
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7

        self.inception4a = Inception(48, 16, (16, 32), (8, 16), 16)  # out:16+32+16+16=80
        self.inception4b = Inception(80, 16, (16, 32), (8, 16), 16)  # out:16+32+16+16=80
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  #7→4

        # 分类层（保留你的适配逻辑）
        self.b5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(80, num_classes)
        )
    def forward(self, x):
        x = self.b1(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.maxpool4(x)
        x = self.b5(x)
        return x

# 2. ResNet（残差网络，适配MNIST，简化版ResNet18）
class BasicBlock(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(BasicBlock, self).__init__()
        self.ReLU = nn.ReLU()

        # 卷积层1：
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,kernel_size=3,padding=1, stride=strides )
        self.bn1 = nn.BatchNorm2d(num_channels) # 卷积1后做BN（加速收敛、防过拟合）

        # 卷积层2：
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=3,padding=1, stride=1 )
        self.bn2 = nn.BatchNorm2d(num_channels) #归一化conv2输出

        # 残差连接：1×1卷积（仅use_1conv=True时启用） # 作用：保证残差相加时，输入x和卷积输出y的通道/尺寸完全匹配
        if use_1conv:#决定是否创建一个 “1×1 卷积层（conv3）”。
            self.conv3 = nn.Conv2d(#原始输入 x
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=strides
            )
        else:
            self.conv3 = None  # 未启用时设为None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:  # 维度不匹配时调整x
            x = self.conv3(x)
        y = self.ReLU(y + x)  # 残差相加， 维度匹配（通道数、宽高
        return y


# ========== ResNet18主体 ==========
class ResNet18(nn.Module):
    def __init__(self, BasicBlock):
        super(ResNet18, self).__init__()
        # 初始卷积层：（适配MNIST调整通道/步长）
        self.b1 = nn.Sequential(
            nn.Conv2d( in_channels=1,out_channels=16,kernel_size=3,  stride=1,   padding=1 ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 保留你适配MNIST的池化
        )
        self.b2 = nn.Sequential(
            BasicBlock(16, 16, use_1conv=False, strides=1),
            BasicBlock(16, 16, use_1conv=False, strides=1)
        )
        self.b3 = nn.Sequential(
            BasicBlock(16, 32, use_1conv=True, strides=2),
            BasicBlock(32, 32, use_1conv=False, strides=1)
        )
        self.b4 = nn.Sequential(
            BasicBlock(32, 64, use_1conv=True, strides=2),
            BasicBlock(64, 64, use_1conv=False, strides=1)
        )
        self.b5 = nn.Sequential(
            BasicBlock(64, 128, use_1conv=True, strides=2),
            BasicBlock(128, 128, use_1conv=False, strides=1)
        )
        #全连接：
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), # 展平：128×1×1 → 128维向量
            nn.Linear(128, 10)  # 对应最终128通道，适配MNIST 10分类
        )

    def forward(self, x):
        x = self.b1(x)# 1×28×28 → 16×14×14
        x = self.b2(x)# 1×28×28 → 16×14×14
        x = self.b3(x) # 16×14×14 → 32×7×7
        x = self.b4(x) # 32×7×7 → 64×4×4
        x = self.b5(x)  # 64×4×4 → 128×2×2
        x = self.b6(x) # 128×2×2 → 10
        return x

# -------------------------- 训练与评估函数（通用） --------------------------
def train_evaluate_model(model_name, model, train_loader, test_loader, criterion, optimizer, epochs):
    print(f"\n========== 开始训练 {model_name} ==========")
    # 存储指标
    train_losses = []
    test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 测试阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        # 计算指标
        accuracy, precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
        test_accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # 打印日志
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    print(f"========== {model_name} 训练完成 ==========\n")
    # 返回指标结果和训练好的模型
    return {
        'model': model,
        'loss': train_losses,
        'accuracy': test_accuracies,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores
    }

# -------------------------- 数字预测函数 --------------------------
def predict_digits(model, model_name, test_dataset, num_samples=5):
    """
    用训练好的模型预测数字
    :param model: 训练好的模型
    :param model_name: 模型名称（用于图表标题）
    :param test_dataset: 测试数据集
    :param num_samples: 要预测的样本数量
    """
    print(f"\n========== 用 {model_name} 预测数字 ==========")
    # 设置模型为评估模式
    model.eval()
    # 随机抽取num_samples个测试样本
    sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)

    # 创建图表展示预测结果
    plt.figure(figsize=(12, 4))
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # 获取单个样本
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0)  # 增加batch维度

            # 模型预测
            output = model(image)
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label.item()

            # 反归一化图像
            image = image.squeeze(0)
            image = image * 0.5 + 0.5
            image = image.numpy().transpose(1, 2, 0)

            # 绘制图像
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'真实标签: {true_label}\n预测标签: {predicted_label}', fontsize=10)
            plt.axis('off')

    plt.suptitle(f'{model_name} 数字预测结果', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 统计预测正确率
    correct = 0
    for idx in sample_indices:
        image, true_label = test_dataset[idx]
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            if pred.item() == true_label:
                correct += 1
    print(f"抽取的{num_samples}个样本中，预测正确{correct}个，正确率：{correct / num_samples:.2f}\n")

# -------------------------- 初始化模型与训练 --------------------------
# 1. GoogleNet
googlenet_model = GoogLeNet()
googlenet_criterion = nn.CrossEntropyLoss()
googlenet_optimizer = optim.Adam(googlenet_model.parameters(), lr=lr)
googlenet_results = train_evaluate_model("GoogleNet", googlenet_model, train_loader, test_loader, googlenet_criterion, googlenet_optimizer, epochs)

# 2. ResNet18
resnet_model = ResNet18(BasicBlock)  # 修复传参错误
resnet_criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=lr)
resnet_results = train_evaluate_model("ResNet18", resnet_model, train_loader, test_loader, resnet_criterion, resnet_optimizer, epochs)

# -------------------------- 结果对比可视化 --------------------------
epoch_range = range(1, epochs + 1)
plt.rcParams['figure.figsize'] = (18, 12)

# 1. Loss曲线对比（2模型）
plt.subplot(2, 3, 1)
plt.plot(epoch_range, googlenet_results['loss'], 'g-', linewidth=2, label='GoogleNet')
plt.plot(epoch_range, resnet_results['loss'], 'y-', linewidth=2, label='ResNet18')
plt.title('训练损失对比（Loss）')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 2. 准确率对比（2模型）
plt.subplot(2, 3, 2)
plt.plot(epoch_range, googlenet_results['accuracy'], 'g-', linewidth=2, label='GoogleNet')
plt.plot(epoch_range, resnet_results['accuracy'], 'y-', linewidth=2, label='ResNet18')
plt.title('测试准确率对比（Accuracy）')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.legend()

# 3. 精度对比（2模型）
plt.subplot(2, 3, 3)
plt.plot(epoch_range, googlenet_results['precision'], 'g-', linewidth=2, label='GoogleNet')
plt.plot(epoch_range, resnet_results['precision'], 'y-', linewidth=2, label='ResNet18')
plt.title('宏平均精度对比（Precision）')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.legend()

# 4. 召回率对比（2模型）
plt.subplot(2, 3, 4)
plt.plot(epoch_range, googlenet_results['recall'], 'g-', linewidth=2, label='GoogleNet')
plt.plot(epoch_range, resnet_results['recall'], 'y-', linewidth=2, label='ResNet18')
plt.title('宏平均召回率对比（Recall）')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.legend()

# 5. F1分数对比（2模型）
plt.subplot(2, 3, 5)
plt.plot(epoch_range, googlenet_results['f1'], 'g-', linewidth=2, label='GoogleNet')
plt.plot(epoch_range, resnet_results['f1'], 'y-', linewidth=2, label='ResNet18')
plt.title('宏平均F1分数对比（F1 Score）')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------- 最终结果汇总表格 --------------------------
print("\n" + "=" * 80)
print("最终结果汇总对比（2模型）")
print("=" * 80)
metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
googlenet_final = [googlenet_results['loss'][-1], googlenet_results['accuracy'][-1], googlenet_results['precision'][-1], googlenet_results['recall'][-1], googlenet_results['f1'][-1]]
resnet_final = [resnet_results['loss'][-1], resnet_results['accuracy'][-1], resnet_results['precision'][-1], resnet_results['recall'][-1], resnet_results['f1'][-1]]

# 打印表格
print(f"{'指标':<15} {'GoogleNet':<15} {'ResNet18':<15}")
print("-" * 80)
for i in range(len(metrics)):
    print(f"{metrics[i]:<15} {googlenet_final[i]:.4f} {'':<7} {resnet_final[i]:.4f}")
print("=" * 80)

# -------------------------- 调用预测函数（2模型） --------------------------
predict_digits(googlenet_results['model'], "GoogleNet", test_dataset, num_samples=5)
predict_digits(resnet_results['model'], "ResNet18", test_dataset, num_samples=5)