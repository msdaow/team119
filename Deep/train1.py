import os
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
import torchvision  # 计算机视觉专用库
import torchvision.transforms as transforms  # 图像预处理模块
import matplotlib.pyplot as plt  # 可视化库
import numpy as np  # 数值计算库
import random  # 随机数生成器
# 仅新增2个必要导入（最小修改）
from sklearn.model_selection import train_test_split
from torch.cuda import seed_all
from torch.utils.data import Subset  # @创建数据集的子集

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 核心参数调整：训练轮次改为50轮
batch_size = 128
epochs = 50
lr = 0.001
save_dir = "./best_model"
os.makedirs(save_dir, exist_ok=True)

# 固定随机种子确保可复现
seed = 42
random.seed(seed)
np.random.seed(seed)  # 2. 固定NumPy随机种子
torch.manual_seed(seed)  # CPU随机种子
if torch.cuda.is_available():  #
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时需加

# -------------------------- 数据预处理与加载 --------------------------
# 优化数据增强顺序，移除冗余Resize
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先随机裁剪（带填充）
    transforms.RandomHorizontalFlip(),  # 再随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

print("正在加载自定义数据集...")
full_train_dataset = torchvision.datasets.ImageFolder(
    root='./cifar_10/train',
    transform=transform
)
test_dataset = torchvision.datasets.ImageFolder(
    root='./cifar_10/test',
    transform=transform
)

# 拆分训练集/验证集
train_indices, val_indices = train_test_split(
    range(len(full_train_dataset)),
    test_size=0.2,
    random_state=42,
    stratify=[full_train_dataset.targets[i] for i in range(len(full_train_dataset))]
)
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# 构建加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(
    f"数据加载完成！训练集样本数：{len(train_dataset)}，验证集样本数：{len(val_dataset)}，测试集样本数：{len(test_dataset)}")


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


# -------------------------- BasicBlock定义 --------------------------
class BasicBlock(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(BasicBlock, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        y = self.ReLU(y + x)
        return y


# -------------------------- ResNet18定义 --------------------------
class ResNet18(nn.Module):
    def __init__(self, BasicBlock):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.b2 = nn.Sequential(BasicBlock(16, 16, use_1conv=False, strides=1),
                                BasicBlock(16, 16, use_1conv=False, strides=1))
        self.b3 = nn.Sequential(BasicBlock(16, 32, use_1conv=True, strides=2),
                                BasicBlock(32, 32, use_1conv=False, strides=1))
        self.b4 = nn.Sequential(BasicBlock(32, 64, use_1conv=True, strides=2),
                                BasicBlock(64, 64, use_1conv=False, strides=1))
        self.b5 = nn.Sequential(BasicBlock(64, 128, use_1conv=True, strides=2),
                                BasicBlock(128, 128, use_1conv=False, strides=1))
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


# -------------------------- 训练与评估函数（新增学习率调度器） --------------------------
def train_evaluate_model(model_name, model, train_loader, criterion, optimizer, epochs, scheduler):
    print(f"\n========== 开始训练 {model_name} ==========")
    train_losses = []  # 训练损失
    val_accuracies = []  # 改：test_accuracies → val_accuracies
    precisions = []  # 验证集精确率
    recalls = []  # 验证集召回率
    f1_scores = []  # 验证集F1
    best_accuracy = 0.0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算验证集指标
        accuracy, precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
        val_accuracies.append(accuracy)  # 改：test_accuracies → val_accuracies
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # 保存最优模型（基于验证集）
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1} 刷新验证集最优准确率: {best_accuracy:.4f}，已保存模型到 {best_model_path}")

        # 学习率调度器更新
        scheduler.step()

        # 打印当前轮次信息（含学习率）
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{epochs}], LR: {current_lr:.6f}, Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # 训练结束后，评估测试集
    print("\n========== 训练完成，评估测试集性能 ==========")
    model.eval()
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_test.extend(predicted.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())
    test_acc, test_prec, test_recall, test_f1 = calculate_metrics(np.array(all_labels_test), np.array(all_preds_test))
    print(
        f"测试集最终结果 - Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # 改：返回值增加测试集指标
    return {
        'model': model,
        'loss': train_losses,
        'val_accuracy': val_accuracies,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores,
        'test_metrics': (test_acc, test_prec, test_recall, test_f1)  # 新增：测试集指标
    }


# -------------------------- 数字预测函数（修改为10个样本，优化布局） --------------------------
def predict_digits(model, model_name, test_dataset, num_samples=10):
    print(f"\n========== 用 {model_name} 预测数字 ==========")
    model.eval()
    sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    # 调整画布大小适配10个样本（2行5列）
    plt.figure(figsize=(15, 6))

    correct = 0

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            output = model(image)
            _, predicted_label = torch.max(output, 1)

            predicted_label = predicted_label.item()

            if predicted_label == true_label:
                correct += 1
            image = image.squeeze(0).cpu() * 0.5 + 0.5

            image = image.numpy().transpose(1, 2, 0)

            # 2行5列布局展示10个样本
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)
            plt.title(f'真实标签: {true_label}\n预测标签: {predicted_label}', fontsize=9)
            plt.axis('off')
    plt.suptitle(f'{model_name} 数字预测结果', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 调整顶部间距避免标题遮挡
    plt.show()
    print(f"抽取的{num_samples}个样本中，预测正确{correct}个，正确率：{correct / num_samples:.2f}\n")


# -------------------------- 主函数 --------------------------
def main():
    # 初始化模型并训练
    resnet_model = ResNet18(BasicBlock).to(device)
    resnet_criterion = nn.CrossEntropyLoss()
    # 优化器添加权重衰减
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=lr, weight_decay=1e-4)
    # 定义学习率调度器（每30轮学习率降为原来的1/10）
    scheduler = optim.lr_scheduler.StepLR(resnet_optimizer, step_size=30, gamma=0.1)

    # 训练函数传入调度器
    resnet_results = train_evaluate_model("ResNet18", resnet_model, train_loader,
                                          resnet_criterion, resnet_optimizer, epochs, scheduler)

    # 结果可视化（改验证集相关）
    epoch_range = range(1, epochs + 1)
    plt.rcParams['figure.figsize'] = (15, 10)

    # 1. 训练损失
    plt.subplot(2, 3, 1)
    plt.plot(epoch_range, resnet_results['loss'], 'y-', linewidth=2, label='ResNet18')
    plt.title('训练损失（Loss）')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 2. 验证集准确率（改：resnet_results['accuracy'] → resnet_results['val_accuracy']）
    plt.subplot(2, 3, 2)
    plt.plot(epoch_range, resnet_results['val_accuracy'], 'y-', linewidth=2, label='ResNet18')
    plt.title('验证集准确率（Val Accuracy）')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()

    # 3. 宏平均精度
    plt.subplot(2, 3, 3)
    plt.plot(epoch_range, resnet_results['precision'], 'y-', linewidth=2, label='ResNet18')
    plt.title('验证集宏平均精度（Precision）')  # 改：加“验证集”
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()

    # 4. 宏平均召回率
    plt.subplot(2, 3, 4)
    plt.plot(epoch_range, resnet_results['recall'], 'y-', linewidth=2, label='ResNet18')
    plt.title('验证集宏平均召回率（Recall）')  # 改：加“验证集”
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()

    # 5. 宏平均F1分数
    plt.subplot(2, 3, 5)
    plt.plot(epoch_range, resnet_results['f1'], 'y-', linewidth=2, label='ResNet18')
    plt.title('验证集宏平均F1分数（F1 Score）')  # 改：加“验证集”
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 最终结果汇总（仅展示测试集指标）
    print("\n" + "=" * 50)
    print("ResNet18 最终测试集指标（核心结果）")
    print("=" * 50)
    # 从返回值中取出测试集指标
    test_acc, test_prec, test_recall, test_f1 = resnet_results['test_metrics']
    # 只展示测试集的4个核心指标
    test_metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score']
    test_final = [test_acc, test_prec, test_recall, test_f1]
    for i in range(len(test_metrics)):
        print(f"{test_metrics[i]:<15} {test_final[i]:.4f}")
    print("=" * 50)

    # 调用预测函数（默认10个样本）
    predict_digits(resnet_results['model'], "ResNet18", test_dataset, num_samples=10)


# -------------------------- 执行主函数 --------------------------
if __name__ == "__main__":
    main()