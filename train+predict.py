import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)
batch_size = 128
epochs = 15
lr = 0.001
save_dir = "./best_model"  # 最优模型保存目录
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

# CIFAR-10类别名称
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿',
               '狗', '青蛙', '马', '船', '卡车']

# -------------------------- 数据预处理与加载 --------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB三通道归一化
])

print("正在加载CIFAR-10数据集...")
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
    range(len(full_train_dataset)),  # 根据索引拆分数据
    test_size=0.2,
    random_state=42,
    stratify=[full_train_dataset.targets[i] for i in range(len(full_train_dataset))]  # 提取所有样本的目标标签
)
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# 构建加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"数据加载完成！训练集样本数: {len(train_dataset)} 验证集样本数: {len(val_dataset)} 测试集样本数: {len(test_dataset)}")
print(f"类别名称：{CLASS_NAMES}")


# -------------------------- 指标计算函数 --------------------------
def calculate_metrics(true_labels, predictions, num_classes=10):
    accuracy = np.mean(true_labels == predictions)#整体准确率
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
    precision_macro = np.mean(precision_per_class)#宏精确率
    recall_macro = np.mean(recall_per_class)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (
                                                                                                precision_macro + recall_macro) > 0 else 0
    return accuracy, precision_macro, recall_macro, f1_macro


# -------------------------- CNN模型定义 --------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 卷积层块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 卷积层块3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 全连接层 将卷积特征映射为类别概率
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),#压缩特征维度
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):#向前传播，从输入到输出·的流程
        x = self.conv1(x)  # [batch, 32, 16, 16] 输出张量的形状
        x = self.conv2(x)  # [batch, 64, 8, 8]
        x = self.conv3(x)  # [batch, 128, 4, 4]
        x = self.fc(x)  # [batch, 10]
        return x


# -------------------------- 训练与评估函数（新增保存最优模型） --------------------------
def train_evaluate_model(model_name, model, train_loader, val_loader, criterion, optimizer, epochs):
    print(f"\n========== 开始训练 {model_name} ==========")
    train_losses = []#记录每轮训练损失
    val_accuracies = []  # 改为验证集准确率
    precisions = []
    recalls = []
    f1_scores = []
    best_accuracy = 0.0  # 记录最优准确率
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)#向前传播
            loss = criterion(outputs, labels)#计算损失
            optimizer.zero_grad()
            loss.backward()#反向传播计算梯度
            optimizer.step()#更新参数
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段（使用验证集）
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)#取得分最高类别作为预测结果
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
        accuracy, precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
        val_accuracies.append(accuracy)  # 改为验证集准确率
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # 保存最优模型（基于验证集准确率）
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1} 刷新最优准确率: {best_accuracy:.4f}，已保存模型到 {best_model_path}")

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    print(f"\n========== {model_name} 训练完成 ==========")
    print(f"最优模型已保存至: {best_model_path}（验证集最优准确率: {best_accuracy:.4f}）")
    return {'model': model, 'loss': train_losses, 'accuracy': val_accuracies,  # 返回验证集准确率
            'precision': precisions, 'recall': recalls, 'f1': f1_scores}


# -------------------------- 类别预测可视化函数 --------------------------
def predict_random_samples(model, model_name, test_dataset, num_samples=10):
    """
    随机选取测试集中的图片进行预测并可视化
    """
    print(f"\n{'=' * 60}")
    print(f"用 {model_name} 随机预测测试集中的图片")
    print('=' * 60)

    model.eval()

    # 简单随机选择：直接随机选择num_samples个索引
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    # 创建可视化
    fig = plt.figure(figsize=(15, 6))
    correct = 0

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            image, true_label = test_dataset[idx]#获取图片和标签
            image_input = image.unsqueeze(0)#增加batch维度
            output = model(image_input)
            _, predicted_label = torch.max(output, 1)#预测类别标签
            predicted_label = predicted_label.item()#从张量转换为标量

            # 检查预测是否正确
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1

            # 准备显示的图像（反归一化）
            display_image = image * 0.5 + 0.5  # 反归一化
            display_image = display_image.numpy().transpose(1, 2, 0)

            # 创建子图
            ax = plt.subplot(2, num_samples // 2, i + 1)
            ax.imshow(display_image)

            # 设置标题颜色：正确为绿色，错误为红色
            title_color = 'green' if is_correct else 'red'
            title = f"真实: {CLASS_NAMES[true_label]}\n预测: {CLASS_NAMES[predicted_label]}"
            ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')
            ax.axis('off')

    plt.suptitle(
        f'{model_name} - 测试集随机预测结果（正确率: {correct}/{num_samples} = {100 * correct / num_samples:.1f}%）',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 打印详细结果
    print(f"\n预测结果汇总：")
    print(f"总样本数: {num_samples}")
    print(f"正确预测: {correct}")
    print(f"准确率: {correct / num_samples:.2%}")
    print('=' * 60)


# -------------------------- 初始化模型并训练 --------------------------
print("\n初始化CNN模型...")
cnn_model = CNN()
print(f"CNN模型结构：")
print(cnn_model)

cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=lr)
cnn_results = train_evaluate_model("CNN", cnn_model, train_loader, val_loader,
                                   cnn_criterion, cnn_optimizer, epochs)

# -------------------------- 结果可视化 --------------------------
epoch_range = range(1, epochs + 1)
plt.rcParams['figure.figsize'] = (15, 10)

# 1. 训练损失
plt.subplot(2, 3, 1)
plt.plot(epoch_range, cnn_results['loss'], 'b-', linewidth=2, label='CNN')
plt.title('训练损失（Loss）')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 2. 验证准确率
plt.subplot(2, 3, 2)
plt.plot(epoch_range, cnn_results['accuracy'], 'b-', linewidth=2, label='CNN')
plt.title('验证准确率（Validation Accuracy）')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()

# 3. 宏平均精度
plt.subplot(2, 3, 3)
plt.plot(epoch_range, cnn_results['precision'], 'b-', linewidth=2, label='CNN')
plt.title('宏平均精度（Precision）')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()

# 4. 宏平均召回率
plt.subplot(2, 3, 4)
plt.plot(epoch_range, cnn_results['recall'], 'b-', linewidth=2, label='CNN')
plt.title('宏平均召回率（Recall）')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()

# 5. 宏平均F1分数
plt.subplot(2, 3, 5)
plt.plot(epoch_range, cnn_results['f1'], 'b-', linewidth=2, label='CNN')
plt.title('宏平均F1分数（F1 Score）')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()

plt.suptitle('CNN模型在CIFAR-10数据集上的训练表现', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------- 最终结果汇总 --------------------------
print("\n" + "=" * 60)
print("CNN模型最终结果汇总（基于验证集）")
print("=" * 60)
metrics = ['Loss', 'Validation Accuracy', 'Precision', 'Recall', 'F1 Score']
cnn_final = [cnn_results['loss'][-1], cnn_results['accuracy'][-1],
             cnn_results['precision'][-1], cnn_results['recall'][-1],
             cnn_results['f1'][-1]]
for i in range(len(metrics)):
    print(f"{metrics[i]:<20} {cnn_final[i]:.4f}")
print("=" * 60)

# -------------------------- 加载最优模型进行随机预测 --------------------------
print("\n加载最优模型进行随机预测（测试集）...")
best_model_path = os.path.join(save_dir, "CNN_best.pth")
if os.path.exists(best_model_path):
    print(f"找到最优模型: {best_model_path}")
    best_model = CNN()
    best_model.load_state_dict(torch.load(best_model_path))

    # 使用最优模型在测试集上进行随机预测
    predict_random_samples(best_model, "CNN最优模型", test_dataset, num_samples=10)
else:
    print("未找到最优模型文件")