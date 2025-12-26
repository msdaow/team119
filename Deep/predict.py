import os
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torchvision  # 计算机视觉专用库
import torchvision.transforms as transforms  # 图像预处理模块
import matplotlib.pyplot as plt  # 可视化库
import numpy as np  # 数值计算库
import random  # 随机数生成器

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


# -------------------------- 指标计算函数（新增：评估测试集全量指标） --------------------------
def calculate_metrics(true_labels, predictions, num_classes=10):
    """
    计算分类任务的核心指标：准确率、宏平均精确率、宏平均召回率、宏平均F1分数
    """
    accuracy = np.mean(true_labels == predictions)
    precision_per_class = []
    recall_per_class = []

    for i in range(num_classes):
        true_positive = np.sum((predictions == i) & (true_labels == i))
        false_positive = np.sum((predictions == i) & (true_labels != i))
        false_negative = np.sum((predictions != i) & (true_labels == i))

        # 计算每个类别的精确率和召回率（避免除零）
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    # 计算宏平均指标
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (
                                                                                                precision_macro + recall_macro) > 0 else 0

    return accuracy, precision_macro, recall_macro, f1_macro


# -------------------------- 测试集全量评估函数（新增） --------------------------
def evaluate_test_set(model, test_loader, device, num_classes=10):
    """
    评估模型在整个测试集上的性能指标
    """
    print("\n========== 开始评估模型在测试集上的性能 ==========")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 收集预测结果和真实标签（转CPU+numpy）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    test_acc, test_prec, test_recall, test_f1 = calculate_metrics(
        np.array(all_labels), np.array(all_preds), num_classes
    )

    # 打印全量测试集指标
    print("=" * 60)
    print("模型在测试集上的全量指标")
    print("=" * 60)
    print(f"测试集准确率 (Accuracy): {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"宏平均精确率 (Precision): {test_prec:.4f}")
    print(f"宏平均召回率 (Recall): {test_recall:.4f}")
    print(f"宏平均F1分数 (F1 Score): {test_f1:.4f}")
    print("=" * 60)

    return test_acc, test_prec, test_recall, test_f1


# -------------------------- BasicBlock定义（模型结构必须保留） --------------------------
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


# -------------------------- ResNet18定义（模型结构必须保留） --------------------------
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


# -------------------------- 数字预测函数（核心保留并优化） --------------------------
def predict_digits(model, model_name, test_dataset, num_samples=10, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n========== 用 {model_name} 随机预测{num_samples}个样本 ==========")
    model.eval()  # 切换评估模式
    # 随机抽取指定数量的样本索引（不重复）
    sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    # 调整画布大小适配10个样本（2行5列）
    plt.figure(figsize=(15, 6))

    correct = 0  # 统计预测正确的样本数

    with torch.no_grad():  # 禁用梯度计算
        for i, idx in enumerate(sample_indices):
            # 获取样本（图像+真实标签）
            image, true_label = test_dataset[idx]
            # 增加batch维度（模型输入要求）+ 移到指定设备
            image = image.unsqueeze(0).to(device)

            # 前向传播：预测
            output = model(image)
            # 取预测概率最大的类别
            _, predicted_label = torch.max(output, 1)
            # 转换为普通Python整数
            predicted_label = predicted_label.item()

            # 统计正确数
            if predicted_label == true_label:
                correct += 1
            # 图像后处理：移回CPU + 去掉batch维度 + 反归一化
            image = image.squeeze(0).cpu() * 0.5 + 0.5
            # 维度转换：(C, H, W) → (H, W, C)（适配matplotlib显示）
            image = image.numpy().transpose(1, 2, 0)

            # 2行5列布局展示10个样本
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)  # 显示图像（RGB图无需cmap）
            plt.title(f'真实标签: {true_label}\n预测标签: {predicted_label}', fontsize=9)
            plt.axis('off')  # 关闭坐标轴

    # 设置总标题
    plt.suptitle(f'{model_name} 随机{num_samples}个样本预测结果', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 调整顶部间距避免标题遮挡
    plt.show()  # 显示图像
    # 打印预测正确率
    acc_rate = correct / num_samples
    print(f"随机抽取的{num_samples}个样本中，预测正确{correct}个，正确率：{acc_rate:.4f} ({acc_rate * 100:.2f}%)\n")


# -------------------------- 主函数（新增测试集评估步骤） --------------------------
def main():
    # 1. 基础配置
    model_path = "./best_model/ResNet18_best.pth"  # 已训练好的模型路径
    num_predict_samples = 10  # 预测样本数（10个）
    batch_size = 128  # 测试集加载批次大小（和训练时一致）
    save_dir = "./best_model"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 3. 数据预处理（必须和训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载测试集（仅加载，不训练）
    print("正在加载测试集...")
    test_dataset = torchvision.datasets.ImageFolder(
        root='./cifar_10/test',
        transform=transform
    )
    # 创建测试集加载器（用于全量评估）
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    print(f"测试集加载完成！样本数：{len(test_dataset)}")

    # 4. 加载已训练好的模型
    print(f"\n正在加载预训练模型：{model_path}")
    # 初始化模型结构
    resnet_model = ResNet18(BasicBlock).to(device)
    # 加载模型权重
    if os.path.exists(model_path):
        resnet_model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    else:
        raise FileNotFoundError(f"模型文件不存在：{model_path}，请先训练模型或检查路径")

    # 5. 评估模型在整个测试集上的指标（新增核心步骤）
    test_acc, test_prec, test_recall, test_f1 = evaluate_test_set(
        resnet_model, test_loader, device, num_classes=10
    )

    # 6. 执行随机样本预测
    predict_digits(resnet_model, "ResNet18", test_dataset, num_samples=num_predict_samples, device=device)


# -------------------------- 执行主函数 --------------------------
if __name__ == "__main__":
    main()