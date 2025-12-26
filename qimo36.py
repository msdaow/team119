import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score

# 基本设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据
train_dataset = datasets.ImageFolder('C:/Users/JSJ-XL-1L/Downloads/cifar_10/train', transform)
test_dataset = datasets.ImageFolder('C:/Users/JSJ-XL-1L/Downloads/cifar_10/test',
                                    transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")


# 简单VGG模型
class SimpleVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 初始化
model = SimpleVGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 类别名称
classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']


# 训练函数（仅保留训练逻辑，不记录训练集指标）
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)


# 测试函数（增加F1、精度、召回率计算）
def test():
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []  # 存储所有预测标签
    all_targets = []  # 存储所有真实标签

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(1)
            correct += (pred == target).sum().item()

            # 收集预测和真实标签（转换为numpy用于计算指标）
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算测试集指标
    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / len(test_loader.dataset)

    # 计算F1、精度、召回率（macro平均：对所有类别求平均）
    test_precision = 100 * precision_score(all_targets, all_preds, average='macro')
    test_recall = 100 * recall_score(all_targets, all_preds, average='macro')
    test_f1 = 100 * f1_score(all_targets, all_preds, average='macro')

    return test_loss, test_acc, test_precision, test_recall, test_f1


# 主训练循环
print("开始训练...")
epochs = 15
# 仅记录测试集指标
test_losses, test_accs = [], []
test_precisions, test_recalls, test_f1s = [], [], []

for epoch in range(1, epochs + 1):
    # 训练（仅返回训练损失，不记录）
    train_loss = train(epoch)
    # 测试（获取所有测试集指标）
    test_loss, test_acc, test_precision, test_recall, test_f1 = test()

    # 仅记录测试集指标
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    test_f1s.append(test_f1)

    scheduler.step()
    print(f'Epoch {epoch}: 测试准确率={test_acc:.2f}%, 测试精度={test_precision:.2f}%, '
          f'测试召回率={test_recall:.2f}%, 测试F1={test_f1:.2f}%')

# 绘制测试集指标曲线（包含损失、准确率、F1、精度、召回率）
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 测试损失曲线
ax1.plot(range(1, epochs + 1), test_losses, 'b-', linewidth=2, label='测试损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('测试集损失曲线')
ax1.grid(True, alpha=0.3)

# 测试准确率曲线
ax2.plot(range(1, epochs + 1), test_accs, 'r-', linewidth=2, label='测试准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.set_title('测试集准确率曲线')
ax2.grid(True, alpha=0.3)

# 测试精度和召回率曲线
ax3.plot(range(1, epochs + 1), test_precisions, 'g-', linewidth=2, label='测试精度')
ax3.plot(range(1, epochs + 1), test_recalls, 'orange', linewidth=2, label='测试召回率')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score (%)')
ax3.legend()
ax3.set_title('测试集精度-召回率曲线')
ax3.grid(True, alpha=0.3)

# 测试F1曲线
ax4.plot(range(1, epochs + 1), test_f1s, 'purple', linewidth=2, label='测试F1分数')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('F1 Score (%)')
ax4.legend()
ax4.set_title('测试集F1分数曲线')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_metrics_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 测试10张图片
print("\n测试10张随机图片:")
indices = np.random.choice(len(test_dataset), 10, replace=False)
correct_predictions = 0

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, idx in enumerate(indices):
    image, true_label = test_dataset[idx]

    # 预测
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_label = output.argmax(1).item()

    # 显示图片（反归一化）
    img_display = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_display = np.clip(img_display, 0, 1)

    axes[i // 5, i % 5].imshow(img_display)
    axes[i // 5, i % 5].axis('off')

    true_class = classes[true_label]
    pred_class = classes[pred_label]

    if true_label == pred_label:
        axes[i // 5, i % 5].set_title(f"真:{true_class}\n预:{pred_class} 正确", color='green', fontsize=10)
        correct_predictions += 1
    else:
        axes[i // 5, i % 5].set_title(f"真:{true_class}\n预:{pred_class} 错误", color='red', fontsize=10)

    print(f"图片{i + 1}: 真实={true_class}, 预测={pred_class} ({'正确' if true_label == pred_label else '错误'})")

plt.tight_layout()
plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 结果汇总
print(f"\n测试结果汇总:")
print(f"10张图片正确预测: {correct_predictions}/10")
print(f"10张图片准确率: {correct_predictions / 10 * 100:.1f}%")
print(f"最终测试准确率: {test_accs[-1]:.2f}%")
print(f"最终测试精度: {test_precisions[-1]:.2f}%")
print(f"最终测试召回率: {test_recalls[-1]:.2f}%")
print(f"最终测试F1分数: {test_f1s[-1]:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'vgg_cifar10_model.pth')
print("\n模型已保存为: vgg_cifar10_model.pth")
print("图片已保存为: test_metrics_curves.png 和 test_results.png")
print("训练完成！")