# ---------------------- 第一步：优先解决OpenMP冲突（必须放在最开头） ----------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------- 第二步：导入其他依赖库 ----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt

# ---------------------- 关键：设置Matplotlib中文显示（解决乱码） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 设备设置 + 超参数定义 ----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")
batch_size = 64
learning_rate = 0.01
momentum = 0.9
epochs = 10  # 训练轮数，决定准确率曲线的点数

# ---------------------- 2. 数据准备（MNIST数据集加载） ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化，提升训练稳定性
])

# 训练集/测试集加载
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# ---------------------- 3. 模型定义（FC全连接 + CNN卷积） ----------------------
class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平为(batch, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)  # 20*4*4=320

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))  # (batch,1,28,28)→(batch,10,12,12)
        x = self.pooling(F.relu(self.conv2(x)))  # (batch,10,12,12)→(batch,20,4,4)
        x = x.view(batch_size, -1)  # 展平为(batch, 320)
        x = self.fc(x)
        return x

# ---------------------- 4. 准确率计算工具函数（通用） ----------------------
def calculate_accuracy(model, data_loader, device):
    model.eval()  # 切换到测试模式（禁用训练特有的层，如Dropout）
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省内存+加速
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)  # 取概率最大的类别
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total  # 转为百分比
    model.train()
    return accuracy

# ---------------------- 5. 训练函数（记录每轮训练/测试准确率） ----------------------
def train_model_with_metrics(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    model.train()
    # 仅保留准确率记录列表
    train_accs = []      # 每轮的训练准确率
    test_accs = []       # 每轮的测试准确率
    start_total = time.time()

    for epoch in range(epochs):
        running_loss = 0.0  # 保留损失计算（用于中间打印，不影响曲线）
        start_epoch = time.time()

        # 1. 本轮训练
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # 梯度清零→前向传播→损失计算→反向传播→参数更新
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # 累计损失
            running_loss += loss.item()

            # 每300批打印一次中间损失
            if batch_idx % 300 == 299:
                avg_batch_loss = running_loss / 300
                print(f"[{model.__class__.__name__}] 第{epoch+1}轮, 第{batch_idx+1}批 | 损失: {avg_batch_loss:.3f}")
                running_loss = 0.0

        # 2. 计算并记录本轮准确率
        train_acc = calculate_accuracy(model, train_loader, device)  # 训练准确率
        test_acc = calculate_accuracy(model, test_loader, device)    # 测试准确率
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 3. 打印本轮结果
        epoch_time = time.time() - start_epoch
        print(f"\n[{model.__class__.__name__}] 第{epoch+1}轮训练完成 | "
              f"耗时: {epoch_time:.2f}s | "
              f"训练准确率: {train_acc:.2f}% | "
              f"测试准确率: {test_acc:.2f}%\n")

    # 4. 训练结束，打印总耗时
    total_time = time.time() - start_total
    print(f"[{model.__class__.__name__}] 全部训练完成 | 总耗时: {total_time:.2f}s\n")
    return train_accs, test_accs, total_time

# ---------------------- 6. 主程序（仅绘制准确率曲线） ----------------------
if __name__ == "__main__":
    # ---------------------- 初始化模型、损失函数、优化器 ----------------------
    # FC模型
    fc_model = FC_Net().to(device)
    fc_criterion = nn.CrossEntropyLoss()
    fc_optimizer = optim.SGD(fc_model.parameters(), lr=learning_rate, momentum=momentum)

    # CNN模型
    cnn_model = CNN_Net().to(device)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=momentum)

    # ---------------------- 训练模型（仅记录准确率） ----------------------
    print("="*80)
    print("开始训练 FC 全连接模型")
    print("="*80)
    # 调用训练函数（仅接收准确率数据）
    fc_train_accs, fc_test_accs, fc_total_time = train_model_with_metrics(
        model=fc_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=fc_criterion,
        optimizer=fc_optimizer,
        epochs=epochs,
        device=device
    )

    print("="*80)
    print("开始训练 CNN 卷积模型")
    print("="*80)
    cnn_train_accs, cnn_test_accs, cnn_total_time = train_model_with_metrics(
        model=cnn_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=cnn_criterion,
        optimizer=cnn_optimizer,
        epochs=epochs,
        device=device
    )

    # ---------------------- 性能对比总结 ----------------------
    print("="*80)
    print("FC 与 CNN 模型最终性能对比总结")
    print("="*80)
    print(f"{'模型类型':<12} {'最终训练准确率':<15} {'最终测试准确率':<15} {'总训练时间':<15}")
    print(f"{'-'*80}")
    print(f"{'FC 全连接':<12} {fc_train_accs[-1]:<15.2f}% {fc_test_accs[-1]:<15.2f}% {fc_total_time:<15.2f}s")
    print(f"{'CNN 卷积':<12} {cnn_train_accs[-1]:<15.2f}% {cnn_test_accs[-1]:<15.2f}% {cnn_total_time:<15.2f}s")
    print("="*80)

    # ---------------------- 核心：仅绘制准确率曲线 ----------------------
    plt.figure(figsize=(12, 6))  # 调整画布大小（适合单个曲线）

    # FC模型准确率曲线（虚线=训练，实线=测试，红色系）
    plt.plot(range(1, epochs+1), fc_train_accs,
             label="FC 训练准确率", color="#FF6B6B", marker="o", linestyle="--", linewidth=2)
    plt.plot(range(1, epochs+1), fc_test_accs,
             label="FC 测试准确率", color="#FF6B6B", marker="o", linewidth=2)

    # CNN模型准确率曲线（虚线=训练，实线=测试，青色系）
    plt.plot(range(1, epochs+1), cnn_train_accs,
             label="CNN 训练准确率", color="#4ECDC4", marker="s", linestyle="--", linewidth=2)
    plt.plot(range(1, epochs+1), cnn_test_accs,
             label="CNN 测试准确率", color="#4ECDC4", marker="s", linewidth=2)

    # 曲线美化与标注
    plt.xlabel("训练轮数（Epoch）", fontsize=12)
    plt.ylabel("准确率（%）", fontsize=12)
    plt.title("FC 全连接网络 vs CNN 卷积网络 准确率对比（MNIST任务）", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="lower right")  # 图例放在右下角（不遮挡曲线）
    plt.grid(alpha=0.3)  # 浅色网格提升可读性
    plt.xticks(range(1, epochs+1))  # x轴刻度=训练轮数（1~10）
    plt.ylim(90, 100)  # 限定y轴范围（MNIST准确率集中在90%+）
    plt.tight_layout()  # 自动调整布局，避免标签截断


    # 显示图片
    plt.show()