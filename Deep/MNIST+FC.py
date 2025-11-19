import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理（与CNN相同）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集（与CNN相同）
print("正在加载MNIST数据集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 数据加载器（与CNN相同）
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义全连接网络（参数与CNN保持一致）
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 保持与CNN相似的参数规模
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层
        self.fc2 = nn.Linear(512, 256)  # 隐藏层1
        self.fc3 = nn.Linear(256, 128)  # 隐藏层2（与CNN的fc1相同）
        self.fc4 = nn.Linear(128, 10)  # 输出层（与CNN相同）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 与CNN相同的dropout率

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# 创建FC模型
fc_model = FCNet().to(device)
print("FC模型创建完成")

# 打印模型结构
print("\nFC模型结构:")
print(fc_model)

# 计算参数量
fc_total_params = sum(p.numel() for p in fc_model.parameters())
print(f"FC模型总参数量: {fc_total_params:,}")

# 损失函数和优化器（与CNN相同）
criterion = nn.CrossEntropyLoss()
fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.001)  # 相同学习率

# 记录训练过程
fc_train_losses = []
fc_train_accuracies = []
fc_test_accuracies = []
fc_training_time = 0


# FC训练函数
def train_fc(epoch):
    fc_model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        fc_optimizer.zero_grad()
        output = fc_model(data)
        loss = criterion(output, target)
        loss.backward()
        fc_optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'FC - Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    fc_train_losses.append(avg_loss)
    fc_train_accuracies.append(accuracy)

    print(f'FC - Epoch {epoch}: 训练损失: {avg_loss:.4f}, 训练准确率: {accuracy:.2f}%')


# FC测试函数
def test_fc():
    fc_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = fc_model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    fc_test_accuracies.append(accuracy)

    print(f'FC - 测试准确率: {accuracy:.2f}%')
    return accuracy


# 开始FC训练
print("\n开始FC网络训练...")
epochs = 5

start_time = time.time()
for epoch in range(1, epochs + 1):
    print(f"\n=== FC Epoch {epoch}/{epochs} ===")
    train_fc(epoch)
    test_fc()
    print('-' * 50)

fc_training_time = time.time() - start_time

# 绘制FC训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(fc_train_losses, 'g-', label='FC训练损失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('FC训练损失曲线')

plt.subplot(1, 2, 2)
plt.plot(fc_train_accuracies, 'g-', label='FC训练准确率', linewidth=2)
plt.plot(fc_test_accuracies, 'orange', label='FC测试准确率', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.title('FC准确率曲线')

plt.tight_layout()
plt.savefig('fc_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFC网络最终结果:")
print(f"最终训练准确率: {fc_train_accuracies[-1]:.2f}%")
print(f"最终测试准确率: {fc_test_accuracies[-1]:.2f}%")
print(f"训练时间: {fc_training_time:.2f}秒")
print(f"总参数量: {fc_total_params:,}")

# 保存FC模型
torch.save(fc_model.state_dict(), 'mnist_fc_model.pth')
print("FC模型已保存为: mnist_fc_model.pth")


# 性能比较函数（如果需要与CNN比较）
def compare_performance(cnn_results=None):
    print("\n" + "=" * 60)
    print("性能比较分析")
    print("=" * 60)

    print(f"{'指标':<15} {'FC网络':<15} {'CNN网络':<15}")
    print(
        f"{'测试准确率':<15} {fc_test_accuracies[-1]:<15.2f} {cnn_results['test_accuracy'] if cnn_results else 'N/A':<15}")
    print(f"{'训练时间(s)':<15} {fc_training_time:<15.2f} {cnn_results['training_time'] if cnn_results else 'N/A':<15}")
    print(f"{'参数量':<15} {fc_total_params:<15,} {cnn_results['total_params'] if cnn_results else 'N/A':<15}")

    if fc_test_accuracies[-1] > (cnn_results['test_accuracy'] if cnn_results else 0):
        print("\n结论: FC网络性能更优")
    elif cnn_results:
        print("\n结论: CNN网络性能更优")
    else:
        print("\n结论: 等待CNN结果进行比较")


# 调用比较函数（在运行CNN后调用）
# compare_performance(cnn_results)

# 可视化一些FC网络的预测结果
def visualize_fc_predictions(model, test_loader, num_samples=12):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, 1)

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    confidence = confidence.cpu().numpy()

    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        color = 'green' if labels[i] == predicted[i] else 'red'
        plt.title(f'True: {labels[i]}, Pred: {predicted[i]}\nConf: {confidence[i]:.3f}',
                  color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle('FC网络在测试集上的预测结果', fontsize=16)
    plt.tight_layout()
    plt.show()


print("\n显示FC网络预测结果...")
visualize_fc_predictions(fc_model, test_loader)