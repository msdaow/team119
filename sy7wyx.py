import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=trans)
test_ds = datasets.MNIST('./data', train=False, transform=trans)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.seq(x)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.fc_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv_part(x)
        return self.fc_part(x)
def train_test(model, loader, opt, crit, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad() if not is_train else torch.enable_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train: opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            if is_train: loss.backward(); opt.step()
            total_loss += loss.item()
            pred = out.argmax(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc
def main():
    fc = FC().to(device)
    cnn = ConvNet().to(device)
    opt_fc = optim.Adam(fc.parameters(), lr=0.001)
    opt_cnn = optim.Adam(cnn.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    epochs = 10
    res = {'fc_acc': [], 'cnn_acc': []}
    # 训练FC
    print("训练全连接网络...")
    t0 = time.time()
    for e in range(epochs):
        _, _ = train_test(fc, train_loader, opt_fc, crit, is_train=True)
        _, acc = train_test(fc, test_loader, opt_cnn, crit, is_train=False)
        res['fc_acc'].append(acc)
        print(f'FC Epoch {e + 1}: 测试准确率 {acc:.2f}%')
    fc_t = time.time() - t0
    # 训练CNN
    print("\n训练卷积网络...")
    t0 = time.time()
    for e in range(epochs):
        _, _ = train_test(cnn, train_loader, opt_cnn, crit, is_train=True)
        _, acc = train_test(cnn, test_loader, opt_cnn, crit, is_train=False)
        res['cnn_acc'].append(acc)
        print(f'CNN Epoch {e + 1}: 测试准确率 {acc:.2f}%')
    cnn_t = time.time() - t0
    print(f"\n{'=' * 50}\n性能比较:\n{'=' * 50}")
    print(f"全连接: 准确率 {res['fc_acc'][-1]:.2f}%, time {fc_t:.2f}s")
    print(f"卷积网: 准确率 {res['cnn_acc'][-1]:.2f}%, time {cnn_t:.2f}s")
    print(f"准确率提升: {res['cnn_acc'][-1] - res['fc_acc'][-1]:.2f}%")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), res['fc_acc'], 'b-o', label='FC', linewidth=2)
    plt.plot(range(1, epochs + 1), res['cnn_acc'], 'r-s', label='CNN', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Accuracy (%)'), plt.title('MNIST Test Accuracy')
    plt.legend(), plt.grid(0.3), plt.xticks(range(1, epochs + 1)), plt.tight_layout()
    plt.savefig('mnist_acc_compare.png', dpi=300), plt.show()
if __name__ == "__main__":
    main()
