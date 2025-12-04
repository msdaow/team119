import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image  # 用于读取和处理手写图片

# ====================== 1. 数据预处理与加载 ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root=r"C:",  # 数据集存储路径（自动下载）
    train=True,
    transform=transform,
    download=False
)
test_dataset = datasets.MNIST(
    root=r"C:",
    train=False,
    transform=transform,
    download=False
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# ====================== 2. 定义模型 ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ====================== 3. 模型训练 ======================
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# ====================== 4. 测试集精确度 ======================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"测试集精确度: {accuracy:.2f}%")


# ====================== 5. 手写数字预测 ======================
def predict_handwritten_digit(image_path, model):
    """预测手写数字的函数"""
    # 1. 读取图片并转为灰度图
    img = Image.open(image_path).convert('L')
    # 2. 调整尺寸为28×28（与MNIST一致）
    img = img.resize((28, 28))
    # 3. 应用与训练数据相同的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    # 4. 模型预测
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# 替换为你手写数字图片的实际路径
handwritten_image_path = r"C:\Users\86131\Pictures\Screenshots\shuzi.jpg"  # 示例路径，需根据实际修改
predicted_digit = predict_handwritten_digit(handwritten_image_path, model)
print(f"手写数字的预测结果: {predicted_digit}")