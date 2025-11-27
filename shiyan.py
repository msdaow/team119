import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import glob

class_names = ['frog', 'truck', 'deer', 'automobile', 'bird',
               'ship', 'cat', 'dog', 'airplane', 'horse']
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.img_paths = [p for p in self.img_paths if self._get_class(p) is not None]

    def _get_class(self, img_path):
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            cls_part = parts[1].split('.')[0]
            return cls_part if cls_part in class_to_idx else None
        return None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        cls = self._get_class(img_path)
        label = class_to_idx[cls]

        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
img_dir = r"C:\Users\86131\Downloads\10分类图像mini"
dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 手动构建完整的VGG16网络
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 由于输入是32x32，经过5次池化后变为1x1
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


# 创建VGG16模型
model= VGG16()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("开始训练......")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

            # 反向传播和优化
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {avg_loss:.4f}")
print("训练完成！")


# 测试函数
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # 增加batch维度
        image = image.to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted_idx = torch.max(output, 1)
        return class_names[predicted_idx.item()]
    except Exception as e:
        return f"预测失败：{str(e)}"


# 测试图片路径
img_path1 = r"C:\Users\86131\Downloads\10分类图像mini\62_ship.png"
img_path2 = r"C:\Users\86131\Downloads\10分类图像mini\21_cat.png"
img_path3 = r"C:\Users\86131\Downloads\10分类图像mini\49_airplane.png"

print(f"\n第一张图片预测类别：{predict_image(img_path1)}")
print(f"第二张图片预测类别：{predict_image(img_path2)}")
print(f"第三张图片预测类别：{predict_image(img_path3)}")