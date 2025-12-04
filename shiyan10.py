import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, models
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


# 数据预处理
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

# 使用预训练的VGG16模型
model = models.vgg16(pretrained=True)

# 冻结特征提取层的参数，只训练分类器
for param in model.features.parameters():
    param.requires_grad = False

# 修改最后的全连接层以适应10分类任务
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器（只优化分类器参数）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

print("开始训练......")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # 打印每个batch的loss
        if (batch_idx + 1) % 10 == 0:  # 每10个batch打印一次
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # 打印每个epoch的平均loss
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{epochs}, 平均训练损失: {avg_loss:.4f}")
    print("-" * 50)

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


# 测试图片路径 - 请替换为您自己的图片路径
img_path1 = r"C:\Users\86131\Downloads\10分类图像mini\62_ship.png"
img_path2 = r"C:\Users\86131\Downloads\10分类图像mini\21_cat.png"
img_path3 = r"C:\Users\86131\Downloads\10分类图像mini\49_airplane.png"

print("\n测试结果：")
print(f"第一张图片预测类别：{predict_image(img_path1)}")
print(f"第二张图片预测类别：{predict_image(img_path2)}")
print(f"第三张图片预测类别：{predict_image(img_path3)}")

# 保存模型
torch.save(model.state_dict(), 'vgg16_10class.pth')
print("\n模型已保存为 'vgg16_10class.pth'")