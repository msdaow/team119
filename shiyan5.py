import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import os
import glob

class_names = ["frog", "truck", "deer", "automobile", "bird", 'ship', "cat", "dog","airplane", "horse"]
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.img_paths = [p for p in self.img_paths if self._get_class(p) is not None]

    def _get_class(self, img_path):
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        if len(parts) >= 2:
            cls_part = parts[1].split(".")[0]
            return cls_part if cls_part in class_to_idx else None
        return None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        cls = self._get_class(img_path)
        label = class_to_idx[cls]

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()

param_path = r"C:\Users\86131\model_model.pth"
# 修复：添加 weights_only=True
model.load_state_dict(torch.load(param_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
print("模型参数加载完成,准备预测")

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        image = image.to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return class_names[predicted.item()]
    except Exception as e:
        return f"预测失败：{str(e)}"

img_path1 = r"C:\Users\86131\Downloads\10分类图像\6_bird.png"
img_path2 = r"C:\Users\86131\Downloads\10分类图像\21_cat.png"
img_path3 = r"C:\Users\86131\Downloads\10分类图像\28_deer.png"

print(f"\n第一张图片预测类别: {predict_image(img_path1)}")
print(f"第二张图片预测类别: {predict_image(img_path2)}")
print(f"第三张图片预测类别: {predict_image(img_path3)}")

# 修复：如果你不需要重新保存模型，可以删除这行
# torch.save(model.state_dict(), r'C:\Users\86131\model_model.pth')
# print("模型已保存至C盘")

# 修复：使用新的 weights 参数
model_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 10)
model_resnet.to(device)

print("ResNet18模型定义完成")