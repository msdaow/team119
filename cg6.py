import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r"C:\Users\86131\Downloads\countries-1760073842817.csv")


# 数据预处理
def clean_numeric_data(series):
    if series.dtype == object:
        cleaned = series.str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return series


# 选择特征
numeric_columns = [
    'Population (millions)', 'HDI', 'GDP per Capita',
    'Cropland Footprint', 'Grazing Footprint', 'Forest Footprint',
    'Carbon Footprint', 'Fish Footprint', 'Total Ecological Footprint',
    'Cropland', 'Grazing Land', 'Forest Land', 'Fishing Water', 'Urban Land',
    'Total Biocapacity', 'Biocapacity Deficit or Reserve', 'Earths Required'
]

df_clean = df[numeric_columns].copy()

# 清理数据
for col in df_clean.columns:
    df_clean[col] = clean_numeric_data(df_clean[col])
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# 准备数据
X = df_clean.drop(['HDI', 'Total Ecological Footprint'], axis=1)
y = df_clean['HDI'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# 创建数据加载器
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 定义模型
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet(X_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
train_losses = []
test_losses = []

for epoch in range(100):
    # 训练
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 测试
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

    train_losses.append(train_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# 评估
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f'\n最终结果: MAE: {mae:.4f}, R²: {r2:.4f}')

# 绘图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title('Training Progress')

plt.subplot(1, 2, 2)
plt.scatter(y_test, test_predictions, alpha=0.6)
min_val, max_val = min(y_test.min(), test_predictions.min()), max(y_test.max(), test_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predictions vs Actual')

plt.tight_layout()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'country_model.pth')
print("模型已保存")