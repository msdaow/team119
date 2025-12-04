import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
import matplotlib.pyplot as plt
import time
import zipfile

# ---------------------- 配置参数（不变）----------------------

# 配置支持中文的字体（Windows系统自带，无需额外安装）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# 避免负号显示为方块（如损失下降时的负斜率）
plt.rcParams['axes.unicode_minus'] = False

HIDDEN_SIZE = 64
BATCH_SIZE = 128
N_LAYER = 1
N_EPOCHS = 5
LR = 0.001
USE_GPU = torch.cuda.is_available()
MAX_SEQ_LEN = 30
N_CLASS = 5

# 数据集路径
TRAIN_ZIP_PATH = r"F:\MobileFile\images\images\train.tsv.zip"
TEST_ZIP_PATH = r"F:\MobileFile\images\images\test.tsv.zip"


# ---------------------- 修复数据预处理（添加空值删除）----------------------
class MovieReviewDataset(Dataset):
    def __init__(self, is_train_set=True):
        zip_path = TRAIN_ZIP_PATH if is_train_set else TEST_ZIP_PATH
        with zipfile.ZipFile(zip_path, 'r') as zf:
            tsv_name = zf.namelist()[0]
            with zf.open(tsv_name, 'r') as f:
                if is_train_set:
                    # 读取训练集 + 删除Phrase为空的行
                    self.df = pd.read_csv(f, sep='\t', usecols=['Phrase', 'Sentiment'])
                    self.df = self.df.dropna(subset=['Phrase'])  # 关键：删除Phrase列空值
                else:
                    # 读取测试集 + 删除Phrase为空的行
                    self.df = pd.read_csv(f, sep='\t', usecols=['Phrase', 'PhraseId'])
                    self.df = self.df.dropna(subset=['Phrase'])  # 关键：删除Phrase列空值

        self.is_train = is_train_set
        self.texts = self.df['Phrase'].tolist()  # 此时texts全是字符串，无float
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)

    def build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in self.texts:
            # 额外加一层判断：确保text是字符串（双重保险）
            if isinstance(text, str):
                for c in text.lower():
                    if c not in vocab:
                        vocab[c] = len(vocab)
        return vocab

    def text2tensor(self, text):
        # 确保text是字符串（避免极端情况）
        if not isinstance(text, str):
            text = ""
        tensor = [self.vocab.get(c.lower(), 1) for c in text]
        if len(tensor) > MAX_SEQ_LEN:
            tensor = tensor[:MAX_SEQ_LEN]
        else:
            tensor += [0] * (MAX_SEQ_LEN - len(tensor))
        return torch.LongTensor(tensor)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tensor = self.text2tensor(text)
        if self.is_train:
            return tensor, self.df.iloc[idx]['Sentiment']
        else:
            return tensor, self.df.iloc[idx]['PhraseId']

    def __len__(self):
        return len(self.texts)


# 加载数据（现在不会有float类型文本）
train_data = MovieReviewDataset(is_train_set=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = MovieReviewDataset(is_train_set=False)  # 此处不再报错
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------- 模型、训练、测试逻辑（不变）----------------------
class SimpleTextRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, HIDDEN_SIZE)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, N_CLASS)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, HIDDEN_SIZE)
        return hidden.cuda() if USE_GPU else hidden

    def forward(self, x):
        seq_len = torch.sum(x != 0, dim=1)
        embed = self.emb(x)
        packed = pack_padded_sequence(embed, seq_len, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed, self.init_hidden(x.size(0)))
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.fc(hidden_cat)


# 初始化
model = SimpleTextRNN(train_data.vocab_size)
if USE_GPU:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 训练函数
def train():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        if USE_GPU:
            x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in train_loader:
            if USE_GPU:
                x, y = x.cuda(), y.cuda()
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# 主循环
start = time.time()
losses = []
accs = []
print(f"开始训练（{N_EPOCHS}轮），GPU: {'可用' if USE_GPU else '不可用'}")
for epoch in range(1, N_EPOCHS + 1):
    loss = train()
    acc = test()
    losses.append(loss)
    accs.append(acc)
    print(f"轮次{epoch:2d} | 耗时{time.time() - start:.1f}s | 损失{loss:.4f} | 准确率{acc:.1f}%")

# 修改1：将准确率图像改为损失值图像
plt.figure(figsize=(8, 4))
plt.plot(losses, label='训练损失', color='red')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.title('训练损失变化曲线')
plt.legend()
plt.grid()
plt.show()


# 修改2：添加预测功能 - 对单个文本进行预测
def predict_single_text(text):
    """
    对单个文本进行情感预测
    """
    model.eval()
    # 将文本转换为tensor
    tensor = train_data.text2tensor(text).unsqueeze(0)  # 添加batch维度

    if USE_GPU:
        tensor = tensor.cuda()

    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    # 情感标签映射
    sentiment_labels = {
        0: "非常负面",
        1: "负面",
        2: "中性",
        3: "正面",
        4: "非常正面"
    }

    print(f"文本: '{text}'")
    print(f"预测情感: {sentiment_labels[prediction]}")
    print("各类别概率:")
    for i, prob in enumerate(probabilities):
        print(f"  {sentiment_labels[i]}: {prob:.4f}")
    print("-" * 50)

    return prediction, probabilities


# 预测测试集
def predict_test():
    model.eval()
    preds = []
    phrase_ids = []
    with torch.no_grad():
        for x, pid in test_loader:
            if USE_GPU:
                x = x.cuda()
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            phrase_ids.extend(pid.numpy())
    pd.DataFrame({
        'PhraseId': phrase_ids,
        'Sentiment': preds
    }).to_csv('submission_simple.csv', index=False)
    print("预测完成！提交文件：submission_simple.csv")


predict_test()

# 演示单个文本预测
print("\n=== 单个文本预测演示 ===")
test_texts = [
    "This movie is absolutely fantastic!",
    "I hate this film, it's terrible.",
    "The movie is okay, nothing special.",
    "Amazing performance by the actors!",
    "Boring and waste of time."
]

for text in test_texts:
    predict_single_text(text)

# 可选：绘制损失和准确率的双图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, label='训练损失', color='red')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.title('训练损失变化')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(accs, label='训练准确率', color='blue')
plt.xlabel('轮次')
plt.ylabel('准确率(%)')
plt.title('训练准确率变化')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()