import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from collections import Counter

# ---------------------- 1. 全局配置 ----------------------
# 设备配置（优先GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据路径（烂番茄数据集）
TRAIN_PATH = 'train.tsv'
TEST_PATH = 'test.tsv'
# 模型参数
EMBEDDING_DIM = 100  # 词嵌入维度
HIDDEN_DIM = 128  # LSTM隐藏层维度
NUM_LAYERS = 2  # LSTM层数
DROPOUT = 0.5  # Dropout率
BATCH_SIZE = 64  # 批次大小
EPOCHS = 10  # 训练轮数
LEARNING_RATE = 0.001  # 学习率
MAX_VOCAB_SIZE = 10000  # 词汇表最大大小
MAX_SEQ_LEN = 50  # 句子最大长度（截断/补齐）
PAD_TOKEN = '<PAD>'  # 补齐标记
UNK_TOKEN = '<UNK>'  # 未知词标记


# 固定随机种子（保证可复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()


# ---------------------- 2. 数据预处理（适配烂番茄数据集） ----------------------
# 加载数据集（区分训练集/测试集）
def load_data(file_path, is_train=True):
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    # 提取评论文本（Phrase列）
    texts = df['Phrase'].dropna().tolist()
    if is_train:
        # 训练集：提取Sentiment标签（0-4）
        labels = df['Sentiment'].dropna().astype(int).tolist()
        return texts, labels, df  # 返回df用于后续随机预测
    else:
        # 测试集：无Sentiment标签，仅返回文本和原始df
        return texts, df


# 构建词汇表（仅基于训练集）
def build_vocab(texts, max_vocab_size):
    word_counter = Counter()
    for text in texts:
        words = text.lower().split()  # 小写+空格分词
        word_counter.update(words)
    # 构建词汇表（保留高频词，PAD/UNK在前）
    vocab = [PAD_TOKEN, UNK_TOKEN] + [word for word, _ in word_counter.most_common(max_vocab_size - 2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word


# 文本转索引（截断/补齐到固定长度）
def text2idx(text, word2idx, max_seq_len):
    words = text.lower().split()
    # 转换为索引（未知词用UNK）
    idx = [word2idx.get(word, word2idx[UNK_TOKEN]) for word in words]
    # 截断/补齐
    if len(idx) > max_seq_len:
        idx = idx[:max_seq_len]
    else:
        idx += [word2idx[PAD_TOKEN]] * (max_seq_len - len(idx))
    return idx


# 自定义数据集类
class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_idx = text2idx(text, self.word2idx, self.max_seq_len)
        return torch.tensor(text_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 加载数据
train_texts, train_labels, train_df = load_data(TRAIN_PATH, is_train=True)
test_texts, test_df = load_data(TEST_PATH, is_train=False)

# 构建词汇表
vocab, word2idx, idx2word = build_vocab(train_texts, MAX_VOCAB_SIZE)
VOCAB_SIZE = len(vocab)
NUM_CLASSES = len(set(train_labels))  # 情感类别数（0-4共5类）

# 构建训练集DataLoader（测试集无标签，仅用训练集拆分验证）
# 从训练集中拆分10%作为验证集（替代无标签的test.tsv）
val_split = 0.1
val_size = int(len(train_texts) * val_split)
train_texts_train = train_texts[val_size:]
train_labels_train = train_labels[val_size:]
val_texts = train_texts[:val_size]
val_labels = train_labels[:val_size]

# 创建数据集和加载器
train_dataset = MovieReviewDataset(train_texts_train, train_labels_train, word2idx, MAX_SEQ_LEN)
val_dataset = MovieReviewDataset(val_texts, val_labels, word2idx, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------- 3. 定义TextRNN模型 ----------------------
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(TextRNN, self).__init__()
        # 词嵌入层（padding_idx指定PAD_TOKEN的索引）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx[PAD_TOKEN])
        # 双向LSTM（batch_first=True：输入格式为[batch, seq_len, feature]）
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 全连接层（双向LSTM输出维度=hidden_dim*2）
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, max_seq_len]
        embed = self.embedding(x)  # [batch_size, max_seq_len, embedding_dim]
        # LSTM输出：output=[batch, seq_len, 2*hidden_dim], (h_n, c_n)
        output, (h_n, c_n) = self.lstm(embed)
        # 拼接双向LSTM最后一层的前向/后向隐藏状态
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch_size, 2*hidden_dim]
        h_n = self.dropout(h_n)
        out = self.fc(h_n)  # [batch_size, num_classes]
        return out


# 初始化模型、损失函数、优化器
model = TextRNN(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()  # 分类任务交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------- 4. 模型训练（显示每轮损失/准确率） ----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_acc = 0.0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts = texts.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失和预测结果
            train_loss += loss.item() * texts.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # 计算训练集指标
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(device)
                labels = labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * texts.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # 计算验证集指标
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_textrnn_model.pth')

        # 打印每轮结果
        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'训练损失: {train_avg_loss:.4f}, 训练准确率: {train_acc:.4f}')
        print(f'验证损失: {val_avg_loss:.4f}, 验证准确率: {val_acc:.4f}')
        print('-' * 60)


# 启动训练
print("开始训练TextRNN模型（适配烂番茄数据集）...")
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)


# ---------------------- 5. 随机选择5条训练集数据预测（含真实标签） ----------------------
def predict_single_text(text, model, word2idx, max_seq_len):
    """单条文本预测函数"""
    model.eval()
    with torch.no_grad():
        text_idx = text2idx(text, word2idx, max_seq_len)
        text_tensor = torch.tensor(text_idx, dtype=torch.long).unsqueeze(0).to(device)
        output = model(text_tensor)
        _, pred = torch.max(output, 1)
        return pred.item()


# 情感标签映射（烂番茄数据集标准映射）
sentiment_map = {0: '消极', 1: '略带消极', 2: '中性', 3: '略带积极', 4: '积极'}

# 从训练集中随机选5条（有真实标签）
random_samples = train_df.sample(n=5, random_state=42)

print("\n随机5条评论预测结果（训练集）：")
print('=' * 80)
for idx, row in random_samples.iterrows():
    text = row['Phrase']
    true_label = row['Sentiment']
    pred_label = predict_single_text(text, model, word2idx, MAX_SEQ_LEN)
    # 输出结果（文本过长时截断）
    display_text = text[:100] + '...' if len(text) > 100 else text
    print(f"文本：{display_text}")
    print(f"真实标签：{true_label} ({sentiment_map[true_label]})")
    print(f"预测标签：{pred_label} ({sentiment_map[pred_label]})")
    print('-' * 80)

# （可选）对测试集随机5条预测（无真实标签）
print("\n随机5条评论预测结果（测试集，无真实标签）：")
print('=' * 80)
test_random_samples = test_df.sample(n=5, random_state=42)
for idx, row in test_random_samples.iterrows():
    text = row['Phrase']
    pred_label = predict_single_text(text, model, word2idx, MAX_SEQ_LEN)
    display_text = text[:100] + '...' if len(text) > 100 else text
    print(f"文本：{display_text}")
    print(f"预测标签：{pred_label} ({sentiment_map[pred_label]})")
    print('-' * 80)