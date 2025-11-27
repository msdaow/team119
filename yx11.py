# 解决OpenMP库冲突（放在代码最顶部）
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
import re
import matplotlib.pyplot as plt

# ---------------------- 1. 配置参数与路径（固定为你的路径） ----------------------
DATA_PATHS = {
    "train": "D:/train.tsv",
    "test": "D:/test.tsv",
    "submission": "D:/sampleSubmission.csv"
}

# 模型超参数（严格遵循PPT配置）
EMBED_SIZE = 64
HIDDEN_SIZE = 48
N_LAYERS = 1
BIDIRECTIONAL = False
NUM_CLASSES = 5  # 0-4情感标签
BATCH_SIZE = 256
EPOCHS = 8
LR = 0.003
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")
print(f"使用设备：{DEVICE}")


# ---------------------- 2. 数据集类（参考PPT NameDataset） ----------------------
class MovieReviewDataset(Dataset):
    def __init__(self, tsv_path, is_train=True):
        self.data = pd.read_csv(tsv_path, sep='\t', encoding='utf-8', dtype={'Phrase': str, 'Sentiment': int})
        self.is_train = is_train

        # 过滤空短语
        self.data['Phrase'] = self.data['Phrase'].fillna('')
        self.data = self.data[self.data['Phrase'].str.strip().str.len() > 0].reset_index(drop=True)

        self.phrases = self.data['Phrase'].values
        if self.is_train:
            self.sentiments = self.data['Sentiment'].values
        else:
            self.phrase_ids = self.data['PhraseId'].values

        # 构建词汇表（PPT字符级编码逻辑）
        self.vocab = self._build_vocab(min_freq=5) if self.is_train else None
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_len = 20  # 缩短序列长度

    def _build_vocab(self, min_freq):
        """参考PPT编码逻辑，批量处理提升速度"""
        all_chars = ''.join([re.sub(r'[^\w\s]', '', str(p).lower()) for p in self.phrases])
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1

        # PPT风格词汇表：<PAD>→0，<UNK>→1
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for char, cnt in char_counts.items():
            if cnt >= min_freq:
                vocab[char] = len(vocab)
        return vocab

    def _phrase_to_seq(self, phrase):
        """参考PPT padding逻辑"""
        cleaned = re.sub(r'[^\w\s]', '', str(phrase).lower()).strip()[:self.max_len]
        cleaned = cleaned if cleaned else '<UNK>'
        seq = [self.vocab.get(char, self.unk_idx) for char in cleaned]
        seq += [self.pad_idx] * (self.max_len - len(seq))
        return seq, len(cleaned)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        seq, seq_len = self._phrase_to_seq(phrase)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        seq_len_tensor = torch.tensor(seq_len, dtype=torch.long)

        if self.is_train:
            sentiment = torch.tensor(self.sentiments[idx], dtype=torch.long)
            return seq_tensor, seq_len_tensor, sentiment
        else:
            phrase_id = torch.tensor(self.phrase_ids[idx], dtype=torch.long)
            return seq_tensor, seq_len_tensor, phrase_id

    def __len__(self):
        return len(self.data)


# ---------------------- 3. 数据加载（参考PPT DataLoader配置） ----------------------
train_dataset = MovieReviewDataset(DATA_PATHS["train"], is_train=True)
test_dataset = MovieReviewDataset(DATA_PATHS["test"], is_train=False)
test_dataset.vocab = train_dataset.vocab
test_dataset.pad_idx = train_dataset.pad_idx
test_dataset.unk_idx = train_dataset.unk_idx
test_dataset.max_len = train_dataset.max_len


# 自定义collate_fn（参考PPT序列排序逻辑）
def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    seqs, seq_lens, labels = zip(*batch)
    seqs = torch.stack(seqs)
    seq_lens = torch.stack(seq_lens)
    labels = torch.stack(labels)
    return seqs.to(DEVICE), seq_lens.to(DEVICE), labels.to(DEVICE)


# CPU适配：num_workers=0，关闭pin_memory
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    num_workers=0, pin_memory=False, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn,
    num_workers=0, pin_memory=False
)


# ---------------------- 4. TextRNN模型（完全遵循PPT GRU结构） ----------------------
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, n_layers, bidirectional):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)  # PPT嵌入层
        self.gru = nn.GRU(
            embed_size, hidden_size, n_layers,
            bidirectional=bidirectional, batch_first=True, bias=False
        )  # PPT GRU层
        self.n_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.n_directions, num_classes)  # PPT全连接层

    def forward(self, x, seq_lens):
        # 参考PPT forward逻辑
        embed = self.embedding(x)  # (batch_size, max_len, embed_size)
        seq_lens = torch.clamp(seq_lens, min=1)
        # 打包序列（PPT优化手段）
        packed_embed = pack_padded_sequence(embed, seq_lens, batch_first=True, enforce_sorted=True)

        _, hidden = self.gru(packed_embed)  # (n_layers*n_dir, batch_size, hidden_size)

        # 双向GRU拼接（PPT核心逻辑）
        if self.n_directions == 2:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        out = self.fc(hidden)
        return out


# 初始化模型
VOCAB_SIZE = len(train_dataset.vocab)
model = TextRNN(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    n_layers=N_LAYERS,
    bidirectional=BIDIRECTIONAL
).to(DEVICE)

# ---------------------- 5. 训练配置（PPT指定：CrossEntropyLoss+Adam） ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
val_accs = []


# ---------------------- 6. 训练函数（参考PPT trainModel） ----------------------
def train_epoch():
    model.train()
    total_loss = 0
    for seqs, seq_lens, targets in train_loader:
        outputs = model(seqs, seq_lens)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    return avg_loss


# ---------------------- 7. 验证函数（参考PPT testModel） ----------------------
def val_epoch():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 简化验证：取部分数据
        for i, (seqs, seq_lens, targets) in enumerate(train_loader):
            if i > len(train_loader) * 0.05:  # 仅用5%数据验证
                break
            outputs = model(seqs, seq_lens)
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

    acc = correct / total
    val_accs.append(acc)
    return acc


# ---------------------- 8. 主函数（解决CPU多进程冲突） ----------------------
if __name__ == '__main__':
    print(f"Vocab Size: {VOCAB_SIZE}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"训练集样本数：{len(train_dataset)}, 测试集样本数：{len(test_dataset)}")

    # 模型训练（参考PPT主循环）
    for epoch in range(EPOCHS):
        train_loss = train_epoch()
        val_acc = val_epoch()
        print(f"Epoch {epoch + 1:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---------------------- 9. 结果可视化（PPT风格曲线） ----------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ---------------------- 10. 测试集预测与提交文件 ----------------------
    def predict_test():
        model.eval()
        predictions = []
        phrase_ids = []
        with torch.no_grad():
            for seqs, seq_lens, pids in test_loader:
                outputs = model(seqs, seq_lens)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                phrase_ids.extend(pids.cpu().numpy())

        # 生成提交文件（匹配sampleSubmission.csv格式）
        submission_df = pd.DataFrame({
            'PhraseId': phrase_ids,
            'Sentiment': predictions
        }).sort_values('PhraseId').reset_index(drop=True)
        submission_df.to_csv('D:/movie_review_submission.csv', index=False)
        print("提交文件已保存到 D:/movie_review_submission.csv")


    predict_test()


    # ---------------------- 11. 单条评论预测（PPT风格应用示例） ----------------------
    def predict_sentiment(phrase):
        model.eval()
        seq, seq_len = train_dataset._phrase_to_seq(phrase)
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE)
        seq_len_tensor = torch.tensor(seq_len).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(seq_tensor, seq_len_tensor)
            _, pred = torch.max(output, 1)

        sentiment_map = {0: '消极', 1: '略带消极', 2: '中性', 3: '略带积极', 4: '积极'}
        return sentiment_map[pred.item()]


    # 测试示例
    test_phrases = [
        "This movie is absolutely fantastic!",
        "I hate this film so much.",
        "It's neither good nor bad.",
        "The acting is brilliant but the plot is boring."
    ]
    print("\n单条评论预测结果：")
    for p in test_phrases:
        print(f"评论：{p} -> 情感：{predict_sentiment(p)}")