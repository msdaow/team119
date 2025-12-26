import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import re
from tqdm import tqdm
import pickle
import os
import warnings
import matplotlib.pyplot as plt  # 新增：导入绘图库

# 设置中文字体，避免乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

warnings.filterwarnings("ignore")


# 配置参数
class Config:
    data_path = r"F:\python\deepl\wenben\train.csv"
    model_save_path = "best_bilstm_model2.pth"
    vocab_save_path = "vocab_15k.pkl"
    label_encoder_save_path = "label_encoder.pkl"
    recall_curve_path = "recall_curve.png"  # 新增：召回率曲线保存路径

    # 文本预处理
    max_seq_len = 150
    vocab_size = 15000
    min_word_freq = 3

    # 模型参数
    embedding_dim = 150
    hidden_dim = 300
    num_layers = 2
    dropout = 0.55
    num_classes = 4

    # 训练参数
    batch_size = 64
    epochs = 15
    learning_rate = 8e-5
    weight_decay = 2e-5
    patience = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# 数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def _tokenize(self, text):
        text = re.sub(r"[^a-zA-Z0-9\- ]", " ", str(text).lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def __getitem__(self, idx):
        text, label = str(self.texts[idx]), self.labels[idx]
        tokens = self._tokenize(text)

        # 文本转索引 + 填充/截断
        indices = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens] if tokens else []
        indices = indices[:self.max_seq_len] if len(indices) >= self.max_seq_len else indices + [
            self.vocab["<PAD>"]] * (self.max_seq_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 构建词汇表
def build_vocab(texts):
    all_tokens = []
    tokenize = lambda x: re.sub(r"[^a-zA-Z0-9\- ]", " ", str(x).lower()).split()
    for text in tqdm(texts, desc="Building vocab"):
        all_tokens.extend(tokenize(text))

    # 过滤低频词
    token_counts = Counter(all_tokens)
    filtered = [t for t, c in token_counts.items() if c >= config.min_word_freq]
    common_tokens = filtered[:config.vocab_size - 3]

    # 特殊符号
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
    for token in common_tokens:
        vocab[token] = len(vocab)

    print(f"Vocab size: {len(vocab)}")
    return vocab


# 数据加载
def load_data():
    df = pd.read_csv(config.data_path)
    # 数据校验
    class_count = df["Class Index"].value_counts().sort_index()
    assert class_count.min() >= 29000, "训练集类别不平衡"

    texts = df["Description"].values
    labels = LabelEncoder().fit_transform(df["Class Index"].values)

    # 构建词汇表并保存
    vocab = build_vocab(texts)
    with open(config.vocab_save_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(config.label_encoder_save_path, "wb") as f:
        pickle.dump(LabelEncoder().fit(df["Class Index"].values), f)

    # 划分训练/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 构建DataLoader
    train_ds = NewsDataset(train_texts, train_labels, vocab, config.max_seq_len)
    val_ds = NewsDataset(val_texts, val_labels, vocab, config.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, vocab


# 双向LSTM模型
class BiLSTMNewsClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            config.embedding_dim, config.hidden_dim, config.num_layers,
            bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.fc1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        lstm_out, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n[-1], h_n[-2]], dim=1)  # 双向最后一层
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.fc2(x)


# 训练/验证函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for texts, labels in tqdm(loader, desc="Training"):
        texts, labels = texts.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 新增：计算训练集的召回率等指标
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, precision, recall, f1  # 修改返回值


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for texts, labels in tqdm(loader, desc="Validating"):
            texts, labels = texts.to(config.device), labels.to(config.device)
            logits = model(texts)
            loss = criterion(logits, labels)

            total_loss += loss.item() * texts.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, precision, recall, f1


# 新增：绘制召回率曲线函数
def plot_recall_curve(train_recalls, val_recalls, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_recalls) + 1), train_recalls, label='训练集召回率', marker='o')
    plt.plot(range(1, len(val_recalls) + 1), val_recalls, label='验证集召回率', marker='s')

    plt.title('训练过程中召回率变化曲线', fontsize=14)
    plt.xlabel(' epoch', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(range(1, len(train_recalls) + 1))
    plt.ylim(0, 1.0)  # 召回率范围在0-1之间
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"召回率曲线已保存至: {save_path}")
    plt.close()


# 主训练流程
def main():
    train_loader, val_loader, vocab = load_data()

    # 初始化模型
    model = BiLSTMNewsClassifier(len(vocab)).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_f1, patience_cnt = 0.0, 0
    print(f"Training on {config.device} | Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 新增：用于记录召回率的列表
    train_recalls = []
    val_recalls = []

    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        # 修改：接收训练集的召回率
        train_loss, train_acc, train_p, train_r, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_p, val_r, val_f1 = val_epoch(model, val_loader, criterion)

        # 新增：记录召回率
        train_recalls.append(train_r)
        val_recalls.append(val_r)

        # 打印日志（增加了召回率显示）
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Recall: {train_r:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_r:.4f} | Val F1: {val_f1:.4f}")

        # 早停 + 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_config = {
                "max_seq_len": config.max_seq_len,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "num_classes": config.num_classes
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": save_config
            }, config.model_save_path)
            print(f"Best model saved (F1: {best_f1:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config.patience:
                print("Early stopping triggered")
                break

    # 新增：训练结束后绘制召回率曲线
    plot_recall_curve(train_recalls, val_recalls, config.recall_curve_path)


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import re
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore")
from train import BiLSTMNewsClassifier

# 核心配置（极简版）
CFG = {
    "test_path": r"F:\python\deepl\wenben\test.csv",
    "model_path": "best_bilstm_model2.pth",
    "vocab_path": "vocab_15k.pkl",
    "label_encoder_path": "label_encoder.pkl",
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "default_cfg": {"max_seq_len": 150, "embedding_dim": 150, "hidden_dim": 300,
                    "num_layers": 2, "dropout": 0.55, "num_classes": 4},
    "random_num": 5,
    "label_map": {0: "世界新闻", 1: "体育新闻", 2: "商业新闻", 3: "科技新闻"}
}


# 通用分词函数
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\- ]", " ", str(text).lower()).strip()
    return re.sub(r"\s+", " ", text).split()


# 文本预处理（统一填充/截断逻辑）
def preprocess_text(text, vocab, max_seq_len):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens] if tokens else []
    return indices[:max_seq_len] + [vocab["<PAD>"]] * (max_seq_len - len(indices)) if len(
        indices) < max_seq_len else indices[:max_seq_len]


# 测试集Dataset（极简版）
class TestDataset(Dataset):
    def __init__(self, texts, vocab, max_seq_len):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(preprocess_text(self.texts[idx], self.vocab, self.max_seq_len), dtype=torch.long)


# 生成随机测试文本（简化模板和填充逻辑）
def gen_rand_texts(num=5):
    templates = {
        0: ["UN: Conflict in {} displaced {} people.", "{} sets {}% carbon cut target by {}."],
        1: ["{} beat {} {}:{} to win {} title.", "{} player out with {} injury for {} games."],
        2: ["{} profit up {}% in Q{}.", "Fed hikes rates by {} bps, hits {} markets."],
        3: ["{} launches {} phone ({}GB RAM, {}MP camera).", "AI firm {} raises ${}M for {} tech."]
    }
    fill = {
        "country": ["Ukraine", "Brazil", "India"], "num": ["500000", "10", "25", "2026"],
        "team": ["Lakers", "Real Madrid"], "company": ["Apple", "Google", "Tesla"],
        "product": ["iPhone 16", "AI assistant"], "injury": ["knee", "ankle"]
    }

    texts, labels = [], []
    for _ in range(num):
        label = random.randint(0, 3)
        tpl = random.choice(templates[label])
        labels.append(label)

        if label == 0:
            text = tpl.format(random.choice(fill["country"]), random.choice(fill["num"])) if tpl.count(
                "{}") == 2 else tpl.format(random.choice(fill["country"]), random.choice(fill["num"]),
                                           random.choice(fill["num"]))
        elif label == 1:
            text = tpl.format(random.choice(fill["team"]), random.choice(fill["team"]), random.choice(fill["num"]),
                              random.choice(fill["num"]), random.choice(fill["num"])) if tpl.count(
                "{}") == 5 else tpl.format(random.choice(fill["team"]), random.choice(fill["injury"]),
                                           random.choice(fill["num"]))
        elif label == 2:
            text = tpl.format(random.choice(fill["company"]), random.choice(fill["num"]),
                              random.choice(fill["num"])) if tpl.count("{}") == 3 else tpl.format(
                random.choice(fill["num"]), random.choice(fill["country"]))
        else:
            text = tpl.format(random.choice(fill["company"]), random.choice(fill["product"]),
                              random.choice(fill["num"]), random.choice(fill["num"]))
        texts.append(text)
    return texts, labels


# 加载资源（合并所有加载逻辑）
def load_resources():
    # 加载数据
    df = pd.read_csv(CFG["test_path"])
    assert len(df) == 7600, "测试集规模错误"

    # 加载词汇表/标签编码器
    with open(CFG["vocab_path"], "rb") as f: vocab = pickle.load(f)
    with open(CFG["label_encoder_path"], "rb") as f: le = pickle.load(f)

    # 加载模型和配置（兜底逻辑）
    ckpt = torch.load(CFG["model_path"], map_location=CFG["device"])
    model_cfg = CFG["default_cfg"].copy()
    model_cfg.update(ckpt.get("config", {}))

    # 初始化模型
    model = BiLSTMNewsClassifier(len(vocab)).to(CFG["device"])
    model.load_state_dict(ckpt["model_state_dict"])

    return df, vocab, le, model, model_cfg


# 主函数（极简版）
def main():
    df, vocab, le, model, m_cfg = load_resources()
    max_seq_len = m_cfg["max_seq_len"]

    # 1. 随机文本预测
    print("=" * 60 + "\n随机文本预测结果\n" + "=" * 60)
    rand_texts, true_labels = gen_rand_texts(CFG["random_num"])
    for i, (text, true_l) in enumerate(zip(rand_texts, true_labels)):
        indices = torch.tensor(preprocess_text(text, vocab, max_seq_len), dtype=torch.long).unsqueeze(0).to(
            CFG["device"])
        pred_l = torch.argmax(model(indices), dim=1).cpu().item()
        res = "✅ 正确" if pred_l == true_l else "❌ 错误"
        print(
            f"\n【文本 {i + 1}】\n内容: {text}\n真实类别: {CFG['label_map'][true_l]} | 预测类别: {CFG['label_map'][pred_l]} | {res}")

    # 2. 批量预测
    print("\n" + "=" * 60 + "\n测试集批量评估\n" + "=" * 60)
    test_texts, test_labels = df["Description"].values, le.transform(df["Class Index"].values)
    test_ds = TestDataset(test_texts, vocab, max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # 批量推理
    preds = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Predicting"):
            preds.extend(torch.argmax(model(x.to(CFG["device"])), dim=1).cpu().numpy())

    # 计算指标并保存
    acc = accuracy_score(test_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
    print(f"\n测试集规模: 7600条 | 准确率: {acc:.4f}")
    print(f"宏平均精度: {p:.4f} | 召回率: {r:.4f} | F1: {f1:.4f}")
    pd.DataFrame({"True_Label": test_labels, "Pred_Label": preds}).to_csv("test_pred_results.csv", index=False)
    print("\n预测结果已保存到 test_pred_results.csv")


if __name__ == "__main__":
    main()