import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置matplotlib字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体
plt.rcParams['axes.unicode_minus'] = False


# 数据加载和预处理
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    print(f"Training data: {len(train_df)} samples")
    print(f"Sentiment distribution:\n{train_df['Sentiment'].value_counts().sort_index()}")

    # 文本清洗
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    train_df['text'] = train_df['Phrase'].apply(clean_text)
    test_df['text'] = test_df['Phrase'].apply(clean_text)

    # 构建词汇表
    words = []
    for text in train_df['text']:
        words.extend(text.split())

    vocab = {'<pad>': 0, '<unk>': 1}
    for word in set(words):
        if words.count(word) >= 2:
            vocab[word] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}")
    return train_df, test_df, vocab


# 数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()[:self.max_len]
        seq = [self.vocab.get(word, 1) for word in tokens]

        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))

        return torch.tensor(seq), torch.tensor(self.labels[idx])


# TextRNN模型
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_size=128, num_classes=5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 训练函数
def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"Start training for {epochs} epochs...")
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print('-' * 40)

    return train_losses, val_losses, train_accs, val_accs


# 绘制训练曲线（英文标签）
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))

    # Loss曲线
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-s', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, val_accs, 'r-s', label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印最终结果
    print(f"\nFinal Results:")
    print(f"Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_accs[-1]:.2f}%")
    print(f"Val Loss: {val_losses[-1]:.4f}, Accuracy: {val_accs[-1]:.2f}%")


# 主函数
def main():
    # 加载数据
    train_df, test_df, vocab = load_data(
        r"C:\Users\86131\Downloads\train.tsv",
        r"C:\Users\86131\Downloads\test.tsv"
    )

    # 分割训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values,
        train_df['Sentiment'].values,
        test_size=0.2,
        random_state=42,
        stratify=train_df['Sentiment']
    )

    # 创建数据集和数据加载器
    train_dataset = ReviewDataset(train_texts, train_labels, vocab)
    val_dataset = ReviewDataset(val_texts, val_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    model = TextRNN(len(vocab)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练（5个epoch）
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, epochs=5
    )

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return model, vocab, test_df


# 预测函数
def predict(model, test_df, vocab):
    model.eval()
    predictions = []

    test_dataset = ReviewDataset(test_df['text'].values, [0] * len(test_df), vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for texts, _ in test_loader:
            texts = texts.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'PhraseId': test_df['PhraseId'],
        'Sentiment': predictions
    })
    submission.to_csv(r"C:\Users\86131\Downloads\submission.csv", index=False)
    print("Prediction completed! Submission file saved.")

    return submission


# 运行
if __name__ == "__main__":
    model, vocab, test_df = main()
    submission = predict(model, test_df, vocab)