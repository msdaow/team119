import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import time
import math
import zipfile
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置超参数
HIDDEN_SIZE = 128
EMBEDDING_DIM = 100
BATCH_SIZE = 64
N_LAYER = 2
N_EPOCHS = 10
MAX_LENGTH = 50
MIN_WORD_COUNT = 2
LEARNING_RATE = 0.001

# 强制使用CPU
device = torch.device("cpu")
print("使用CPU进行训练")


# 数据预处理类
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, sentiments, word2idx, max_length=MAX_LENGTH):
        self.reviews = reviews
        self.sentiments = sentiments
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        sentiment = self.sentiments[index]

        # 将文本转换为索引序列
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in review.split()]

        # 截断或填充序列
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_length - len(sequence))

        return torch.tensor(sequence, dtype=torch.long), torch.tensor(sentiment, dtype=torch.long)


# 文本预处理函数 - 增强版
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 转换为小写
    text = text.lower()

    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 移除URL
    text = re.sub(r'http\S+', '', text)

    # 移除标点符号和特殊字符，但保留基本标点用于情感分析
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)

    # 处理重复字符（如"coooool" -> "cool"）
    text = re.sub(r'(.)\1+', r'\1\1', text)

    return text.strip()


# 构建词汇表
def build_vocab(texts, min_count=MIN_WORD_COUNT):
    word_counter = Counter()
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_counter.update(words)

    # 创建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2

    # 按词频排序
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    for word, count in sorted_words:
        if count >= min_count:
            vocab[word] = idx
            idx += 1

    print(f"构建词汇表完成，共 {len(vocab)} 个词")
    return vocab


# 改进的TextRNN模型
class ImprovedTextRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_layers=2, dropout=0.3):
        super(ImprovedTextRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # GRU层 - 使用双向
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 嵌入层
        embedded = self.embedding(x)

        # GRU层
        gru_out, _ = self.gru(embedded)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)

        # 分类
        output = self.classifier(context_vector)
        return output


# 解压并加载数据
def extract_and_load_data(zip_path, extract_to='.'):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            tsv_file = [f for f in file_list if f.endswith('.tsv')][0]

            zip_ref.extract(tsv_file, extract_to)
            extracted_path = os.path.join(extract_to, tsv_file)

            df = pd.read_csv(extracted_path, sep='\t')
            print(f"成功加载数据，共 {len(df)} 条记录")

            os.remove(extracted_path)
            return df
    except Exception as e:
        print(f"解压或加载数据失败: {e}")
        return None


# 加载和预处理数据 - 修复测试集问题
def load_data(train_zip_path, test_zip_path=None, sample_ratio=0.3):
    # 加载训练数据
    train_df = extract_and_load_data(train_zip_path)
    if train_df is None:
        print("使用示例数据进行演示...")
        sample_reviews = [
            "this movie is great and wonderful",
            "terrible and boring film",
            "average movie nothing special",
            "amazing performance by actors",
            "worst movie I have ever seen",
            "fantastic plot and acting",
            "disappointing and poorly made",
            "mediocre at best",
            "brilliant cinematography and story",
            "awful waste of time"
        ]
        sample_sentiments = [4, 0, 2, 4, 0, 4, 0, 1, 4, 0]
        return sample_reviews * 10, sample_sentiments * 10, None, None

    # 数据采样 - 修复分组采样问题
    if sample_ratio < 1.0:
        # 使用简单采样而不是分组采样
        train_df = train_df.sample(frac=sample_ratio, random_state=42)
        print(f"采样后数据量: {len(train_df)}")

    # 预处理文本
    train_df['cleaned_phrase'] = train_df['Phrase'].apply(preprocess_text)

    # 过滤空文本
    train_df = train_df[train_df['cleaned_phrase'].str.len() > 0]

    train_reviews = train_df['cleaned_phrase'].tolist()
    train_sentiments = train_df['Sentiment'].tolist()

    # 检查情感分布
    sentiment_counts = pd.Series(train_sentiments).value_counts().sort_index()
    print("训练集情感分布:")
    for sentiment, count in sentiment_counts.items():
        print(f"  情感 {sentiment}: {count} 条")

    # 加载测试数据 - 修复测试集问题
    test_reviews, test_sentiments = None, None
    if test_zip_path and os.path.exists(test_zip_path):
        test_df = extract_and_load_data(test_zip_path)
        if test_df is not None:
            # 测试数据采样
            if sample_ratio < 1.0:
                test_df = test_df.sample(frac=sample_ratio, random_state=42)

            test_df['cleaned_phrase'] = test_df['Phrase'].apply(preprocess_text)
            test_df = test_df[test_df['cleaned_phrase'].str.len() > 0]
            test_reviews = test_df['cleaned_phrase'].tolist()

            # 检查测试集是否有标签
            if 'Sentiment' in test_df.columns:
                test_sentiments = test_df['Sentiment'].tolist()
                print("测试集情感分布:")
                sentiment_counts_test = pd.Series(test_sentiments).value_counts().sort_index()
                for sentiment, count in sentiment_counts_test.items():
                    print(f"  情感 {sentiment}: {count} 条")
            else:
                print("测试集没有标签列，将仅用于预测")
                test_sentiments = None

    return train_reviews, train_sentiments, test_reviews, test_sentiments


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


# 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # 记录每个类别的准确率
    class_correct = [0] * 5
    class_total = [0] * 5

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # 计算每个类别的准确率
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (pred[i] == label).item()
                class_total[label] += 1

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')

    # 打印每个类别的准确率
    print("各类别准确率:")
    sentiment_labels = {0: '消极', 1: '略带消极', 2: '中性', 3: '略带积极', 4: '积极'}
    for i in range(5):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"  {sentiment_labels[i]}: {class_correct[i]}/{class_total[i]} ({class_acc:.2f}%)")

    return test_loss, accuracy


# 时间计算函数
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 主函数
def main():
    # 设置文件路径
    train_zip_path = r"D:\train.tsv.zip"
    test_zip_path = r"D:\test.tsv.zip"

    # 加载数据
    print("加载数据...")
    train_reviews, train_sentiments, test_reviews, test_sentiments = load_data(
        train_zip_path, test_zip_path, sample_ratio=0.3
    )

    # 构建词汇表
    print("构建词汇表...")
    word2idx = build_vocab(train_reviews)
    vocab_size = len(word2idx)
    print(f"词汇表大小: {vocab_size}")

    # 如果没有测试数据或测试数据没有标签，从训练数据中划分
    if test_reviews is None or test_sentiments is None:
        print("未找到测试数据或测试数据没有标签，从训练数据中划分...")
        train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(
            train_reviews, train_sentiments, test_size=0.2, random_state=42, stratify=train_sentiments
        )

    # 创建数据集和数据加载器
    train_dataset = MovieReviewDataset(train_reviews, train_sentiments, word2idx)
    test_dataset = MovieReviewDataset(test_reviews, test_sentiments, word2idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 初始化模型
    output_size = 5  # 5个情感类别
    model = ImprovedTextRNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, output_size, N_LAYER)
    model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params}, 可训练参数: {trainable_params}")

    # 定义损失函数和优化器
    # 使用带权重的交叉熵损失处理类别不平衡
    sentiment_counts = pd.Series(train_sentiments).value_counts().sort_index()
    class_weights = 1.0 / sentiment_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print(f"开始训练，共 {N_EPOCHS} 个轮次...")
    start_time = time.time()

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_accuracy = 0
    patience = 3
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n轮次 {epoch}/{N_EPOCHS} - {time_since(start_time)}")

        # 训练
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 测试
        test_loss, test_acc = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 学习率调度
        scheduler.step()

        # 早停法
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'word2idx': word2idx,
                'vocab_size': vocab_size,
                'hidden_size': HIDDEN_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'output_size': output_size,
                'n_layers': N_LAYER
            }, 'best_textrnn_sentiment_model.pth')
            print("保存最佳模型!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停! 在轮次 {epoch} 停止训练")
            break

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练和测试损失')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='训练准确率')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='测试准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('训练和测试准确率')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"训练完成! 总用时: {time_since(start_time)}")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")


# 预测函数
def predict_sentiment(model, word2idx, text):
    model.eval()

    # 预处理文本
    processed_text = preprocess_text(text)
    sequence = [word2idx.get(word, word2idx['<UNK>']) for word in processed_text.split()]

    # 填充序列
    if len(sequence) < MAX_LENGTH:
        sequence = sequence + [word2idx['<PAD>']] * (MAX_LENGTH - len(sequence))
    else:
        sequence = sequence[:MAX_LENGTH]

    # 转换为tensor
    input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    sentiment_labels = {0: '消极', 1: '略带消极', 2: '中性', 3: '略带积极', 4: '积极'}
    return sentiment_labels[prediction], confidence


# 加载已保存的模型进行预测
def load_model_for_prediction(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    word2idx = checkpoint['word2idx']

    model = ImprovedTextRNNClassifier(
        checkpoint['vocab_size'],
        checkpoint['embedding_dim'],
        checkpoint['hidden_size'],
        checkpoint['output_size'],
        checkpoint['n_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, word2idx


if __name__ == '__main__':
    main()

    # 示例预测
    print("\n示例预测:")
    try:
        model, word2idx = load_model_for_prediction('best_textrnn_sentiment_model.pth')
        test_texts = [
            "This movie is absolutely fantastic and I love it!",
            "Terrible movie, waste of time.",
            "It's okay, nothing special.",
            "Amazing performance by all actors, great storyline!",
            "Boring and predictable, would not recommend.",
            "The cinematography was beautiful but the plot was weak."
        ]
        for text in test_texts:
            sentiment, confidence = predict_sentiment(model, word2idx, text)
            print(f"文本: '{text}' -> 情感: {sentiment} (置信度: {confidence:.2f})")
    except Exception as e:
        print(f"预测示例失败: {e}")
        # 如果没有保存最佳模型，使用最后训练的模型
        try:
            model, word2idx = load_model_for_prediction('textrnn_sentiment_model.pth')
            test_texts = [
                "This movie is absolutely fantastic and I love it!",
                "Terrible movie, waste of time.",
                "It's okay, nothing special."
            ]
            for text in test_texts:
                sentiment, confidence = predict_sentiment(model, word2idx, text)
                print(f"文本: '{text}' -> 情感: {sentiment} (置信度: {confidence:.2f})")
        except Exception as e2:
            print(f"再次尝试预测失败: {e2}")