import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import os
import random

# ====================== 文件路径配置 ======================
TRAIN_CSV_PATH = "C:/Users/JSJ-XL-1L/Desktop/wenben/train.csv"
TEST_CSV_PATH = "C:/Users/JSJ-XL-1L/Desktop/wenben/test.csv"
SAVE_DIR = "C:/Users/JSJ-XL-1L/Desktop/wenben/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================== 1. 数据加载与预处理 ======================
try:
    train_df = pd.read_csv(TRAIN_CSV_PATH, encoding='utf-8')
    test_df = pd.read_csv(TEST_CSV_PATH, encoding='utf-8')
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：找不到CSV文件，请检查路径！")
    print(f"训练集路径：{TRAIN_CSV_PATH}")
    print(f"测试集路径：{TEST_CSV_PATH}")
    exit()
except Exception as e:
    print(f"数据加载失败：{str(e)}")
    exit()

# 合并标题和描述，处理空值
train_df['text'] = train_df['Title'].fillna('') + ' ' + train_df['Description'].fillna('')
test_df['text'] = test_df['Title'].fillna('') + ' ' + test_df['Description'].fillna('')

# 标签调整（1-4 → 0-3）
train_df['label'] = train_df['Class Index'] - 1
test_df['label'] = test_df['Class Index'] - 1

# 提取特征和标签
X_train = train_df['text'].values
y_train = train_df['label'].values
X_test = test_df['text'].values
y_test = test_df['label'].values

# ====================== 2. 文本序列化 ======================
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128

# 初始化分词器
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

# 文本转序列 + 填充
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

# 标签转one-hot
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=4)

# 划分验证集
X_train_pad, X_val_pad, y_train_onehot, y_val_onehot = train_test_split(
    X_train_pad, y_train_onehot, test_size=0.1, random_state=42
)

# ====================== 3. 构建RNN模型 ======================
model = Sequential([
    Embedding(
        input_dim=MAX_NUM_WORDS,
        output_dim=EMBEDDING_DIM
    ),
    Dropout(0.2),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n模型结构：")
model.summary()

# ====================== 4. 模型训练（移除早停，完整15轮） ======================
print("\n开始训练模型（完整15轮训练）...")
BATCH_SIZE = 256
EPOCHS = 15

# 训练模型（无早停，完整训练15轮）
history = model.fit(
    X_train_pad, y_train_onehot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_pad, y_val_onehot),
    verbose=1
)

# ====================== 5. 训练曲线可视化 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制准确率和损失曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 准确率曲线
ax1.plot(history.history['accuracy'], label='训练准确率', color='#3498db')
ax1.plot(history.history['val_accuracy'], label='验证准确率', color='#e74c3c')
ax1.set_title('模型准确率变化（15轮训练）')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('准确率')
ax1.legend()
ax1.grid(alpha=0.3)

# 损失曲线
ax2.plot(history.history['loss'], label='训练损失', color='#2ecc71')
ax2.plot(history.history['val_loss'], label='验证损失', color='#f39c12')
ax2.set_title('模型损失变化（15轮训练）')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('损失')
ax2.legend()
ax2.grid(alpha=0.3)

# 保存图片
plt.tight_layout()
train_history_path = os.path.join(SAVE_DIR, "training_history_15epochs_no_earlystop.png")
plt.savefig(train_history_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n训练曲线已保存：{train_history_path}")

# ====================== 6. 模型评估（保留核心验证 + 随机抽样检验） ======================
# 定义类别名称映射
category_map = {0: '政治/军事', 1: '体育', 2: '财经', 3: '科技/IT'}

# 6.1 基础准确率验证
print("\n========== 基础准确率验证 ==========")
# 测试集整体准确率
test_loss, test_acc = model.evaluate(X_test_pad, y_test_onehot, verbose=0)
print(f"测试集整体准确率：{test_acc:.4f} ({test_acc*100:.2f}%)")

# 6.2 分类报告（详细准确率/召回率/F1）
print("\n========== 详细分类指标 ==========")
y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=list(category_map.values())))

# 6.3 从test.csv随机选择样本验证（核心功能）
print("\n========== 从test.csv随机抽样验证（10个样本） ==========")
# 随机选择10个测试样本（可修改数字调整抽样数量）
sample_count = 10
sample_indices = random.sample(range(len(X_test)), sample_count)
sample_texts = X_test[sample_indices]
sample_true_labels = y_test[sample_indices]
sample_pred_labels = y_pred[sample_indices]

# 输出抽样结果
correct_count = 0
for i, idx in enumerate(sample_indices):
    true_cat = category_map[sample_true_labels[i]]
    pred_cat = category_map[sample_pred_labels[i]]
    is_correct = "正确" if true_cat == pred_cat else "错误"
    if is_correct == "正确":
        correct_count += 1
    # 截断文本避免输出过长（保留前100个字符）
    short_text = sample_texts[i][:100] + "..." if len(sample_texts[i]) > 100 else sample_texts[i]
    print(f"\n样本{i+1} - 预测{is_correct}")
    print(f"文本内容：{short_text}")
    print(f"真实类别：{true_cat} | 预测类别：{pred_cat}")

# 输出抽样准确率
sample_acc = correct_count / sample_count
print(f"\n本次抽样{sample_count}个样本，准确率：{sample_acc:.4f} ({sample_acc*100:.2f}%)")

# 6.4 混淆矩阵可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(category_map.values()),
            yticklabels=list(category_map.values()))
plt.title('混淆矩阵（15轮训练，无早停）')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
cm_path = os.path.join(SAVE_DIR, "confusion_matrix_15epochs_no_earlystop.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n混淆矩阵已保存：{cm_path}")

# ====================== 7. 保存模型 ======================
model_path = os.path.join(SAVE_DIR, "news_rnn_model_15epochs_no_earlystop.h5")
model.save(model_path)
print(f"\n模型已保存：{model_path}")
print("\n所有验证任务完成！")