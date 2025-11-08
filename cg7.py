import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import matplotlib


# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    try:
        # 尝试不同的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 验证字体是否设置成功
        test_font = plt.matplotlib.font_manager.FontProperties(family='SimHei')
        print("中文字体设置成功: SimHei")
        return True
    except:
        try:
            # 如果SimHei不可用，尝试其他字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
            print("使用备选中文字体")
            return True
        except:
            print("警告: 无法设置中文字体，图表可能无法正常显示中文")
            return False


# 设置中文字体
set_chinese_font()

# 设置TensorFlow日志级别
tf.get_logger().setLevel('ERROR')

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)


# 数据加载函数（保持不变）
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f'无效的MNIST图像文件: {filename}')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f'无效的MNIST标签文件: {filename}')
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# 文件路径
file_paths = {
    'train_images': r"C:\Users\86131\Downloads\train-images.idx3-ubyte",
    'train_labels': r"C:\Users\86131\Downloads\train-labels.idx1-ubyte",
    'test_images': r"C:\Users\86131\Downloads\t10k-images.idx3-ubyte",
    'test_labels': r"C:\Users\86131\Downloads\t10k-labels.idx1-ubyte"
}

# 加载数据
print("正在加载MNIST数据集...")
x_train = load_mnist_images(file_paths['train_images'])
y_train = load_mnist_labels(file_paths['train_labels'])
x_test = load_mnist_images(file_paths['test_images'])
y_test = load_mnist_labels(file_paths['test_labels'])

print(f"训练集形状: {x_train.shape}")
print(f"训练标签形状: {y_train.shape}")
print(f"测试集形状: {x_test.shape}")
print(f"测试标签形状: {y_test.shape}")

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 分割训练集和验证集
x_val = x_train[55000:]
y_val = y_train[55000:]
x_train = x_train[:55000]
y_train = y_train[:55000]

print(f"训练集: {x_train.shape[0]} 张图片")
print(f"验证集: {x_val.shape[0]} 张图片")
print(f"测试集: {x_test.shape[0]} 张图片")

# 准备数据
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_val_cnn = x_val.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

x_train_fc = x_train.reshape(-1, 784)
x_val_fc = x_val.reshape(-1, 784)
x_test_fc = x_test.reshape(-1, 784)


# 模型构建函数
def build_fc_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model


def build_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model


# 训练函数
def train_model(model, x_train, y_train, x_val, y_val, model_name):
    print(f"\n开始训练{model_name}模型...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=15,
                        validation_data=(x_val, y_val),
                        verbose=1)
    return history


# 训练模型
print("=" * 50)
fc_model = build_fc_model()
fc_history = train_model(fc_model, x_train_fc, y_train, x_val_fc, y_val, "FC")

print("=" * 50)
cnn_model = build_cnn_model()
cnn_history = train_model(cnn_model, x_train_cnn, y_train, x_val_cnn, y_val, "CNN")

# 评估模型
print("\n评估模型性能...")
fc_test_loss, fc_test_acc = fc_model.evaluate(x_test_fc, y_test, verbose=0)
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)

print(f"FC模型测试准确率: {fc_test_acc:.4f}")
print(f"CNN模型测试准确率: {cnn_test_acc:.4f}")

# 绘制准确率和损失曲线
plt.figure(figsize=(15, 5))

# 准确率图表
plt.subplot(1, 2, 1)
plt.plot(fc_history.history['accuracy'], label='FC训练准确率', color='blue', linestyle='-', linewidth=2)
plt.plot(fc_history.history['val_accuracy'], label='FC验证准确率', color='blue', linestyle='--', linewidth=2)
plt.plot(cnn_history.history['accuracy'], label='CNN训练准确率', color='red', linestyle='-', linewidth=2)
plt.plot(cnn_history.history['val_accuracy'], label='CNN验证准确率', color='red', linestyle='--', linewidth=2)
plt.title('训练和验证准确率对比', fontsize=14)
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(True, alpha=0.3)

# 损失图表
plt.subplot(1, 2, 2)
plt.plot(fc_history.history['loss'], label='FC训练损失', color='blue', linestyle='-', linewidth=2)
plt.plot(fc_history.history['val_loss'], label='FC验证损失', color='blue', linestyle='--', linewidth=2)
plt.plot(cnn_history.history['loss'], label='CNN训练损失', color='red', linestyle='-', linewidth=2)
plt.plot(cnn_history.history['val_loss'], label='CNN验证损失', color='red', linestyle='--', linewidth=2)
plt.title('训练和验证损失对比', fontsize=14)
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 绘制测试准确率比较柱状图
plt.figure(figsize=(10, 6))
models = ['FC模型', 'CNN模型']
test_accuracies = [fc_test_acc, cnn_test_acc]
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(models, test_accuracies, color=colors, alpha=0.7, width=0.6)
plt.title('FC vs CNN 模型测试准确率比较', fontsize=16)
plt.ylabel('测试准确率', fontsize=12)
plt.ylim(0.9, 1.0)

# 在柱状图上显示准确率数值
for bar, accuracy in zip(bars, test_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{accuracy:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 打印详细比较结果
print("\n" + "=" * 60)
print("模型性能比较总结")
print("=" * 60)
print(f"FC模型最终训练准确率: {fc_history.history['accuracy'][-1]:.4f}")
print(f"FC模型最终验证准确率: {fc_history.history['val_accuracy'][-1]:.4f}")
print(f"FC模型测试准确率: {fc_test_acc:.4f}")
print(f"CNN模型最终训练准确率: {cnn_history.history['accuracy'][-1]:.4f}")
print(f"CNN模型最终验证准确率: {cnn_history.history['val_accuracy'][-1]:.4f}")
print(f"CNN模型测试准确率: {cnn_test_acc:.4f}")

# 性能提升计算
improvement = cnn_test_acc - fc_test_acc
print(f"\nCNN相比FC的性能提升: {improvement:.4f} ({improvement * 100:.2f}%)")

# 模型参数数量比较
fc_params = fc_model.count_params()
cnn_params = cnn_model.count_params()
print(f"\n模型参数数量:")
print(f"FC模型参数数量: {fc_params:,}")
print(f"CNN模型参数数量: {cnn_params:,}")
print(f"参数数量比: {fc_params / cnn_params:.2f}:1")