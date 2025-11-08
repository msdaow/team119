import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 简化字体配置：只保留Windows必有的「SimHei（黑体）」，避免多余警告
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据加载（路径已指定，若文件位置不同需修改）
data_path = "D:/data.csv"
df = pd.read_csv(data_path)

# 2. 数据探索
print("数据集基本信息：")
print(df.info())
print("\n数据集前5行：")
print(df.head())
print("\n数据集统计描述：")
print(df.describe())

# 查看标签分布（良性B/恶性M）
print("\n标签分布：")
print(df['diagnosis'].value_counts())
sns.countplot(x='diagnosis', data=df)
plt.title('乳腺癌诊断分布（M=恶性，B=良性）')
plt.show()

# 3. 数据预处理（关键修复：处理无效列，而非删除样本）
# 3.1 查看缺失值（确认Unnamed: 32是全缺失列）   4. Breast Cancer Wisconsin (Diagnostic)
# 简介：乳腺癌诊断数据
# 特征维度：30个特征（细胞核特征）
# 输出：二元分类（恶性/良性）
# 适合任务：医疗诊断分类
# 数据量：569个样本
print("\n缺失值情况：")
print(df.isnull().sum())

# 3.2 修复核心问题：删除全是缺失值的列（Unnamed: 32），而非删除样本
# 先删除无关列：id（样本标识）和Unnamed: 32（全缺失）
df = df.drop(['id', 'Unnamed: 32'], axis=1)
# 此时数据已无缺失值（其他列无缺失），无需再删行

# 3.3 标签编码（M=1表示恶性，B=0表示良性）
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3.4 特征与标签分离
X = df.drop('diagnosis', axis=1)  # 特征：所有列除了诊断结果
y = df['diagnosis']  # 标签：诊断结果（0/1）

# 3.5 划分训练集（80%）和测试集（20%），按标签分布分层
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")  # 验证样本数是否正常

# 3.6 特征标准化（消除量纲影响，适合SVM/KNN等模型）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 多模型训练与评估
# 定义待对比的5种分类模型
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),  # 增加max_iter避免收敛警告
    '支持向量机': SVC(random_state=42, probability=True),  # probability=True用于ROC曲线
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(random_state=42),
    'K近邻': KNeighborsClassifier()
}

# 存储各模型性能结果
results = {}

print("\n各模型性能评估：")
for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 模型预测（标签+概率）
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # 取恶性（1）的预测概率

    # 计算核心评估指标
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    # 输出评估结果
    print(f"\n{name}：")
    print(f"准确率：{accuracy:.4f}")
    print("分类报告（精确率/召回率/F1）：")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵（直观展示预测错误类型）
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['良性', '恶性'],
                yticklabels=['良性', '恶性'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{name}混淆矩阵')
    plt.show()

# 5. 模型性能对比
# 5.1 准确率对比（柱状图）
plt.figure(figsize=(10, 6))
acc_scores = [results[name]['accuracy'] for name in models.keys()]
sns.barplot(x=list(models.keys()), y=acc_scores)
plt.title('各模型准确率比较')
plt.ylim(0.8, 1.0)  # 聚焦高准确率区间，突出差异
# 在柱状图顶部添加准确率数值
for i, v in enumerate(acc_scores):
    plt.text(i, v + 0.005, f'{v:.4f}', ha='center')
plt.show()

# 5.2 ROC曲线对比（评估模型区分能力）
plt.figure(figsize=(8, 6))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# 添加随机猜测基准线（AUC=0.5）
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率（误诊良性为恶性）')
plt.ylabel('真阳性率（正确识别恶性）')
plt.title('各模型ROC曲线对比')
plt.legend(loc="lower right")
plt.show()

# 6. 最佳模型调优（以随机森林为例，医疗场景中鲁棒性更强）
print("\n随机森林模型调优（网格搜索找最佳超参数）：")
# 定义待搜索的超参数组合
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树数量
    'max_depth': [None, 10, 20, 30],  # 树最大深度（控制过拟合）
    'min_samples_split': [2, 5, 10],  # 分裂节点所需最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶节点最小样本数
}

# 初始化随机森林模型
rf = RandomForestClassifier(random_state=42)
# 网格搜索（5折交叉验证，并行计算）
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# 输出调优结果
print(f"最佳超参数组合：{grid_search.best_params_}")
best_rf = grid_search.best_estimator_  # 获取调优后的最佳模型

# 评估调优后的模型性能
y_pred_best = best_rf.predict(X_test_scaled)
print("\n调优后随机森林性能：")
print(f"准确率：{accuracy_score(y_test, y_pred_best):.4f}")
print("分类报告：")
print(classification_report(y_test, y_pred_best))

# 7. 特征重要性分析（随机森林可解释性优势）
if hasattr(best_rf, 'feature_importances_'):
    importances = best_rf.feature_importances_  # 各特征的重要性得分
    features = X.columns  # 特征名称
    indices = np.argsort(importances)[::-1]  # 按重要性从高到低排序

    # 绘制特征重要性柱状图
    plt.figure(figsize=(12, 8))
    plt.bar(range(X.shape[1]), importances[indices], color='#ff7f0e')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.title('乳腺癌诊断关键特征重要性排序')
    plt.xlabel('特征名称（细胞核特征）')
    plt.ylabel('特征重要性得分')
    plt.tight_layout()  # 自动调整布局，避免标签截断
    plt.show()