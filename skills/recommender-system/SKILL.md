---
name: recommender-system
description: |
  推荐系统（Recommender System）—— 基于内容的推荐 / 协同过滤，纯 NumPy 实现。
  当用户提到"推荐系统"、"recommender system"、"协同过滤"、"collaborative filtering"、
  "内容推荐"、"content-based filtering"、"推荐算法"、"评分预测"、"帮我推荐"、
  "电影推荐"、"商品推荐"、"recommendation system"、"推荐引擎"、"基于内容的推荐"、
  "基于协同过滤"、"预测评分"、"推荐电影"、"推荐商品"、"similar items"、"相似物品"、
  "找出相似"等类似场景时触发。
  适用于任何"有用户-物品评分数据，想预测缺失评分或推荐物品"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 推荐系统（Recommender System）

## 概述

基于吴恩达课程第 9 周的推荐系统算法，核心原理：

- **基于内容的推荐**：利用物品特征 x^(i)，学习用户偏好参数 theta^(j)，predict = theta^T x
- **协同过滤**：同时学习物品特征 x^(i) 和用户参数 theta^(j)，从评分数据自动发现隐含特征
- **均值归一化**：消除评分偏置，解决新用户冷启动问题
- **相似物品**：基于学习到的特征向量 ‖x^(i) - x^(k)‖² 找相似物品
- 所有代码使用纯 NumPy + Matplotlib，向量化实现，自动保存为带时间戳的 `.py` 文件

### 核心规则

- 数据输入支持三种方式：粘贴数组、CSV 文件、口头描述
- 自动判断走"基于内容推荐"还是"协同过滤"
- 协同过滤时自动推荐隐含特征数 n_features
- lambda、alpha、iters 三个参数智能推荐，用户可调整
- 输出包含：预测函数、Top-N 推荐、相似物品、`.py` 代码文件、`.png` 可视化图表

---

## 交互流程

### 步骤 1：数据输入

> 请提供你的评分数据。支持三种方式：
>
> **A. 粘贴数组** — 直接粘贴 Python 格式的评分矩阵
> **B. CSV 文件** — 提供文件路径，我来读取
> **C. 口头描述** — 描述场景，我来生成模拟数据
>
> 如果你有物品特征（如电影类型、主演等），也请一并提供。

自动判断输入类型：

```
包含 ".csv"                            → CSV 文件模式
包含 "[[", "np.array", "Y =", "R =" 等 → 数组粘贴模式
包含 "大概", "模拟", "比如" 等          → 口头描述模式
```

#### 模式 A：粘贴数组

> 请粘贴你的数据。格式示例：
> ```python
> # 评分矩阵（0 = 未评分）：
> # 行=电影, 列=用户
> Y = [[5, 4, 0, 0],
>      [3, 0, 0, 4],
>      [0, 4, 5, 0],
>      [0, 0, 5, 4]]
>
> # 或者评分记录列表：
> # (物品索引, 用户索引, 评分)
> ratings = [(0, 0, 5.0), (0, 1, 4.0), (1, 0, 3.0), ...]
> ```
>
> 如果有物品特征：
> ```python
> X_features = [[0.9, 0.0, 1.0, 0.1],   # 电影1的特征
>               [0.1, 0.0, 0.9, 1.0]]    # 电影2的特征
> ```
>
> 还需提供：物品名列表、用户名列表、推荐主题

提取：`Y`（评分矩阵，n_items × n_users）、`R`（指示矩阵，自动从 Y 计算）、可选的 `X_features`（物品特征矩阵，n_features × n_items）、`item_names`、`user_names`、`title_topic`

#### 模式 B：CSV 文件

> 请提供 CSV 文件路径。支持两种格式：
> - **矩阵式**：行=物品, 列=用户, 值=评分, 空=未评分
> - **记录式**：user_id, item_id, rating 三列
>
> 如果物品特征在另一个 CSV 中，也请提供路径。

读取逻辑：先尝试 UTF-8，失败则尝试 GBK。自动跳过非数值列。空值/0 标记为"未评分"。

#### 模式 C：口头描述

> 请描述你的推荐场景：
> - 有多少个物品？多少个用户？评分范围是多少？
> - 大概有多少条评分记录？
> - 是什么类型的推荐（如电影推荐、商品推荐）？

根据描述用 `np.random.seed(42)` 生成模拟评分数据，打印预览，让用户确认。

---

### 步骤 2：数据初探确认

解析完成后，打印数据摘要，确认无误再继续：

> 📊 **数据概览**
>
> | 指标 | 值 |
> |------|-----|
> | 物品数 (n_items) | {n_items} |
> | 用户数 (n_users) | {n_users} |
> | 已评分条目 | {n_rated} |
> | 评分稀疏度 | {sparsity:.1%} |
> | 评分范围 | [{min_rating:.1f}, {max_rating:.1f}] |
> | 提供物品特征 | {是/否} |
>
> 评分分布：
> ```
> 1分: ██ 2条
> 2分: ████████ 8条
> 3分: ██████████████ 14条
> 4分: ██████████ 10条
> 5分: ██████ 6条
> ```
>
> 数据看起来正确吗？（回复"可以"继续，或指出问题）

数据健康检查：

| 检查项 | 处理 |
|--------|------|
| n_rated < 10 | ⚠️ 警告评分太少，模型可能不可靠 |
| 某用户无评分 | 标记为"新用户"，均值归一化自动处理 |
| 某物品无评分 | ⚠️ 建议移除该物品（无法学到其特征） |
| 所有评分相同 | ⚠️ 数据无区分度，无法学习 |

---

### 步骤 3：算法选择 + 特征数推荐

```
用户是否提供了物品特征（X_features）？
  │
  ├── 是 → 基于内容的推荐
  │   理由：物品特征已知，只需学习用户偏好参数 theta
  │   参数少、收敛快
  │
  └── 否 → 协同过滤
      理由：无物品特征，需要从评分中自动发现隐含特征
      需推荐学习多少个隐含特征（n_features）
```

汇报：

> 🧠 **算法选择**
>
> 检测到：{有/无}物品特征
> 推荐算法：**{基于内容的推荐 / 协同过滤}**
> 原因：{说明}

#### 协同过滤额外步骤：推荐 n_features

> 推荐学习 **{n_features}** 个隐含特征。
> （推荐逻辑：默认 `min(10, n_items//2, n_users//2)`，评分稀疏则减少）
>
> - 更多特征 → 捕获更多细节，但可能过拟合
> - 更少特征 → 模型更简单稳定
>
> 是否接受？（回复"可以"确认，或输入新值如 `5`）

---

### 步骤 4：参数推荐（lambda、alpha、iters）

```
lambda 推荐逻辑（正则化参数）：
  - 默认：lambda = 1.0
  - 评分稀疏度 > 0.8 → lambda = 3.0（强正则化防过拟合）
  - 评分稀疏度 < 0.3 → lambda = 0.1（弱正则化）
  - 有验证集 → 尝试 lambda ∈ {0, 0.01, 0.1, 1, 3, 10}，选验证误差最小的

alpha 推荐逻辑（学习率）：
  - 默认：alpha = 0.01
  - n_rated > 1000 → alpha = 0.003（更谨慎）
  - n_rated < 50 → alpha = 0.03（更大胆）

iters 推荐逻辑：
  - 默认：500
  - 如果参数少（只有 theta，内容推荐）→ 200
  - 如果参数多（x + theta，协同过滤）→ 500
```

汇报：

> ⚙️ **参数推荐**
>
> | 参数 | 推荐值 | 原因 |
> |------|--------|------|
> | lambda | {lambda} | {原因} |
> | alpha  | {alpha} | {原因} |
> | iters  | {iters} | {原因} |
>
> 是否需要调整？（回复"可以"确认，或输入修改如 `lambda=3, alpha=0.003`）

---

### 步骤 5：训练模型

执行步骤：

```
1. 均值归一化：Y_norm = Y - Y_mean（仅对已评分项）
2. 随机初始化参数（小随机值，对称破坏）
3. 向量化梯度下降
   - 每 50 轮打印一次代价 J
   - 若 J 上升 → alpha 自动减半重试
   - 若连续 10 轮 ΔJ < 1e-6 → 提前停止
4. 打印最终参数
```

汇报训练进度：

> 🏋️ **训练中...**
>
> 迭代   50/500 | J = 12.3456
> 迭代  100/500 | J = 8.9012
> 迭代  150/500 | J = 5.6789
> ...
> 迭代  500/500 | J = 1.2345
>
> ✅ 训练完成！最终 J = {final_J:.4f}，耗时 {elapsed:.1f}s

---

### 步骤 6：输出结果

#### 6a. 预测函数

```python
def predict(user_id, item_id):
    """预测用户 user_id 对物品 item_id 的评分"""
    return theta[:, user_id] @ x[:, item_id] + Y_mean[item_id]
```

#### 6b. 每用户 Top-N 推荐

> 📋 **推荐结果**
>
> 用户 {user1} 的推荐（Top-5 未评分物品）：
>
> | 排名 | 物品 | 已有评分 | 预测评分 |
> |------|------|---------|---------|
> | 1    | {item3} | —    | 4.8 |
> | 2    | {item5} | —    | 4.5 |
> | 3    | {item7} | 3.0  | 4.2 |
> | 4    | {item2} | —    | 4.0 |
> | 5    | {item8} | —    | 3.8 |
>
> ...（共 {n_users} 个用户的推荐）

#### 6c. 相似物品

> 🔗 **相似物品**
>
> {item1} 的相似物品（Top-5）：
>
> | 排名 | 物品 | ‖x 差‖² | 相似度 |
> |------|------|----------|--------|
> | 1    | {item3} | 0.12 | ★★★★★ |
> | 2    | {item5} | 0.45 | ★★★★☆ |
> | 3    | {item8} | 0.89 | ★★★☆☆ |
> | 4    | {item2} | 1.34 | ★★☆☆☆ |
> | 5    | {item6} | 2.01 | ★★☆☆☆ |

---

### 步骤 7：生成图表

生成一张多子图大图，保存为 `{filename_base}_recommender.png`：

```
子图布局（3 行 1 列）：

  Row 1: 代价函数收敛曲线
    - 折线图，X=迭代次数, Y=J（代价）
    - 标注 lambda 值
    - 如果做了 lambda 搜索 → 多条曲线对比不同 lambda

  Row 2: 评分矩阵热力图
    - 行=物品, 列=用户
    - 颜色：实际评分（蓝色=低分, 红色=高分）
    - 白色/灰 = 未评分
    - 在未评分格上叠加预测值（白色文字）

  Row 3: 预测 vs 实际散点图
    - X=实际评分, Y=预测评分
    - 红色虚线：y=x（理想预测）
    - 标注 RMSE 和 MAE
```

Matplotlib 设置：
```python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

### 步骤 8：生成 .py 文件

文件名格式：`{prefix}_recommender_{YYYYMMDD}_{HHMMSS}.py`

生成的 `.py` 包含完整可运行代码：

1. 数据定义/加载
2. 均值归一化
3. 代价函数 + 梯度计算（向量化）
4. 参数设置
5. 梯度下降循环
6. `predict()` 预测函数
7. Top-N 推荐函数
8. 相似物品检索函数
9. 结果输出（推荐列表 + 相似物品）
10. 3 张子图的 matplotlib 代码
11. 完整中英文注释

**可脱离技能环境独立运行**，只需 NumPy + Matplotlib。

---

## 代码模板

### 模板 A：基于内容的推荐

适用于：用户提供了物品特征 X_features

```python
# -*- coding: utf-8 -*-
"""
{ALGORITHM}：{TITLE_TOPIC}
生成时间：{TIMESTAMP}
算法：基于内容的推荐（Content-based Filtering）
参数：lambda={LAMBDA}, alpha={ALPHA}, iters={ITERS}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import time

# ========== 1. 数据 ==========
# 评分矩阵（行=物品, 列=用户, 0=未评分）
Y = np.array({Y_DATA})
# 指示矩阵：R[i,j] = 1 表示用户 j 对物品 i 有评分
R = (Y > 0).astype(float)

n_items, n_users = Y.shape
item_names = {ITEM_NAMES}
user_names = {USER_NAMES}
title_topic = "{TITLE_TOPIC}"

# 物品特征（已提供）
X_given = np.array({X_FEATURES_DATA})  # shape: (n_features, n_items)
n_features = X_given.shape[0]

print("=" * 60)
print(f"推荐系统：{title_topic}")
print(f"算法：基于内容的推荐（Content-based Filtering）")
print(f"物品数: {n_items}, 用户数: {n_users}")
print(f"特征数: {n_features}（已提供）")
print(f"已评分: {int(np.sum(R))} 条")
print("=" * 60)

# ========== 2. 均值归一化 ==========
Y_mean = np.zeros(n_items)
for i in range(n_items):
    rated_indices = np.where(R[i, :] == 1)[0]
    if len(rated_indices) > 0:
        Y_mean[i] = np.mean(Y[i, rated_indices])
    else:
        Y_mean[i] = 0.0

Y_norm = Y.copy()
for i in range(n_items):
    for j in range(n_users):
        if R[i, j] == 1:
            Y_norm[i, j] = Y[i, j] - Y_mean[i]

print(f"\n均值归一化完成。Y_mean[:5]: {Y_mean[:5]}")

# ========== 3. 代价函数和梯度（基于内容） ==========
def cost_function_content(Y_norm, R, X, theta, lam):
    """内容推荐代价函数 J(theta)"""
    n_items, n_users = Y_norm.shape
    J = 0
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R
    J = 0.5 * np.sum(errors ** 2)
    J += (lam / 2) * np.sum(theta ** 2)
    return J

def gradient_content(Y_norm, R, X, theta, lam):
    """内容推荐梯度（向量化）"""
    n_items, n_users = Y_norm.shape
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R  # (n_items, n_users)
    # dJ/dtheta: X @ errors.T → (n_features, n_users)
    grad = X @ errors.T + lam * theta  # (n_features, n_users)
    return grad

# ========== 4. 参数 ==========
lam = {LAMBDA}
alpha = {ALPHA}
iters = {ITERS}

print(f"\n参数: lambda={lam}, alpha={alpha}, iters={iters}")

# ========== 5. 训练（梯度下降） ==========
# 随机初始化 theta
np.random.seed(42)
theta = np.random.randn(n_features, n_users) * 0.1

J_history = []
t_start = time.time()

for t in range(iters):
    J = cost_function_content(Y_norm, R, X_given, theta, lam)
    grad = gradient_content(Y_norm, R, X_given, theta, lam)
    theta -= alpha * grad
    J_history.append(J)
    
    if (t + 1) % 50 == 0 or t == 0:
        print(f"  迭代 {t+1:4d}/{iters} | J = {J:.6f}")
    
    # 收敛检测
    if t > 10 and abs(J_history[-1] - J_history[-2]) < 1e-6:
        print(f"  => 收敛（DeltaJ < 1e-6），提前停止于第 {t+1} 轮")
        break

elapsed = time.time() - t_start
print(f"\n训练完成！最终 J = {J_history[-1]:.6f}, 耗时 {elapsed:.1f}s")

X = X_given  # 内容推荐：特征即给定值

# ========== 6. 预测函数 ==========
def predict(user_id, item_id):
    """预测用户 user_id 对物品 item_id 的评分"""
    return theta[:, user_id] @ X[:, item_id] + Y_mean[item_id]

# ========== 7. Top-N 推荐 ==========
print("\n" + "=" * 60)
print("Top-N 推荐结果")
print("=" * 60)

for j in range(n_users):
    unrated = np.where(R[:, j] == 0)[0]
    if len(unrated) == 0:
        continue
    preds = np.array([predict(j, i) for i in unrated])
    top_idx = unrated[np.argsort(-preds)][:5]
    
    print(f"\n用户 {user_names[j]} 的推荐（Top-5）:")
    print(f"  {'排名':>4} | {'物品':>15} | {'预测评分':>8}")
    print("  " + "-" * 35)
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:4d} | {item_names[i]:>15} | {predict(j, i):8.2f}")

# ========== 8. 相似物品 ==========
print("\n" + "=" * 60)
print("相似物品")
print("=" * 60)

for i in range(n_items):
    if i >= 5:  # 只展示前 5 个物品的相似物品
        break
    distances = []
    for k in range(n_items):
        if k != i:
            dist = np.sum((X[:, i] - X[:, k]) ** 2)
            distances.append((k, dist))
    distances.sort(key=lambda x: x[1])
    
    print(f"\n{item_names[i]} 的相似物品:")
    print(f"  {'排名':>4} | {'物品':>15} | {'距离':>8} | {'相似度':>8}")
    print("  " + "-" * 45)
    for rank, (k, dist) in enumerate(distances[:5], 1):
        stars = "★" * max(1, int(5 * (1 - dist / max(d[1] for d in distances[:5] + [(0, 1)]))))
        print(f"  {rank:4d} | {item_names[k]:>15} | {dist:8.4f} | {stars}")

# ========== 9. 可视化 ==========
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Row 1: 代价函数收敛曲线
ax1 = axes[0]
ax1.plot(range(len(J_history)), J_history, 'b-', linewidth=1.5)
ax1.set_xlabel('迭代次数', fontsize=11)
ax1.set_ylabel('代价 J', fontsize=11)
ax1.set_title(f'代价函数收敛曲线 (lambda={lam})', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.annotate(f'初始 J={J_history[0]:.2f}', xy=(0, J_history[0]),
             xytext=(len(J_history)*0.1, J_history[0]), fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
ax1.annotate(f'最终 J={J_history[-1]:.4f}', xy=(len(J_history)-1, J_history[-1]),
             xytext=(len(J_history)*0.6, J_history[0]*0.5), fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))

# Row 2: 评分矩阵热力图
ax2 = axes[1]
masked_Y = np.where(R == 1, Y, np.nan)
im = ax2.imshow(masked_Y, cmap='RdBu_r', aspect='auto', vmin=Y[Y>0].min(), vmax=Y.max())
ax2.set_xticks(range(n_users))
ax2.set_xticklabels(user_names, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n_items))
ax2.set_yticklabels(item_names, fontsize=9)
ax2.set_title('评分矩阵（白=未评分, 数字=预测值）', fontsize=12, fontweight='bold')
# 在未评分格标注预测值
for i in range(n_items):
    for j in range(n_users):
        if R[i, j] == 0:
            pred_val = predict(j, i)
            ax2.text(j, i, f'{pred_val:.1f}', ha='center', va='center',
                     color='white' if abs(pred_val - 2.5) > 1 else 'black', fontsize=8)
        else:
            ax2.text(j, i, f'{Y[i, j]:.0f}', ha='center', va='center',
                     fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.8, label='rating')

# Row 3: 预测 vs 实际散点图
ax3 = axes[2]
actual_ratings = []
predicted_ratings = []
for j in range(n_users):
    for i in range(n_items):
        if R[i, j] == 1:
            actual_ratings.append(Y[i, j])
            predicted_ratings.append(predict(j, i))

actual_ratings = np.array(actual_ratings)
predicted_ratings = np.array(predicted_ratings)

rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
mae = np.mean(np.abs(predicted_ratings - actual_ratings))

ax3.scatter(actual_ratings, predicted_ratings, alpha=0.5, s=50, c='steelblue', edgecolors='white')
min_val, max_val = min(actual_ratings.min(), predicted_ratings.min()), max(actual_ratings.max(), predicted_ratings.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测 y=x')
ax3.set_xlabel('实际评分', fontsize=11)
ax3.set_ylabel('预测评分', fontsize=11)
ax3.set_title('预测 vs 实际评分', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.text(0.95, 0.1, f'RMSE = {rmse:.3f}\nMAE = {mae:.3f}',
         transform=ax3.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'推荐系统：{title_topic}', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('{FILENAME_BASE}_recommender.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {FILENAME_BASE}_recommender.png")
plt.close()

print(f"\nRMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
```

### 模板 B：协同过滤

适用于：用户没有提供物品特征，仅有评分矩阵

模板 B 的结构与模板 A **90% 相同**，仅在以下部分不同：

```python
# ========== 1. 数据（修改） ==========
# 无物品特征

# ========== 3. 代价函数和梯度（协同过滤替代内容推荐） ==========
def cost_function_cf(Y_norm, R, X, theta, lam):
    """协同过滤代价函数 J(X, theta)"""
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R
    J = 0.5 * np.sum(errors ** 2)
    J += (lam / 2) * np.sum(X ** 2)
    J += (lam / 2) * np.sum(theta ** 2)
    return J

def gradient_cf(Y_norm, R, X, theta, lam):
    """协同过滤梯度（向量化）"""
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R  # (n_items, n_users)
    # dJ/dX: theta @ errors.T → (n_features, n_items)
    X_grad = theta @ errors.T + lam * X  # (n_features, n_items)
    # dJ/dtheta: X @ errors → (n_features, n_users)
    theta_grad = X @ errors + lam * theta  # (n_features, n_users)
    return X_grad, theta_grad

# ========== 4. 参数（增加 n_features） ==========
n_features = {N_FEATURES}  # 隐含特征数（学出来的）
lam = {LAMBDA}
alpha = {ALPHA}
iters = {ITERS}

# ========== 5. 训练（同时学习 X 和 theta） ==========
# 随机初始化 X 和 theta（小值确保对称破坏）
np.random.seed(42)
X = np.random.randn(n_features, n_items) * 0.1
theta = np.random.randn(n_features, n_users) * 0.1

J_history = []
t_start = time.time()

for t in range(iters):
    J = cost_function_cf(Y_norm, R, X, theta, lam)
    X_grad, theta_grad = gradient_cf(Y_norm, R, X, theta, lam)
    X -= alpha * X_grad
    theta -= alpha * theta_grad
    J_history.append(J)
    
    if (t + 1) % 50 == 0 or t == 0:
        print(f"  迭代 {t+1:4d}/{iters} | J = {J:.6f}")
    
    # 收敛检测
    if t > 10 and abs(J_history[-1] - J_history[-2]) < 1e-6:
        print(f"  => 收敛（DeltaJ < 1e-6），提前停止于第 {t+1} 轮")
        break

elapsed = time.time() - t_start
print(f"\n训练完成！最终 J = {J_history[-1]:.6f}, 耗时 {elapsed:.1f}s")

# 注意：后续预测函数和 Top-N 推荐代码与模板 A 完全相同
# predict() 函数共享（因为都基于 theta @ X + Y_mean 的形式）

# 相似物品检索也相同（都是基于学习到的 X 特征向量）

# 可视化部分也相同

# 修改标题和说明：
# - algorithm_name = "协同过滤（Collaborative Filtering）"
# - algorithm_detail = f"协同过滤, n_features={n_features}, lambda={lam}"
```

> **注意**：模板 B 完整代码在生成时基于模板 A + 上述差异拼接而成，避免重复维护。

---

## 占位符说明

| 占位符 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `{Y_DATA}` | 数组 | 评分矩阵 | `np.array([[5,4,0],[3,0,0],[0,4,5]])` |
| `{N_ITEMS}` | 整数 | 物品数 | `5` |
| `{N_USERS}` | 整数 | 用户数 | `4` |
| `{N_FEATURES}` | 整数 | 特征数 | `3` |
| `{N_RATED}` | 整数 | 已评分条目数 | `12` |
| `{SPARSITY}` | 字符串 | 稀疏度百分比 | `"40.0%"` |
| `{X_FEATURES_DATA}` | 数组/None | 物品特征矩阵（内容推荐时有值） | `np.array([[0.9, 0.1],...])` 或 `None` |
| `{ITEM_NAMES}` | 列表 | 物品名 | `["The Matrix", "Toy Story", ...]` |
| `{USER_NAMES}` | 列表 | 用户名 | `["Alice", "Bob", "Carol", "Dave"]` |
| `{ALGORITHM}` | 字符串 | 算法名称 | `"基于内容的推荐"` 或 `"协同过滤"` |
| `{ALGORITHM_REASON}` | 字符串 | 选择原因 | `"已提供物品特征，直接学习用户偏好"` |
| `{LAMBDA}` | 浮点数 | 正则化参数 | `1.0` |
| `{LAMBDA_METHOD}` | 字符串 | lambda 选择方法 | `"默认值"` 或 `"验证集最优(lambda=0.1)"` |
| `{ALPHA}` | 浮点数 | 学习率 | `0.01` |
| `{ITERS}` | 整数 | 迭代次数 | `500` |
| `{Y_MEAN}` | 数组 | 每物品平均评分 | `[3.7, 4.1, 2.8, 3.5, 4.0]` |
| `{FINAL_J}` | 浮点数 | 最终代价 | `1.2345` |
| `{X_LEARNED}` | 数组 | 学习到的物品特征（仅 CF） | `np.array([[0.9, 0.1],...])` |
| `{THETA_LEARNED}` | 数组 | 学习到的用户参数 | `np.array([[2.1, 4.3,...],...])` |
| `{TOP_RECOMMENDATIONS}` | 代码块 | Top-N 推荐结果 | 循环打印每用户推荐 |
| `{SIMILAR_ITEMS}` | 代码块 | 相似物品结果 | 循环打印每物品相似项 |
| `{RMSE}` | 浮点数 | 均方根误差 | `0.345` |
| `{MAE}` | 浮点数 | 平均绝对误差 | `0.267` |
| `{FILENAME_BASE}` | 字符串 | 输出文件前缀 | `"movie_recommender_20260802_143000"` |
| `{TITLE_TOPIC}` | 字符串 | 推荐主题 | `"电影推荐系统"` |
| `{DATA_SOURCE}` | 字符串 | 数据来源说明 | `"从 ratings.csv 加载"` |
| `{TIMESTAMP}` | 字符串 | 生成时间 | `"2026-08-02 14:30:00"` |

---

## 关键注意事项

### 编码与兼容性

- **Windows GBK 终端**：打印语句中避免使用 Unicode 特殊字符
  - `θ` → `theta`，`λ` → `lambda`，`μ` → `mu`，`²` → `^2`
- **无头环境**：使用 `matplotlib.use('Agg')`，`.py` 文件末尾加 `plt.close()`
- **中文字体**：`rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']`

### 算法实现

- **向量化优先**：梯度计算使用矩阵乘法而非 for 循环，效率高且匹配吴恩达课程风格
- **对称破坏**：随机初始化 X 和 theta 用小值（`randn * 0.1`），确保不是全零
- **均值归一化必须**：训练用 `Y_norm`，预测加回 `Y_mean`
- **新用户处理**：theta 初始化为随机小值 → 预测 ≈ Y_mean（相当于全局平均推荐）

### 参数选择

- `np.random.seed(42)` 保证可复现
- lambda 候选集：`{0, 0.01, 0.1, 1, 3, 10}`
- alpha 默认 0.01，训练时若 J 上升则自动减半重试（最多 3 次）
- 收敛判断：`|Delta J| < 1e-6` 连续 10 轮

### 输出文件

- `.py` 文件命名：`{prefix}_recommender_{YYYYMMDD}_{HHMMSS}.py`
- `.png` 文件命名：同前缀 + `_recommender.png`
- 保存到当前工作目录

---

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|---------|
| [references/collaborative-filtering.md](references/collaborative-filtering.md) | 协同过滤完整实现（代价+梯度+训练+向量化） | 走路线 B 时 |
| [references/content-based-filtering.md](references/content-based-filtering.md) | 基于内容推荐实现（给定特征学 theta） | 走路线 A 时 |
| [references/mean-normalization.md](references/mean-normalization.md) | 均值归一化原理+代码+冷启动 | 总是（两种路线都需要） |
| [references/similar-items.md](references/similar-items.md) | 相似物品检索方法（距离+Top-K+可视化） | 总是（需要相似物品时） |
