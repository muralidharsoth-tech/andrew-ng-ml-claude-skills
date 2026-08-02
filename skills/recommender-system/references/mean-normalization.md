# 均值归一化参考

> 📚 隶属于 `recommender-system` 技能。在步骤 5（训练模型前）始终调用。

## 概述

均值归一化（Mean Normalization）是推荐系统中**必须**做的预处理步骤。它的核心作用是消除评分偏置，解决**新用户冷启动**问题。

### 为什么必须做？

**场景**：有一个新用户 Eve，她刚注册，还没有任何评分记录。

如果不做均值归一化：
- Eve 的 theta 无法更新（因为没有误差梯度）
- theta 保持初始值（随机小值≈0）
- 预测：theta^T·x ≈ 0 对所有物品 → **全都预测 0 分**（完全错误的推荐）

如果做均值归一化：
- 训练时用 `Y_norm = Y - Y_mean`（已经减去每物品的平均分）
- 预测时加回来：`pred = theta^T·x + Y_mean[i]`
- Eve 的 theta≈0 → `pred ≈ Y_mean[i]` → 预测为**每物品的平均分**
- 这至少是"合理的保守推荐"

---

## 数学原理

```
原始：Y[i,j] = theta^(j)^T · x^(i)

均值归一化：
  Y_mean[i] = 物品 i 的所有已评分用户的平均分
  Y_norm[i,j] = Y[i,j] - Y_mean[i]   （仅对有评分的项）
  
训练：学习 theta 和 x 来拟合 Y_norm
预测：theta^(j)^T · x^(i) + Y_mean[i]
```

### 为什么这样有效？

- `Y_mean[i]` 是物品 i 的"基准分"（不依赖任何用户特征）
- `theta^(j)^T · x^(i)` 捕获用户 j 对物品 i 的"个性化偏差"
- 新用户 theta≈0 → 只输出基准分（合理的默认推荐）
- 老用户 theta 已学到 → 输出基准分 + 个性化调整

---

## 代码实现

### 基础版

```python
def mean_normalize(Y, R):
    """
    均值归一化
    
    参数:
        Y: 评分矩阵 (n_items, n_users), 0 表示未评分
        R: 指示矩阵 (n_items, n_users), 1=已评分, 0=未评分
    
    返回:
        Y_norm: 归一化后的评分矩阵
        Y_mean: 每物品的平均评分 (n_items,)
    """
    n_items, n_users = Y.shape
    Y_mean = np.zeros(n_items)
    Y_norm = np.zeros_like(Y, dtype=float)
    
    for i in range(n_items):
        # 找到所有给物品 i 评过分的用户
        rated_indices = np.where(R[i, :] == 1)[0]
        
        if len(rated_indices) > 0:
            Y_mean[i] = np.mean(Y[i, rated_indices])
            # 对已评分项做归一化
            for j in rated_indices:
                Y_norm[i, j] = Y[i, j] - Y_mean[i]
        else:
            # 该物品无人评分（罕见），均值设为全局平均
            Y_mean[i] = np.mean(Y[R == 1]) if np.sum(R) > 0 else 0.0
    
    return Y_norm, Y_mean
```

### 向量化版（更快）

```python
def mean_normalize_vectorized(Y, R):
    """
    均值归一化（向量化版本）
    """
    n_items, n_users = Y.shape
    
    # 每物品的评分总和 / 评分数 = 均值
    # 注意：只对有评分的项求和
    Y_sum = np.sum(Y * R, axis=1)        # (n_items,) — 每物品总分
    n_ratings = np.sum(R, axis=1)        # (n_items,) — 每物品评分数
    
    # 避免除零
    n_ratings_safe = np.where(n_ratings > 0, n_ratings, 1)
    Y_mean = Y_sum / n_ratings_safe
    
    # 对没有评分的物品，均值设为 0
    Y_mean[n_ratings == 0] = 0.0
    
    # 归一化：Y_norm[i,j] = Y[i,j] - Y_mean[i]（仅对 R[i,j]==1 的项）
    Y_mean_expanded = Y_mean.reshape(-1, 1)  # (n_items, 1) 方便广播
    Y_norm = (Y - Y_mean_expanded) * R  # 未评分项保持为 0
    
    return Y_norm, Y_mean
```

---

## 完整使用示例

```python
# ===== 1. 原始数据 =====
Y = np.array([
    [5, 4, 0, 0],   # 电影1：Alice=5, Bob=4
    [3, 0, 0, 4],   # 电影2：Alice=3, Dave=4
    [0, 4, 5, 0],   # 电影3：Bob=4, Carol=5
    [0, 0, 5, 4],   # 电影4：Carol=5, Dave=4
])

R = (Y > 0).astype(float)

# ===== 2. 均值归一化 =====
Y_norm, Y_mean = mean_normalize_vectorized(Y, R)

print("Y_mean (每物品平均分):")
print(Y_mean)
# [4.5, 3.5, 4.5, 4.5]

print("\nY_norm (归一化后，训练用):")
print(Y_norm)
# [[ 0.5   -0.5   0.    0.  ]
#  [-0.5    0.    0.    0.5 ]
#  [ 0.    -0.5   0.5   0.  ]
#  [ 0.     0.    0.5  -0.5 ]]

# ===== 3. 训练（用 Y_norm）=====
# ... 协同过滤 / 内容推荐训练 ...

# ===== 4. 预测（加回 Y_mean）=====
def predict(user_id, item_id, X, theta, Y_mean):
    return theta[:, user_id] @ X[:, item_id] + Y_mean[item_id]

# 新用户 Eve (user_id=4, theta≈初始值≈0)
# 对所有物品的预测 ≈ Y_mean
print("\n新用户 Eve 的预测:")
for i in range(len(Y_mean)):
    # theta 近似为 0 → prediction ≈ Y_mean[i]
    print(f"  物品 {i}: {Y_mean[i]:.1f}")  # 输出每物品的平均分
```

---

## 新用户冷启动的处理

### 场景 1：训练集中就有新用户

```
Y 中某用户列全为 0（全部未评分）
→ R[:, j] 全为 0
→ 训练过程中该用户的 theta 完全不更新
→ theta[:, j] 保持初始值（≈ 0）
→ predict() ≈ Y_mean[i]（自动输出基准分）
→ ✅ 正确处理！
```

### 场景 2：训练后新用户加入

```python
# 新用户 Eve 还没有任何评分
# 直接创建新的 theta 列（初始化为 0）
theta_new_user = np.zeros(n_features)

# 预测 = theta_new_user @ X[:, item_id] + Y_mean[item_id]
#      = 0 + Y_mean[item_id]
#      = 物品的平均分
# → ✅ 合理的"保守推荐"
```

### 场景 3：新用户给了几个评分后

```python
# Eve 评了 3 部电影 → 可以微调她的 theta
# 固定 X 不变，只对 Eve 的 theta 做梯度下降
for t in range(iters):
    theta_eve -= alpha * gradient_for_user(theta_eve, eve_ratings, X, lam)

# 现在预测包含了 Eve 的个性化偏好
```

---

## 关键注意事项

### 1. 只对已评分项做归一化

```python
# ✅ 正确：只对 R[i,j]==1 的项做减法
Y_norm = (Y - Y_mean.reshape(-1, 1)) * R

# ❌ 错误：对未评分项也做了归一化
Y_norm = Y - Y_mean.reshape(-1, 1)  # 未评分项变成了负的 Y_mean！
```

### 2. 均值在预测时必须加回来

```python
# 训练时的代价计算用的是 Y_norm
# 预测时必须加回 Y_mean
pred = theta[:, j] @ X[:, i] + Y_mean[i]  # ✅
pred = theta[:, j] @ X[:, i]               # ❌ 少了 Y_mean！
```

### 3. 无评分物品的处理

```python
# 如果某个物品完全没有评分
if n_ratings[i] == 0:
    Y_mean[i] = global_mean  # 用全局平均分代替
```

### 4. 与 z-score 标准化的区别

| | 均值归一化 | z-score 标准化 |
|------|-----------|---------------|
| 公式 | `x' = x - μ` | `x' = (x - μ) / σ` |
| 目的 | 消除偏置 | 统一量纲 |
| 逆转 | `x = x' + μ` | `x = x'·σ + μ` |
| 用于 | 推荐系统评分 | 一般特征预处理 |
