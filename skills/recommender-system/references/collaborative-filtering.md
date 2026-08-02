# 协同过滤参考

> 📚 隶属于 `recommender-system` 技能。在用户未提供物品特征时调用。

## 概述

协同过滤（Collaborative Filtering）是推荐系统中最核心的算法。与基于内容的推荐不同，协同过滤**不需要物品特征**——它从用户评分数据中自动学习物品的隐含特征和用户的偏好参数。

### 核心思路

```
给定：评分矩阵 Y（n_items × n_users）
      指示矩阵 R（哪些有评分）
目标：自动学习物品特征 x^(i)（n_features 维）
      和用户参数 theta^(j)（n_features 维）
使得：theta^(j)^T · x^(i) ≈ Y[i,j]（对已评分项）
```

### 适用条件

```
✅ 拥有用户-物品评分矩阵
✅ 没有物品特征（或特征不全）
✅ 评分数据足够（建议 ≥ 20 条评分记录）
```

---

## 代价函数

### 完整公式

```
J(x^(1),...,x^(n_items), theta^(1),...,theta^(n_users))
  = ½ Σᵢⱼ R(i,j) · (theta^(j)^T · x^(i) - y^(i,j))²
  + (λ/2) Σᵢₖ (x_k^(i))²
  + (λ/2) Σⱼₖ (theta_k^(j))²
```

- 第一项：预测误差（只看已评分项）
- 第二项：物品特征的正则化
- 第三项：用户参数的正则化
- 两个正则化使用**同一个 λ**（吴恩达课程惯例）

### 与内容推荐代价函数的区别

| 方面 | 内容推荐 | 协同过滤 |
|------|---------|---------|
| 学习哪些参数 | 只学 theta | 同时学 X 和 theta |
| 正则化 | 只对 theta | 对 X 和 theta 都需要 |
| 参数总量 | n_features × n_users | n_features × (n_items + n_users) |

---

## 梯度计算（向量化）

### 向量化推导

```
predictions = X^T @ theta              # shape: (n_items, n_users)
errors = (predictions - Y_norm) * R    # shape: (n_items, n_users) — 只看已评分项

X_grad     = theta @ errors.T + λ·X    # shape: (n_features, n_items)
theta_grad = X @ errors       + λ·theta # shape: (n_features, n_users)
```

**为什么 X_grad 是 `theta @ errors.T`？**
- 对每个特征 k 和每个物品 i：∂J/∂x_k^(i) = Σⱼ R(i,j)·(θ^(j)ᵀx^(i) - y^(i,j)) · θ_k^(j) + λ·x_k^(i)
- `theta @ errors.T` 中，theta 的第 k 行乘以 errors.T 的第 i 列 = 对所有用户的加权求和

### 完整代码实现

```python
def cost_function_cf(Y_norm, R, X, theta, lam):
    """
    协同过滤代价函数
    
    参数:
        Y_norm: 均值归一化后的评分矩阵 (n_items, n_users)
        R: 指示矩阵 (n_items, n_users), 1=已评分
        X: 物品特征矩阵 (n_features, n_items)
        theta: 用户参数矩阵 (n_features, n_users)
        lam: 正则化参数
    
    返回:
        J: 代价（标量）
    """
    n_items, n_users = Y_norm.shape
    
    # 预测所有评分
    predictions = X.T @ theta  # (n_items, n_users)
    
    # 只看已评分项的误差
    errors = (predictions - Y_norm) * R  # (n_items, n_users)
    
    # 代价
    J = 0.5 * np.sum(errors ** 2)
    J += (lam / 2) * np.sum(X ** 2)
    J += (lam / 2) * np.sum(theta ** 2)
    
    return J


def gradient_cf(Y_norm, R, X, theta, lam):
    """
    协同过滤梯度（向量化）
    
    返回:
        X_grad: X 的梯度 (n_features, n_items)
        theta_grad: theta 的梯度 (n_features, n_users)
    """
    # 预测误差
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R  # (n_items, n_users)
    
    # X 的梯度：对每个物品的每个特征
    # theta (n_features, n_users) @ errors.T (n_users, n_items) → (n_features, n_items)
    X_grad = theta @ errors.T + lam * X
    
    # theta 的梯度：对每个用户的每个特征
    # X (n_features, n_items) @ errors (n_items, n_users) → (n_features, n_users)
    theta_grad = X @ errors + lam * theta
    
    return X_grad, theta_grad
```

---

## 训练流程

### 完整梯度下降循环

```python
def train_collaborative_filtering(Y_norm, R, n_features, lam=1.0, alpha=0.01, iters=500, verbose=True):
    """
    训练协同过滤模型
    
    参数:
        Y_norm: 均值归一化后的评分矩阵 (n_items, n_users)
        R: 指示矩阵 (n_items, n_users)
        n_features: 要学习的隐含特征数
        lam: 正则化参数
        alpha: 学习率
        iters: 最大迭代次数
        verbose: 是否打印进度
    
    返回:
        X: 学习到的物品特征 (n_features, n_items)
        theta: 学习到的用户参数 (n_features, n_users)
        J_history: 代价历史
    """
    n_items, n_users = Y_norm.shape
    
    # 随机初始化（小值确保对称破坏）
    np.random.seed(42)
    X = np.random.randn(n_features, n_items) * 0.1
    theta = np.random.randn(n_features, n_users) * 0.1
    
    J_history = []
    alpha_current = alpha
    retry_count = 0
    
    for t in range(iters):
        # 计算代价和梯度
        J = cost_function_cf(Y_norm, R, X, theta, lam)
        X_grad, theta_grad = gradient_cf(Y_norm, R, X, theta, lam)
        
        # 更新参数
        X -= alpha_current * X_grad
        theta -= alpha_current * theta_grad
        
        J_history.append(J)
        
        # 检测 J 是否上升 → 学习率过大
        if t > 0 and J > J_history[-2] * 1.1 and retry_count < 3:
            if verbose:
                print(f"  ⚠️ J 上升 (J={J:.4f}), alpha 从 {alpha_current} 减半到 {alpha_current/2}")
            alpha_current /= 2
            retry_count += 1
            # 回退参数
            X += alpha_current * X_grad
            theta += alpha_current * theta_grad
            J_history[-1] = J_history[-2]  # 重新用旧 J
            continue
        
        # 打印进度
        if verbose and ((t + 1) % 50 == 0 or t == 0):
            print(f"  迭代 {t+1:4d}/{iters} | J = {J:.6f}")
        
        # 收敛检测
        if t > 10 and abs(J_history[-1] - J_history[-2]) < 1e-6:
            if verbose:
                print(f"  => 收敛（|Delta J| < 1e-6），提前停止于第 {t+1} 轮")
            break
    
    return X, theta, J_history
```

### 使用示例

```python
# 假设已经做完均值归一化，得到 Y_norm, R

# 训练
n_features = 5  # 学习 5 个隐含特征
X, theta, J_history = train_collaborative_filtering(
    Y_norm, R, n_features, lam=1.0, alpha=0.01, iters=500
)

# 预测
def predict(user_id, item_id, X, theta, Y_mean):
    return theta[:, user_id] @ X[:, item_id] + Y_mean[item_id]

# 为新用户（user_id=3, 无任何评分）推荐
# theta 初始化为 ~0 → 预测 ≈ Y_mean[item_id]
print(f"新用户的推荐: {Y_mean}")
```

---

## 特征数 n_features 选择指南

| 数据规模 | 推荐 n_features | 原因 |
|---------|----------------|------|
| < 50 条评分 | 2-3 | 数据少，学不了太多特征 |
| 50-500 条评分 | 3-8 | 中等，5 是安全默认 |
| 500-5000 条评分 | 8-20 | 数据充足，可捕获更多模式 |
| > 5000 条评分 | 20-50 | 大型数据，可用更多特征 |

公式：`n_features = min(10, n_items // 2, n_users // 2)`

---

## 关键注意事项

### 1. 对称破坏（Symmetry Breaking）

**必须随机初始化 X 和 theta！** 如果初始化为全零，所有特征的梯度相同 → 更新后仍然相同 → 所有特征永远相等于零 → 模型学不到任何东西。

```python
# ✅ 正确：小随机值
X = np.random.randn(n_features, n_items) * 0.1
theta = np.random.randn(n_features, n_users) * 0.1

# ❌ 错误：全零 → 对称永不破坏
X = np.zeros((n_features, n_items))
theta = np.zeros((n_features, n_users))
```

### 2. 向量化 vs 循环

协同过滤的朴素实现需要三层嵌套 for 循环（物品 × 用户 × 特征），在数据稍大时极慢。向量化版本只需矩阵乘法，快 10-100 倍：

```python
# ❌ 循环版（慢）：
for j in range(n_users):
    for i in range(n_items):
        if R[i, j] == 1:
            pred = 0
            for k in range(n_features):
                pred += X[k, i] * theta[k, j]
            error = (pred - Y_norm[i, j])
            for k in range(n_features):
                X_grad[k, i] += error * theta[k, j]
                theta_grad[k, j] += error * X[k, i]

# ✅ 向量化版（快）：
predictions = X.T @ theta           # 替代层层 for
errors = (predictions - Y_norm) * R
X_grad = theta @ errors.T + lam * X
theta_grad = X @ errors + lam * theta
```

### 3. 代价函数不单调下降

协同过滤的代价函数**不是凸函数**（因为同时优化 X 和 θ）。可能陷入局部最优。实践中：
- 多次随机初始化（不同 seed），选 J 最小的
- 学习率不要太大（建议 0.01 或 0.003）
- 通常能找到足够好的解

### 4. 与内容推荐的对比切换

如果后来获得了物品特征，可以将协同过滤学到的 X 替换为给定特征，再微调 theta：

```python
# 从协同过滤切换到内容推荐
X_given = ...  # 新获得的物品特征
# 用 X_given 初始化，微调 theta
theta, J_history = train_content_based(Y_norm, R, X_given, theta_init, lam, alpha, iters)
```
