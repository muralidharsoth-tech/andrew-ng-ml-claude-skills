# 基于内容的推荐参考

> 📚 隶属于 `recommender-system` 技能。在用户提供了物品特征时调用。

## 概述

基于内容的推荐（Content-based Filtering）利用物品自身的特征来学习用户偏好。优点是**收敛快、可解释性强**，因为特征本身有明确的含义（如电影类型、导演、主演等）。

### 核心思路

```
给定：评分矩阵 Y（n_items × n_users）
      物品特征 x^(i)（n_features 维，已提供）
目标：学习用户参数 theta^(j)（n_features 维）
使得：theta^(j)^T · x^(i) ≈ Y[i,j]（对已评分项）
```

### 适用条件

```
✅ 有用户-物品评分矩阵
✅ 有物品特征（如电影类型、关键词、属性等）
✅ 特征值最好是数值型（类别型需做 One-Hot 编码）
```

---

## 代价函数

### 完整公式

```
J(theta^(1),...,theta^(n_users))
  = ½ Σᵢⱼ R(i,j) · (theta^(j)^T · x^(i) - y^(i,j))²
  + (λ/2) Σⱼₖ (theta_k^(j))²
```

注意：与协同过滤不同，这里**只对 theta 做正则化**，不对 X 做（因为 X 是给定的，不是学习的）。

### 本质上就是多个线性回归

每个用户 j 有一个独立的线性回归：
- `x` = 物品特征（输入）
- `y` = 该用户对该物品的评分（输出）
- `theta^(j)` = 该用户的偏好向量（参数）

如果用户 j 对 5 个物品有评分 → 5 个训练样本 → 学出 theta^(j)。

---

## 梯度计算

### 公式

```
∂J/∂theta_k^(j) = Σᵢ R(i,j)·(theta^(j)^T·x^(i) - y^(i,j)) · x_k^(i) + λ·theta_k^(j)
```

### 向量化实现

```python
def cost_function_content(Y_norm, R, X, theta, lam):
    """
    内容推荐代价函数
    
    参数:
        Y_norm: 均值归一化后的评分矩阵 (n_items, n_users)
        R: 指示矩阵 (n_items, n_users)
        X: 物品特征矩阵 (n_features, n_items) — 给定的！
        theta: 用户参数矩阵 (n_features, n_users)
        lam: 正则化参数
    
    返回:
        J: 代价（标量）
    """
    n_items, n_users = Y_norm.shape
    
    # 预测所有评分
    predictions = X.T @ theta  # (n_items, n_users)
    
    # 只看已评分项的误差
    errors = (predictions - Y_norm) * R
    
    # 代价（只对 theta 正则化）
    J = 0.5 * np.sum(errors ** 2)
    J += (lam / 2) * np.sum(theta ** 2)
    
    return J


def gradient_content(Y_norm, R, X, theta, lam):
    """
    内容推荐梯度（向量化）
    
    返回:
        theta_grad: theta 的梯度 (n_features, n_users)
    """
    predictions = X.T @ theta  # (n_items, n_users)
    errors = (predictions - Y_norm) * R  # (n_items, n_users)
    
    # dJ/dtheta = X @ errors + lam * theta
    # X (n_features, n_items) @ errors (n_items, n_users) → (n_features, n_users)
    theta_grad = X @ errors + lam * theta
    
    return theta_grad
```

---

## 训练流程

### 完整梯度下降循环

```python
def train_content_based(Y_norm, R, X_given, lam=1.0, alpha=0.01, iters=200, verbose=True):
    """
    训练基于内容的推荐模型
    
    参数:
        Y_norm: 均值归一化后的评分矩阵 (n_items, n_users)
        R: 指示矩阵 (n_items, n_users)
        X_given: 物品特征（已提供）(n_features, n_items)
        lam: 正则化参数
        alpha: 学习率
        iters: 最大迭代次数
        verbose: 是否打印进度
    
    返回:
        theta: 学习到的用户参数 (n_features, n_users)
        J_history: 代价历史
    """
    n_features, n_items = X_given.shape
    n_users = Y_norm.shape[1]
    
    # 随机初始化 theta
    np.random.seed(42)
    theta = np.random.randn(n_features, n_users) * 0.1
    
    J_history = []
    
    for t in range(iters):
        # 计算代价和梯度
        J = cost_function_content(Y_norm, R, X_given, theta, lam)
        theta_grad = gradient_content(Y_norm, R, X_given, theta, lam)
        
        # 更新参数
        theta -= alpha * theta_grad
        
        J_history.append(J)
        
        # 打印进度
        if verbose and ((t + 1) % 50 == 0 or t == 0):
            print(f"  迭代 {t+1:4d}/{iters} | J = {J:.6f}")
        
        # 收敛检测
        if t > 10 and abs(J_history[-1] - J_history[-2]) < 1e-6:
            if verbose:
                print(f"  => 收敛（|Delta J| < 1e-6），提前停止于第 {t+1} 轮")
            break
    
    return theta, J_history
```

---

## 特征工程建议

### 数值型特征

直接使用，建议做 z-score 标准化：

```python
X_scaled = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
```

### 类别型特征

使用 One-Hot 编码：

```python
# 电影类型：动作、爱情、喜剧
genres = ["action", "romance", "comedy", "action", "romance"]
# → One-Hot
from sklearn.preprocessing import OneHotEncoder
# 或手动：
unique_genres = list(set(genres))
X_onehot = np.zeros((len(unique_genres), len(genres)))
for i, g in enumerate(genres):
    X_onehot[unique_genres.index(g), i] = 1.0
```

### 文本特征

提取关键词：
```python
# 电影描述中提取关键词
keywords = ["space", "robot", "love", "war", "future"]
# 每个电影的关键词 TF-IDF 权重
X_tfidf = np.array([...])  # n_keywords × n_items
```

### 特征拼接

将不同类型的特征拼接在一起：

```python
# X_genre: (3, n_items)  One-Hot 电影类型
# X_year:  (1, n_items)  上映年份（数值）
# X_tfidf: (5, n_items)  关键词 TF-IDF

X_combined = np.vstack([X_genre, X_year, X_tfidf])  # (9, n_items)
```

---

## 使用示例

```python
# 准备数据
Y_norm = ...  # 均值归一化后的评分
R = ...       # 指示矩阵
X = ...       # 物品特征矩阵（已提供，shape: n_features × n_items）

# 训练
theta, J_history = train_content_based(
    Y_norm, R, X, lam=1.0, alpha=0.01, iters=200
)

# 解读 theta
# theta[:, 0] = [0.8, -0.3, 1.2] → 用户0喜欢特征2（权重1.2），不喜欢特征1（权重-0.3）
for j in range(n_users):
    print(f"用户 {j} 偏好: {theta[:, j]}")

# 预测
def predict(user_id, item_id, X, theta, Y_mean):
    return theta[:, user_id] @ X[:, item_id] + Y_mean[item_id]
```

---

## 关键注意事项

### 1. 稀疏数据的正则化

如果某些用户的评分很少（< 3 条），theta 估计会很不稳定。此时应增大 lambda 以加强正则化：

```python
# 对评分少的用户加大正则化
n_ratings_per_user = np.sum(R, axis=0)  # 每用户评分条数
if np.min(n_ratings_per_user) < 3:
    lam = 3.0  # 增大 lambda
    print(f"⚠️ 有用户只有 {int(np.min(n_ratings_per_user))} 条评分，建议 lambda >= 3.0")
```

### 2. 收敛速度

内容推荐的参数比协同过滤少很多（只有 theta，没有 X），所以：
- 收敛更快（iters 可以设为 200 而非 500）
- 代价更稳定（不容易震荡）
- 对初始值不太敏感

### 3. 新物品冷启动

**内容推荐的独特优势**：新物品加入时，只要有特征，就可以直接预测！

```python
# 新电影《Inception》的特征
X_new_movie = np.array([0.8, 0.3, 0.9, 0.2, 0.6])  # (n_features,)

# 直接预测所有用户对新电影的评分
for j in range(n_users):
    rating = theta[:, j] @ X_new_movie + Y_mean_mean
    print(f"用户 {j} 预测给新电影打 {rating:.1f} 分")
```

协同过滤做不到这一点——新物品没有评分记录，无法学到它的特征向量。

### 4. 特征质量决定上限

基于内容推荐的效果**完全取决于特征质量**。如果特征不能区分物品间的差异，模型就学不到有用的偏好模式。建议：
- 特征数不宜太少（至少 3 个，否则学不到足够的信息）
- 特征间最好有区分度（如果所有电影的特征都一样，无法区分）
- 类别特征必须 One-Hot 编码（不能直接将"动作"=1, "爱情"=2 赋值）
