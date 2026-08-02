# 相似物品检索参考

> 📚 隶属于 `recommender-system` 技能。在步骤 6c（输出相似物品）时调用。

## 概述

推荐系统的一个核心功能是**找相似物品**——给定一个物品，找出和它最相似的其他物品。这基于一个简单的思想：如果两个物品的**特征向量很接近**，它们就是相似的。

### 为什么有用？

- **推荐解释**："因为你喜欢《The Matrix》，推荐《Inception》——它们很相似"
- **替代品推荐**：某商品缺货时，推荐最相似的商品
- **多样性发现**：从相似物品列表中排除用户已看过的，推荐新的

---

## 核心原理

### 距离度量

协同过滤或内容推荐训练完毕后，每个物品 i 都有一个特征向量 x^(i)（n_features 维）。

物品 i 和物品 k 的相似度 = 它们的特征向量之间的距离：

```
dist(i, k) = ‖x^(i) - x^(k)‖²
           = Σₐ (x_a^(i) - x_a^(k))²
```

**距离越小 → 越相似。**

### 为什么是 Euclidean 距离？

吴恩达课程推荐用 L2 距离：
- 计算简单：`np.sum((x[:, i] - x[:, k]) ** 2)`
- 对特征维度同等对待（所有特征同等重要）
- 如果某些特征更重要，可以先做特征加权

---

## 代码实现

### 基础版：计算一个物品的相似物品

```python
def find_similar_items(item_idx, X, top_k=5):
    """
    找出与 item_idx 最相似的 top_k 个物品
    
    参数:
        item_idx: 目标物品的索引
        X: 物品特征矩阵 (n_features, n_items)
        top_k: 返回前 K 个最相似的
    
    返回:
        similar_items: [(物品索引, 距离), ...] 按距离升序排列
    """
    n_items = X.shape[1]
    
    distances = []
    target_vec = X[:, item_idx]
    
    for k in range(n_items):
        if k == item_idx:
            continue  # 跳过自己
        # L2 距离的平方
        dist = np.sum((target_vec - X[:, k]) ** 2)
        distances.append((k, dist))
    
    # 按距离升序排列
    distances.sort(key=lambda x: x[1])
    
    return distances[:top_k]
```

### 批量版：计算所有物品的相似矩阵

```python
def compute_similarity_matrix(X):
    """
    计算所有物品对之间的相似度矩阵（向量化）
    
    参数:
        X: 物品特征矩阵 (n_features, n_items)
    
    返回:
        dist_matrix: 距离矩阵 (n_items, n_items)
            dist_matrix[i, k] = ‖x^(i) - x^(k)‖²
    """
    n_features, n_items = X.shape
    
    # 向量化计算所有物品对的距离
    # ‖x^(i) - x^(k)‖² = ‖x^(i)‖² + ‖x^(k)‖² - 2·x^(i)^T·x^(k)
    
    # X^T @ X → (n_items, n_items): dot products
    dot_products = X.T @ X  # (n_items, n_items)
    
    # ‖x^(i)‖² for each item
    norms = np.sum(X ** 2, axis=0)  # (n_items,)
    
    # dist^2 = ‖x^(i)‖² + ‖x^(k)‖² - 2·x^(i)^T·x^(k)
    dist_matrix = norms.reshape(-1, 1) + norms.reshape(1, -1) - 2 * dot_products
    
    # 对角线设为大值（自己到自己的距离不是"相似"）
    np.fill_diagonal(dist_matrix, np.inf)
    
    return dist_matrix


def get_top_similar_from_matrix(dist_matrix, top_k=5):
    """
    从预计算的距离矩阵中提取每物品的 Top-K 相似物品
    
    返回:
        top_similar: 列表的列表，top_similar[i] = [(k, dist), ...]
    """
    n_items = dist_matrix.shape[0]
    top_similar = []
    
    for i in range(n_items):
        # 找到距离最小的 top_k 个索引
        sorted_indices = np.argsort(dist_matrix[i, :])[:top_k]
        similar = [(int(k), float(dist_matrix[i, k])) for k in sorted_indices]
        top_similar.append(similar)
    
    return top_similar
```

### 完整使用示例

```python
# 训练完毕后，X 已知（无论是协同过滤学到的还是内容推荐的给定特征）

# 方法 1：逐个查询
item_idx = 0
similar = find_similar_items(item_idx, X, top_k=5)
print(f"\n与 {item_names[item_idx]} 最相似的物品:")
for rank, (k, dist) in enumerate(similar, 1):
    # 将距离转化为相似度分数（0~1）
    max_dist = similar[-1][1] + 1e-8  # 最大距离
    similarity = 1.0 - dist / max_dist
    stars = int(similarity * 5)
    print(f"  {rank}. {item_names[k]:>15} | dist={dist:.4f} | {'★'*stars}{'☆'*(5-stars)}")

# 方法 2：批量计算相似矩阵（物品多时更高效）
dist_matrix = compute_similarity_matrix(X)
top_similar = get_top_similar_from_matrix(dist_matrix, top_k=5)

for i in range(min(5, n_items)):  # 只展示前 5 个物品
    print(f"\n{item_names[i]} 的相似物品:")
    for rank, (k, dist) in enumerate(top_similar[i], 1):
        print(f"  {rank}. {item_names[k]:>15} | 距离={dist:.4f}")
```

---

## 将距离转化为相似度分数

方便展示和排序：

```python
def distance_to_similarity(distances):
    """
    将距离列表转化为 0~1 的相似度分数
    
    方法：similarity = 1 / (1 + dist)
    - dist = 0 → similarity = 1.0（完全相同）
    - dist = ∞ → similarity = 0.0（完全不同）
    """
    similarities = 1.0 / (1.0 + np.array(distances))
    return similarities


# 使用
dist = 0.5
sim = 1.0 / (1.0 + 0.5)  # = 0.667
print(f"距离={dist:.4f} → 相似度={sim:.3f}")
```

---

## 带类别约束的相似物品

有时我们想找"同类型中最相似的"：

```python
def find_similar_items_same_category(item_idx, X, item_categories, top_k=5):
    """
    只在与 item_idx 同类型的物品中找相似
    
    参数:
        item_categories: 每个物品的类别标签列表
    """
    target_category = item_categories[item_idx]
    same_category_indices = [k for k, cat in enumerate(item_categories) 
                             if cat == target_category and k != item_idx]
    
    distances = []
    target_vec = X[:, item_idx]
    
    for k in same_category_indices:
        dist = np.sum((target_vec - X[:, k]) ** 2)
        distances.append((k, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
```

---

## 可视化：相似度矩阵

```python
def plot_similarity_heatmap(X, item_names, top_k=None):
    """
    绘制物品相似度热力图
    """
    import matplotlib.pyplot as plt
    
    n_items = X.shape[1]
    if top_k is None:
        top_k = n_items
    
    # 计算距离矩阵
    dist_matrix = compute_similarity_matrix(X)
    
    # 转为相似度：sim = 1 / (1 + dist)
    sim_matrix = 1.0 / (1.0 + dist_matrix)
    np.fill_diagonal(sim_matrix, 1.0)  # 自己和自己 = 1.0
    
    # 只展示前 top_k 个物品
    sim_subset = sim_matrix[:top_k, :top_k]
    item_names_subset = item_names[:top_k]
    
    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.8), max(6, top_k * 0.6)))
    
    im = ax.imshow(sim_subset, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(item_names_subset)))
    ax.set_xticklabels(item_names_subset, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(item_names_subset)))
    ax.set_yticklabels(item_names_subset, fontsize=9)
    
    # 在格子上标注数值
    for i in range(len(item_names_subset)):
        for j in range(len(item_names_subset)):
            color = 'white' if sim_subset[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{sim_subset[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color=color)
    
    ax.set_title(f'物品相似度矩阵（Top {top_k}）', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='similarity')
    plt.tight_layout()
    plt.savefig('item_similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 关键注意事项

### 1. 特征归一化影响相似度

如果不同特征量纲不同（如"上映年份" vs "评分次数"），量纲大的特征会主导距离计算。建议：
- 训练前对特征做 z-score 标准化
- 或者用余弦相似度代替 Euclidean 距离（不受量纲影响）

### 2. 余弦相似度（替代 Euclidean）

```python
def cosine_similarity(x_i, x_k):
    """余弦相似度：-1 到 1，越大越相似"""
    return np.dot(x_i, x_k) / (np.linalg.norm(x_i) * np.linalg.norm(x_k) + 1e-8)

# 转为距离：dist = 1 - cosine_similarity
# cosine=1（完全相同）→ dist=0
# cosine=0（正交）→ dist=1
# cosine=-1（完全相反）→ dist=2
```

### 3. 排重

如果用户已经评过分/购买过的物品，在推荐相似物品时应排除：

```python
def recommend_similar_unrated(user_id, liked_item_idx, X, R, Y_mean, theta, top_k=5):
    """
    推荐与 liked_item_idx 相似但用户 {user_id} 还未评分的物品
    """
    similar = find_similar_items(liked_item_idx, X, top_k=len(X[0]))
    
    recommendations = []
    for k, dist in similar:
        if R[k, user_id] == 0:  # 只推荐未评分的
            pred_rating = theta[:, user_id] @ X[:, k] + Y_mean[k]
            recommendations.append((k, dist, pred_rating))
        if len(recommendations) >= top_k:
            break
    
    return recommendations
```

### 4. 特征数量对距离的影响

高维空间中 Euclidean 距离的区分度会下降（维度灾难）。如果 n_features > 20：
- 考虑降维（PCA）后再计算距离
- 或使用余弦相似度（对高维更鲁棒）
