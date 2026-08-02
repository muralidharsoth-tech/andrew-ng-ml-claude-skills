# 多元高斯异常检测参考

## 概述

当特征之间存在**显著相关性**时（|r| > 0.6），一元高斯模型会把「协方差为零」强加给数据，导致异常检测的边界是对齐坐标轴的椭圆——如果数据有倾斜的关联模式，这会产生大量的漏报或误报。

**多元高斯模型**用协方差矩阵 Σ 捕获特征间的关联，生成的决策边界是**可以倾斜的椭圆**，更贴合实际数据的形状。

### 适用条件

```
✅ 特征间存在显著相关性（|r| > 0.6 的配对至少有一对）
✅ 样本数 > 特征数（m > n）— 否则 Σ 不可逆
✅ 数据大致服从多元正态分布
```

### 对比：一元 vs 多元

| 特性 | 一元高斯 | 多元高斯 |
|------|---------|---------|
| p(x) 公式 | Π_j p(x_j; μ_j, σ²_j) | 一个联合概率密度公式 |
| 捕获相关性 | ❌ 假设特征独立 | ✅ 通过 Σ 自动捕获 |
| 决策边界 | 对齐坐标轴的椭圆 | 可倾斜的椭圆 |
| 计算成本 | O(n) | O(n³)（矩阵求逆） |
| 数据要求 | m ≥ 1 即可 | **必须 m > n** |
| 参数数量 | 2n（n 个 μ + n 个 σ²） | n + n(n+1)/2（μ + Σ） |

---

## 核心公式

### 概率密度函数

```
p(x; μ, Σ) = 1 / ((2π)^(n/2) · |Σ|^(1/2))  ·  exp(-½ · (x-μ)ᵀ · Σ⁻¹ · (x-μ))
                └─── 归一化常数 ────┘           └────── 马氏距离 ──────┘
```

其中：
- `μ` = 均值向量，shape: (n,)
- `Σ` = 协方差矩阵，shape: (n, n)
- `|Σ|` = Σ 的行列式
- `(x-μ)ᵀ Σ⁻¹ (x-μ)` = **马氏距离**（Mahalanobis distance），衡量 x 在考虑了相关性后离中心的距离

### 为什么需要 m > n

Σ 是 n×n 的矩阵，需要可逆。如果 m ≤ n：
- Σ 的秩 ≤ m - 1 < n
- Σ 是奇异的（不可逆）
- `|Σ| = 0`，`Σ⁻¹` 不存在 → p(x) 公式无意义

数学直观：n 维空间需要至少 n+1 个点才能确定一个「体积」（行列式非零）。

---

## 代码实现

### 模型训练

```python
def train_multivariate_gaussian(X):
    """
    训练多元高斯模型
    
    参数:
        X: 训练数据，shape (m, n)
    
    返回:
        mu: 均值向量，shape (n,)
        Sigma: 协方差矩阵，shape (n, n)
    """
    m, n = X.shape
    
    # 均值向量
    mu = X.mean(axis=0)  # shape: (n,)
    
    # 协方差矩阵 — bias=True 用 1/m（匹配吴恩达课程）
    Sigma = np.cov(X.T, bias=True)  # shape: (n, n)
    
    # 小正则化防止奇异（数值安全网）
    Sigma += 1e-6 * np.eye(n)
    
    return mu, Sigma
```

### 计算 p(x) — log 空间版本

高维时 `(2π)^(n/2) · |Σ|^(1/2)` 会下溢到 0，必须用 log 空间：

```python
def multivariate_gaussian_log_p(X, mu, Sigma):
    """
    计算多元高斯 log-p(x)
    
    参数:
        X: 输入数据，shape (m, n)
        mu: 均值向量，shape (n,)
        Sigma: 协方差矩阵，shape (n, n)
    
    返回:
        log_p: log p(x)，shape (m,)
    """
    m, n = X.shape
    
    # 1. log|Sigma| — 用 slogdet 避免下溢
    sign, logdet = np.linalg.slogdet(Sigma)
    # sign 应为正（行列式为负意味着矩阵有问题）
    
    # 2. Sigma 的逆
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        # Σ 奇异 → 增大正则化重试
        print("⚠️ 协方差矩阵奇异，增大正则化重试...")
        Sigma += 1e-3 * np.eye(n)
        Sigma_inv = np.linalg.inv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
    
    # 3. 马氏距离：对每个样本算 (x_i - μ)^T Σ^{-1} (x_i - μ)
    diff = X - mu  # shape: (m, n)
    # diff @ Sigma_inv  → (m, n), 然后 * diff → (m, n), sum(axis=1) → (m,)
    mahalanobis = np.sum(diff @ Sigma_inv * diff, axis=1)
    
    # 4. log p(x)
    log_p = -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)
    
    return log_p


def multivariate_gaussian_p(X, mu, Sigma):
    """计算 p(x)（返回实际概率密度，可能接近 0）"""
    log_p = multivariate_gaussian_log_p(X, mu, Sigma)
    return np.exp(log_p)
```

### 完整使用示例

```python
# 训练
mu, Sigma = train_multivariate_gaussian(X)

print("均值向量 mu:")
for j in range(n):
    print(f"  {feature_names[j]}: {mu[j]:.4f}")
print(f"\n协方差矩阵 Sigma ({n}x{n}):")
print(Sigma)

# 计算 p(x)
log_p_vals = multivariate_gaussian_log_p(X, mu, Sigma)
p_vals = np.exp(log_p_vals)

# epsilon 选择（与一元高斯完全相同）
# ...（F1 或分位数法）

# 检测异常
anomaly_mask = p_vals < epsilon
anomaly_indices = np.where(anomaly_mask)[0]
print(f"检出 {len(anomaly_indices)} 个异常")
```

---

## 可视化：多元高斯等高线

二元时绘制倾斜椭圆等高线：

```python
if n == 2:
    # 生成网格
    margin = 1.0
    x1_grid = np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, 100)
    x2_grid = np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    
    # 计算网格上的 p(x)
    log_p_grid = multivariate_gaussian_log_p(grid_points, mu, Sigma)
    p_grid = np.exp(log_p_grid).reshape(X1.shape)
    
    # 绘制
    ax.scatter(X[normal_indices, 0], X[normal_indices, 1], 
               c='steelblue', alpha=0.6, label='正常')
    ax.scatter(X[anomaly_indices, 0], X[anomaly_indices, 1], 
               c='red', marker='X', s=120, label='异常')
    
    # 等高线（倾斜椭圆！与一元高斯的正椭圆不同）
    ax.contour(X1, X2, p_grid, levels=8, colors='gray', alpha=0.4, linewidths=0.8)
    # ε 阈值等高线
    ax.contour(X1, X2, p_grid, levels=[epsilon], colors='red', linewidths=2.5, 
               linestyles='--', label=f'epsilon={epsilon:.2e}')
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('多元高斯：特征空间 + 倾斜决策边界', fontweight='bold')
    ax.legend()
```

---

## 什么时候不用多元高斯

即使 m > n，以下情况也建议退回到一元高斯：

1. **相关性很弱**（所有 |r| < 0.6）：一元和多元几乎等效，没必要为微小的相关性付出 O(n³) 代价
2. **样本数仅略大于特征数**（如 m = 10, n = 8）：Σ 的估计不稳定，正则化依赖重
3. **计算资源紧张**：n 很大时矩阵求逆 O(n³) 可能很慢

---

## 关键注意事项

1. **协方差无偏估计**：`np.cov(X.T, bias=True)` 用 1/m（课程惯例），`bias=False` 用 1/(m-1)（统计标准）。对异常检测影响不大，但要保持一致
2. **数值稳定性**：永远用 `slogdet` 而非 `det`，用 log 空间计算
3. **正则化**：`Sigma += 1e-6 * I` 是最后的安全网，不要依赖它来解决 m ≤ n 的问题
4. **马氏距离**：多元高斯本质上用马氏距离替代了欧几里得距离
