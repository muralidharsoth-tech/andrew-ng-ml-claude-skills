# 高斯分布变换参考

## 概述

当数据不符合高斯分布（偏度 |skewness| ≥ 1）时，通过数学变换让数据更接近正态分布，从而提高异常检测的准确性。

本参考文件包括：
1. 判断标准（什么时候需要变换）
2. 变换方法选择指南
3. 完整变换代码模板

---

## 判断标准

```python
def compute_skewness(x):
    """计算偏度 — 衡量分布的对称性"""
    m = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    return np.sum(((x - mu) / sigma) ** 3) / m

# 判断
if abs(skewness) < 1:
    # 大致符合高斯分布，不需要变换
elif skewness > 1:
    # 右偏（长尾在右边）→ 需要压缩大值
else:  # skewness < -1
    # 左偏（长尾在左边）→ 需要拉伸大值
```

---

## 变换方法指南

### 右偏数据（偏度 > 0，大多数实际场景）

| 方法 | 公式 | 适用场景 | 强度 |
|------|------|---------|------|
| log 变换 | `x' = log(x + offset)` | 严重右偏，值全为正 | 强 |
| sqrt 变换 | `x' = sqrt(x + offset)` | 中度右偏，值非负 | 中 |
| Box-Cox 变换 | `x' = (x^λ - 1) / λ` | 自动搜索最优 λ | 自适应 |

### 左偏数据（偏度 < 0，较少见）

| 方法 | 公式 | 适用场景 | 强度 |
|------|------|---------|------|
| 平方变换 | `x' = x²` | 轻度左偏 | 中 |
| 指数变换 | `x' = exp(x / c)` | 严重左偏 | 强 |
| Box-Cox 变换 | `x' = (x^λ - 1) / λ` | 自动搜索最优 λ | 自适应 |

---

## 变换代码模板

### 完整变换尝试函数

```python
def try_transforms(x, feature_name, verbose=True):
    """
    对特征 x 尝试所有可能的变换，返回第一个使 |skewness| < 1 的结果。
    
    参数:
        x: 原始数据 (1D array)
        feature_name: 特征名（用于日志）
        verbose: 是否打印尝试过程
    
    返回:
        x_transformed: 变换后的数据
        method: 使用的变换方法名
        params: 变换参数（如 offset, lambda）
        skew_before: 变换前偏度
        skew_after: 变换后偏度
    """
    m = len(x)
    skew_before = compute_skewness(x)
    
    if verbose:
        print(f"\n  变换特征 '{feature_name}'（原始偏度={skew_before:+.3f}）")
    
    # 已经符合高斯 → 不需要变换
    if abs(skew_before) < 1:
        if verbose:
            print(f"    ✅ 已符合高斯分布，无需变换")
        return x, 'none', {}, skew_before, skew_before
    
    results = []
    
    # ---- 右偏处理方法 ----
    if skew_before > 0:
        # 方法1: log 变换
        offset = max(0, -np.min(x)) + 1e-8  # 确保全部为正
        x_log = np.log(x + offset)
        sk_log = compute_skewness(x_log)
        results.append(('log', x_log, sk_log, {'offset': offset}))
        if verbose:
            print(f"    尝试 log(x+{offset:.2e}): 偏度={sk_log:+.3f} {'✅' if abs(sk_log)<1 else '❌'}")
        if abs(sk_log) < 1:
            return x_log, 'log', {'offset': offset}, skew_before, sk_log
        
        # 方法2: sqrt 变换
        offset2 = max(0, -np.min(x)) + 1e-8
        x_sqrt = np.sqrt(x + offset2)
        sk_sqrt = compute_skewness(x_sqrt)
        results.append(('sqrt', x_sqrt, sk_sqrt, {'offset': offset2}))
        if verbose:
            print(f"    尝试 sqrt(x+{offset2:.2e}): 偏度={sk_sqrt:+.3f} {'✅' if abs(sk_sqrt)<1 else '❌'}")
        if abs(sk_sqrt) < 1:
            return x_sqrt, 'sqrt', {'offset': offset2}, skew_before, sk_sqrt
    
    else:  # 左偏（skew_before < 0）
        # 方法1: 平方变换
        x_sq = x ** 2
        sk_sq = compute_skewness(x_sq)
        results.append(('square', x_sq, sk_sq, {}))
        if verbose:
            print(f"    尝试 x^2: 偏度={sk_sq:+.3f} {'✅' if abs(sk_sq)<1 else '❌'}")
        if abs(sk_sq) < 1:
            return x_sq, 'square', {}, skew_before, sk_sq
        
        # 方法2: 指数变换
        c = np.std(x) * 2  # 缩放因子
        x_exp = np.exp(x / (c + 1e-8))
        sk_exp = compute_skewness(x_exp)
        results.append(('exp', x_exp, sk_exp, {'c': c}))
        if verbose:
            print(f"    尝试 exp(x/{c:.2f}): 偏度={sk_exp:+.3f} {'✅' if abs(sk_exp)<1 else '❌'}")
        if abs(sk_exp) < 1:
            return x_exp, 'exp', {'c': c}, skew_before, sk_exp
    
    # 方法3（通用）: Box-Cox 变换
    # 手动实现简化版 Box-Cox：搜索最优 λ
    x_positive = x - np.min(x) + 1e-8  # 确保全正
    lambda_candidates = np.linspace(-2, 2, 41)  # 40 个候选
    best_lambda, best_sk = 0, float('inf')
    x_best = x
    
    for lam in lambda_candidates:
        if abs(lam) < 1e-8:
            x_boxcox = np.log(x_positive)  # λ=0 等价于 log
        else:
            x_boxcox = (x_positive ** lam - 1) / lam
        sk = compute_skewness(x_boxcox)
        if abs(sk) < abs(best_sk):
            best_sk = sk
            best_lambda = lam
            x_best = x_boxcox
    
    if verbose:
        print(f"    尝试 Box-Cox(lambda={best_lambda:.2f}): 偏度={best_sk:+.3f} {'✅' if abs(best_sk)<1 else '❌'}")
    
    if abs(best_sk) < 1:
        return x_best, 'boxcox', {'lambda': best_lambda}, skew_before, best_sk
    
    # 所有方法都失败 → 选偏度绝对值最小的
    best_result = min(results, key=lambda r: abs(r[2]))
    if verbose:
        print(f"    ⚠️ 所有变换尝试后仍未达到 |skewness|<1")
        print(f"    将使用 {best_result[0]} 变换（偏度={best_result[2]:+.3f}），结果仅供参考")
    
    return best_result[1], best_result[0], best_result[3], skew_before, best_result[2]
```

### 使用示例

```python
# 对每个特征执行变换
transform_records = []  # 记录变换详情
X = X_raw.copy()

for j in range(n):
    x_trans, method, params, sk_before, sk_after = try_transforms(
        X_raw[:, j], feature_names[j]
    )
    X[:, j] = x_trans
    if method != 'none':
        transform_records.append({
            'feature': feature_names[j],
            'method': method,
            'params': params,
            'skew_before': sk_before,
            'skew_after': sk_after,
        })
        # 打印记录
        desc = f"{feature_names[j]}: {method}变换, 偏度 {sk_before:+.3f} -> {sk_after:+.3f}"
        print(f"  ✅ {desc}")
        transform_desc.append(desc)
```

### 变换记录示例输出

```
变换记录:
  mem_usage:   log变换,    偏度 +2.50 -> +0.25
  disk_io:     sqrt变换,   偏度 +1.83 -> +0.61
  cpu_temp:    boxcox变换, 偏度 -1.45 -> -0.12 (lambda=1.52)
```

---

## 注意事项

1. **offset 的选择**：确保变换后的值域合法（log 必须全正，sqrt 必须非负）
2. **Box-Cox 限制**：要求输入全为正，需要先做平移
3. **变换可逆性**：检测到的异常需要用**变换后的数据空间**来解释，如需在原始空间解释，需要逆变换
4. **保守原则**：如果所有方法都无法让偏度降到 |1| 以下，选效果最好的那个，并在输出中标注警告
5. **拟合数据**：μ 和 σ² 必须在**变换后的数据**上计算，异常标记也基于变换后的 p(x)
