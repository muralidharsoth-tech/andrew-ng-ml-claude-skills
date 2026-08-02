# 构造组合特征参考

## 概述

当检测到特征间存在**显著相关性**（|r| > 0.6），但样本数 ≤ 特征数（m ≤ n），导致**无法使用多元高斯**（Σ 奇异不可逆），此时需要构造组合特征来捕获相关性，再对扩展后的特征集使用一元高斯。

### 核心思路

```
原始问题：
  x1 和 x2 高度正相关 (r=0.9)
  但 m=5, n=2 → 不能用多元高斯

解决方案：
  构造 x3 = x1 / x2（捕获相对关系）
  → 新特征集 = [x1, x2, x3]（n 从 2 变成 3）
  → 用一元高斯在新特征集上做异常检测
  → x3 捕获了 x1 和 x2 之间的关联模式
```

### 适用条件

```
✅ 特征间存在显著相关性（至少一对 |r| > 0.6）
✅ m ≤ n（无法使用多元高斯）
✅ 理解构造特征的含义
```

---

## 构造策略

根据相关性的**方向**和**类型**选择不同的构造方法：

### 策略 1：比值特征（Ratio）

**适用**：两个特征正相关（一个大致是另一个的倍数）

```python
# x_new = x_i / (x_j + eps)
x_ratio = X[:, i] / (X[:, j] + 1e-8)
```

**含义**：捕获比例关系。例如 CPU 温度和功耗正相关，比值特征表示「单位温度的功耗」——如果某样本比值异常大，说明同样的温度下功耗不正常地高。

### 策略 2：差值特征（Difference）

**适用**：两个特征强正相关且量纲相同

```python
# x_new = x_i - x_j
x_diff = X[:, i] - X[:, j]
```

**含义**：捕获偏离程度。例如两台服务器的网络流量强相关（互为镜像），差值特征表示「两台服务器的流量差」——正常情况下差值接近 0，差值突然变大说明某台异常。

### 策略 3：和特征（Sum）

**适用**：两个特征负相关（一升一降）

```python
# x_new = x_i + x_j
x_sum = X[:, i] + X[:, j]
```

**含义**：捕获总量关系。例如内存使用率和可用内存负相关（r≈-1），和特征 = 总内存——正常情况下大致恒定，突变说明总内存变了（可能内存泄漏）。

### 策略 4：PCA 风格组合（PCA-like）

**适用**：多对特征相关，不确定具体模式

```python
# 取相关系数最大的一对，构造加权组合
corr_ij = corr[i, j]
# 根据相关性符号选择组合方式
if corr_ij > 0:
    x_new = X[:, i] / (X[:, j] + 1e-8)  # 正相关 → 比值
else:
    x_new = X[:, i] + X[:, j]            # 负相关 → 和
```

---

## 代码模板

### 自动构造组合特征

```python
def construct_engineered_features(X, corr_matrix, feature_names, threshold=0.6):
    """
    检测高相关特征对，自动构造组合特征。
    
    参数:
        X: 特征矩阵 (m, n)
        corr_matrix: 相关系数矩阵 (n, n)
        feature_names: 特征名列表
        threshold: 相关性阈值
    
    返回:
        X_new: 扩展后的特征矩阵 (m, n + k)
        new_feature_names: 扩展后的特征名列表
        engineering_log: 构造日志（用于写入 reference）
    """
    m, n = X.shape
    new_features = []
    new_names = []
    engineering_log = []
    
    # 找高相关对（去重）
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    if not high_corr_pairs:
        print("未检测到显著相关的特征对，不需要构造组合特征。")
        return X, feature_names[:], []
    
    print(f"\n检测到 {len(high_corr_pairs)} 对显著相关特征（|r| > {threshold}）：")
    
    for i, j, r_val in high_corr_pairs:
        print(f"  {feature_names[i]} <-> {feature_names[j]}: r = {r_val:+.3f}")
        
        if r_val > 0:  # 正相关 → 比值
            ratio = X[:, i] / (X[:, j] + 1e-8)
            new_features.append(ratio)
            name = f"{feature_names[i]}/{feature_names[j]}"
            new_names.append(name)
            log_entry = f"构造比值特征 {name}（{feature_names[i]}与{feature_names[j]}正相关 r={r_val:+.2f}）"
            print(f"    → 构造: {name}")
            
        else:  # 负相关 → 和
            sum_feat = X[:, i] + X[:, j]
            new_features.append(sum_feat)
            name = f"{feature_names[i]}+{feature_names[j]}"
            new_names.append(name)
            log_entry = f"构造和特征 {name}（{feature_names[i]}与{feature_names[j]}负相关 r={r_val:+.2f}）"
            print(f"    → 构造: {name}")
        
        engineering_log.append(log_entry)
    
    # 拼接到原特征矩阵
    X_extended = np.column_stack([X] + new_features)
    extended_names = feature_names[:] + new_names
    
    print(f"\n特征从 {n} 维扩展到 {len(extended_names)} 维")
    print(f"新特征集: {extended_names}")
    
    return X_extended, extended_names, engineering_log


# ===== 使用 =====
X_extended, extended_names, engineering_log = construct_engineered_features(
    X, corr, feature_names
)
```

### 集成到主代码中

```python
# 在步骤 4（相关性分析）之后，步骤 5（模型训练）之前：

if use_multivariate:
    # 走多元高斯路线
    # ...（见 multivariate-gaussian.md）
else:
    # 一元高斯路线
    if len(high_corr_pairs) > 0:
        # m ≤ n 时：构造组合特征
        X_extended, extended_names, engineering_log = construct_engineered_features(
            X, corr, feature_names
        )
        # 将构造说明写入注释
        engineering_comment = "\n".join(f"# {log}" for log in engineering_log)
        
        # 之后在整个流程中使用 X_extended 替代 X
        X_for_model = X_extended
    else:
        X_for_model = X
        engineering_comment = "# 无显著相关性，未构造组合特征"
    
    # 用 X_for_model 训练一元高斯...
```

---

## 注意事项

### 1. 不要过度构造

每增加一个特征，就需要多估计 μ_j 和 σ²_j（增加 2 个参数）。数据本来就少（m ≤ n），构造太多特征会进一步稀释样本密度。建议：
- 每对高相关特征**最多构造 1 个**组合特征
- 如果已有 5+ 个特征，只对 |r| > 0.8 的最强相关对构造

### 2. 构造特征的分布检查

构造后的新特征**也需要检查是否符合高斯分布**。例如 `x1/x2` 可能反而更偏（比值分布通常右偏）。如果新特征不符合高斯，再次尝试变换：

```python
# 对新构造的特征也做高斯检验
for j in range(len(engineering_log)):
    new_idx = n + j  # 新特征在扩展矩阵中的索引
    x_new_feat = X_extended[:, new_idx]
    sk = compute_skewness(x_new_feat)
    if abs(sk) >= 1:
        print(f"  ⚠️ 新特征 {extended_names[new_idx]} 偏度={sk:+.2f}，建议变换")
        # 可以继续做 log/sqrt/Box-Cox
```

### 3. 解读性

构造的特征（如比值、差值）有明确的物理含义，比 PCA 更可解释。告诉用户：
- 异常样本在某比值特征上偏高 → 说明**相对关系**不正常
- 异常样本在某差值特征上偏高 → 说明**偏离程度**不正常

### 4. 与多元高斯的关系

```
                特征相关
                   │
          ┌────────┴────────┐
          │                 │
       m > n             m ≤ n
          │                 │
    多元高斯自动      构造特征 +
    捕获相关性        一元高斯
    （一个模型）      （扩展特征集）
```

两种方式**目的相同**（捕获相关性），**方法不同**（一个改模型，一个改数据），**不混用**。
