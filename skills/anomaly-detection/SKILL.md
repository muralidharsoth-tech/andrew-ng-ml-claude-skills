---
name: anomaly-detection
description: |
  异常检测（Anomaly Detection）—— 基于高斯分布的一元/多元异常检测，纯 NumPy 实现。
  当用户提到"异常检测"、"anomaly detection"、"找异常"、"离群点"、"异常值检测"、
  "outlier detection"、"高斯异常检测"、"检测异常数据"、"帮我检测异常"、"异常样本"、
  "找出异常的"、"哪些数据有问题"、"不正常的数据"、"反常数据"等类似场景时触发。
  适用于任何"给一堆数据，想找出哪些样本不太正常"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 异常检测（Anomaly Detection）

## 概述

基于吴恩达课程第 9 周的异常检测算法，核心原理：

- **训练阶段**：假设正常数据服从高斯分布，从数据中学习分布参数（均值 μ、方差 σ²）
- **检测阶段**：计算新样本的 p(x)，若 p(x) < ε（阈值）则标记为异常
- **一元高斯**：各特征独立建模，p(x) = Π p(x_j; μ_j, σ²_j)
- **多元高斯**：用协方差矩阵 Σ 捕获特征间相关性（需 m > n）
- 所有代码使用纯 NumPy + Matplotlib，自动保存为带时间戳的 `.py` 文件

### 核心规则

- 数据输入支持三种方式：粘贴数组、CSV 文件、口头描述
- 自动检验每个特征是否符合高斯分布，不符合则尝试 log/sqrt/Box-Cox 变换
- 自动检测特征间相关性，智能选择一元或多元高斯
- ε 阈值智能推荐（有标签用 F1 最大化，无标签用分位数法），推荐后可调整
- 输出包含：检测结果、`.py` 代码文件、`.png` 可视化图表

---

## 交互流程

### 环境预检（生成代码前必须执行）

先确认当前 Python 解释器可以导入 numpy 和 matplotlib。

缺少依赖时暂停执行并提示：python -m pip install -r requirements.txt，然后运行 python check_env.py；预检未通过时不要声称代码可运行。


### 步骤 1：数据输入

> 请提供你的数据。支持三种方式：
>
> **A. 粘贴数组** — 直接粘贴 Python 格式的数据
> **B. CSV 文件** — 提供文件路径，我来读取
> **C. 口头描述** — 描述数据含义和范围，我来构造示例
>
> 如果有异常标签（0=正常, 1=异常），也请一并提供。

自动判断输入类型：

```
包含 ".csv"                            → CSV 文件模式
包含 "[[", "np.array", "X =" 等       → 数组粘贴模式
包含 "大概", "模拟", "比如" 等         → 口头描述模式
```

#### 模式 A：粘贴数组

> 请粘贴你的数据。格式示例：
> ```python
> # 多特征数据：
> x1: [14.2, 13.8, 14.5, 25.0, 13.9]
> x2: [2.8,  2.9,  3.0,  8.5,  2.7]
> # 或二维数组：
> [[14.2, 2.8], [13.8, 2.9], [14.5, 3.0], [25.0, 8.5], [13.9, 2.7]]
> ```
>
> 如果有标签：`y: [0, 0, 0, 1, 0]`（0=正常, 1=异常）
>
> 还需提供：每个特征的名字（用于图表标注）、检测主题（用于图表标题）

提取：`X_raw`（m×n 数组）、`y_raw`（可选，0/1 数组）、`feature_names`、`title_topic`

#### 模式 B：CSV 文件

> 请提供 CSV 文件路径。
> - 文件要求：每行一条样本，每列一个特征
> - 如果有标签列，请告知列名或列索引

读取逻辑：先尝试 UTF-8，失败则尝试 GBK，再失败尝试 latin-1。自动跳过非数值列。

#### 模式 C：口头描述

> 请描述你的数据：
> - 有多少个特征？每个特征代表什么？值范围大概多少？
> - 异常样本大概长什么样？
> - 大概有多少条数据？

根据描述用 `np.random.seed(42)` 生成模拟数据并打印，让用户确认。

---

### 步骤 2：数据初探确认

解析完成后，打印数据摘要，确认无误再继续：

> 📊 **数据概览**
>
> | 指标 | 值 |
> |------|-----|
> | 样本数 (m) | {m} |
> | 特征数 (n) | {n} |
> | 有标签 | {是/否} |
> | 特征名 | {feature_names} |
>
> | 特征 | 均值 | 标准差 | 最小值 | 最大值 |
> |------|------|--------|--------|--------|
> | {name1} | {mean1} | {std1} | {min1} | {max1} |
> | ... | ... | ... | ... | ... |
>
> 数据看起来正确吗？（回复"可以"继续，或指出问题）

同时做数据健康检查：

| 检查项 | 处理 |
|--------|------|
| m < 10 | ⚠️ 警告样本太少，高斯估计不可靠 |
| 某特征方差 ≈ 0 | ⚠️ 建议删掉该特征（无区分度） |
| n > m | ⚠️ 提示后续只能走一元高斯路线 |

---

### 步骤 3：高斯分布检验

对每个特征逐一检测：

**检测方法（在生成的 .py 中实现）：**
1. 计算偏度 (skewness)：衡量对称性，0 = 完全对称
2. 计算峰度 (kurtosis)：衡量尾部厚度，3 = 标准正态
3. 绘制 Q-Q 图（理论分位数 vs 实际分位数）
4. 绘制直方图 + 正态曲线叠加

**判断标准：**

```
|skewness| < 1 且 Q-Q 图大致呈直线  →  ✅ 符合高斯
|skewness| >= 1 或 Q-Q 图明显偏离   →  ❌ 不符合，需要变换
```

汇报结果：

> 🔍 **高斯分布检验结果**
>
> | 特征 | 偏度 | 峰度 | Q-Q图 | 判断 |
> |------|------|------|-------|------|
> | {name1} | 0.32 | 2.95 | 直线 | ✅ 符合 |
> | {name2} | 2.50 | 8.12 | 弯曲 | ❌ 不符合 |

#### 当有特征不符合时——执行变换

按顺序尝试，**任一方法使 |skewness| < 1 即停止**：

```
右偏（偏度 > 0，长尾在右边）：
  1. log(x + offset)   — offset 使所有值为正的最小常数
  2. sqrt(x + offset)  — 比 log 温和
  3. Box-Cox           — (x^λ - 1) / λ，自动搜索最优 λ

左偏（偏度 < 0，长尾在左边）：
  1. x²                — 平方拉伸左侧
  2. exp(x / c)         — 指数拉伸
  3. Box-Cox            — 同上

三种都失败 → ⚠️ 警告 + 继续运行 + 详细记录到 references/gaussian-transform.md
```

对用户说：

> 特征 `{name}` 不符合高斯分布（偏度={skew:.2f}）。
> 建议：**{推荐变换方法}**，预期偏度可降至 {estimated_skew}。
>
> 选项：
> - **A. 按推荐变换（推荐）**
> - **B. 尝试其他变换**
> - **C. 不变换**（结果可能不准）

变换完成后汇报：

> ✅ 变换完成！偏度变化：
> - `{name}`：{before} → {after}（{method}变换）

将变换细节写入 **references/gaussian-transform.md**。

---

### 步骤 4：特征相关性分析

计算相关系数矩阵 `np.corrcoef(X.T)`，汇报：

> 📈 **特征相关性矩阵**
>
> |       | feat1 | feat2 | feat3 |
> |-------|-------|-------|-------|
> | feat1 | 1.00  | 0.12  | -0.85 |
> | feat2 | 0.12  | 1.00  | 0.08  |
> | feat3 | -0.85 | 0.08  | 1.00  |
>
> 检测到：**feat1 与 feat3 显著负相关 (r=-0.85)**

#### 决策分支

```
任意 |r| > 0.6 ?
  ├── 否 → 无显著相关 → 使用【一元高斯】
  │
  └── 是 → 进入 m vs n 判断：
            ├── m > n  → 推荐【多元高斯】
            │   原因：Σ 可逆（不会奇异），自动捕获相关性
            │   写入 references/multivariate-gaussian.md
            │
            └── m ≤ n  → 强制【一元高斯 + 构造组合特征】
                原因：Σ 奇异不可逆，多元高斯不可用
                写入 references/feature-engineering.md
```

对用户说（m > n 时）：

> 检测到特征间存在显著相关性，且样本数({m}) > 特征数({n})。
> **推荐：多元高斯模型**，用协方差矩阵自动捕获特征间的关联。
>
> 是否使用多元高斯？（回复"可以"确认，或回复"不用"退回到一元高斯）

对用户说（m ≤ n 时）：

> 检测到特征间存在显著相关性，但样本数({m}) ≤ 特征数({n})，无法使用多元高斯。
> **推荐：构造组合特征 + 一元高斯**。
>
> 建议构造：`x_new = feat1 / feat2`（或其他捕获相关性的组合）
>
> 是否接受？（回复"可以"确认，或自行指定组合方式）

**两种情况只选一种，不两者都做。**

---

### 步骤 5：训练模型

根据步骤 3-4 的决策：

**路线 A：一元高斯**
```python
mu = X.mean(axis=0)                    # 每个特征的均值
sigma2 = X.var(axis=0)                 # 每个特征的方差
# p(x) = Π_j (1/√(2π·σ²_j)) · exp(-(x_j - μ_j)² / (2σ²_j))
```

**路线 B：多元高斯**
```python
mu = X.mean(axis=0)                    # 均值向量 (n,)
Sigma = np.cov(X.T, bias=True)         # 协方差矩阵 (n, n)
# p(x) = 1/((2π)^(n/2)·|Σ|^(1/2)) · exp(-½(x-μ)ᵀ Σ⁻¹ (x-μ))
```

使用 log 空间计算避免数值下溢（高维时乘积 p(x) 可能趋近 0）。

---

### 步骤 6：推荐 ε 并询问用户

#### 有标签数据 → F1-score 最大化

```
1. 在 [min(p), max(p)] 区间均匀采样 1000 个候选 ε
2. 对每个候选：算 precision, recall, F1
3. 选 F1 最大的 ε
```

汇报：

> 🎯 **F1-score 最大化结果**
>
> 最佳 ε = **{value:.6f}**（F1 = {f1:.4f}）
> - 精确率 (Precision)：{prec:.1%}（报警中有多少是真异常）
> - 召回率 (Recall)：{rec:.1%}（真异常中有多少被检出）
>
> F1 vs ε 曲线已生成。是否需要调整 ε？
> （直接回复"可以"确认，或输入新值）

#### 无标签数据 → 分位数法

```
1. 计算所有训练样本的 p(x)
2. ε = 第 1% 分位数（约 1% 的数据被标为异常）
```

汇报：

> 🎯 **分位数法推荐**
>
> ε = **{value:.6f}**（p(x) 最小的 1% 分位）
> 约 **{m*0.01:.0f}** 条样本会被标记为异常。
>
> 是否需要调整？你可以：
> - 回复"可以"使用 1%
> - 输入新比例（如 `2%`、`0.5%`）
> - 直接输入新的 ε 值

---

### 步骤 7：输出检测结果

用最终确定的 ε 运行检测：

> 📋 **异常检测结果**
>
> | 指标 | 值 |
> |------|-----|
> | 模型类型 | {一元高斯 / 多元高斯} |
> | ε 阈值 | {epsilon}（{选择方法}） |
> | 总样本 | {m} |
> | 检出异常 | {n_anomalies}（{pct:.1%}） |
>
> **Top 异常样本（p(x) 最小）**：
>
> | 序号 | p(x) | {feat1} | {feat2} | ... |
> |------|------|---------|---------|-----|
> | #3   | {p3:.2e} | {val} | {val} | ... |
> | #17  | {p17:.2e} | {val} | {val} | ... |
> | ...  | ... | ... | ... | ... |
>
> 异常样本 vs 正常样本的特征均值对比：
>
> | 特征 | 异常组均值 | 正常组均值 | 差异 |
> |------|-----------|-----------|------|
> | ... | ... | ... | ... |

---

### 步骤 8：生成图表

生成一张多子图大图，保存为 `{filename_base}_anomaly_detection.png`：

```
子图布局（自适应 n）：

  Row 1: 各特征分布直方图 + 正态曲线叠加（1×n 子图）
  Row 2: 各特征 Q-Q 图（1×n 子图）
  Row 3: 相关性热力图（n×n）+ p(x) 直方图 + ε 阈值红线
  Row 4: [n=2] 二维散点 + 高斯等高线 + 异常红色高亮
         [有变换] 变换前后偏度对比柱状图
         [有标签] F1 vs ε 曲线图
```

Matplotlib 设置：
```python
import matplotlib
matplotlib.use('Agg')  # 无头环境
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

### 步骤 9：生成 .py 文件

文件名格式：`{prefix}_anomaly_{YYYYMMDD}_{HHMMSS}.py`

生成的 `.py` 包含完整可运行代码：
1. 数据定义/加载
2. 高斯检验（偏度、Q-Q 图）
3. 数据变换（如果有）
4. 相关性分析
5. 模型训练（一元/多元）
6. ε 选择
7. 异常检测 + 结果输出
8. 所有 matplotlib 图表代码
9. 完整中英文注释

**可脱离技能环境独立运行**，只需 NumPy + Matplotlib。

---

## 代码模板

### 模板 A：一元高斯异常检测

适用于：特征间无显著相关 或 m ≤ n（走构造特征路线）

```python
# -*- coding: utf-8 -*-
"""
一元高斯异常检测：{TITLE_TOPIC}
生成时间：{TIMESTAMP}
模型：一元高斯（Univariate Gaussian）
ε 选择：{EPSILON_METHOD}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据 ==========
X_raw = np.array({X_DATA})
m, n = X_raw.shape
feature_names = {FEATURE_NAMES}
title_topic = "{TITLE_TOPIC}"

{HAS_LABELS_CODE}

print("=" * 60)
print(f"异常检测：{title_topic}")
print(f"样本数: {m}, 特征数: {n}")
print("=" * 60)

# ========== 2. 高斯分布检验 ==========
def compute_skewness(x):
    """计算偏度"""
    m = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    return np.sum(((x - mu) / sigma) ** 3) / m

def compute_kurtosis(x):
    """计算峰度（excess kurtosis）"""
    m = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    return np.sum(((x - mu) / sigma) ** 4) / m - 3

print("\n" + "=" * 60)
print("各特征高斯分布检验")
print("=" * 60)
for j in range(n):
    sk = compute_skewness(X_raw[:, j])
    kt = compute_kurtosis(X_raw[:, j])
    status = "OK" if abs(sk) < 1 else "需要变换"
    print(f"  {feature_names[j]}: 偏度={sk:+.3f}, 峰度(excess)={kt:+.3f} → {status}")

# ========== 3. 数据变换（如果需要） ==========
X = X_raw.copy()
transform_desc = []
{TRANSFORM_CODE}

if transform_desc:
    print(f"\n变换记录: {'; '.join(transform_desc)}")

# ========== 4. 特征相关性分析 ==========
corr = np.corrcoef(X.T)
print("\n" + "=" * 60)
print("特征相关性矩阵")
print("=" * 60)
header = "         " + "".join(f"{n:>10}" for n in feature_names)
print(header)
for i in range(n):
    row = f"{feature_names[i]:>8} " + "".join(f"{corr[i,j]:10.3f}" for j in range(n))
    print(row)

{FEATURE_ENGINEERING_CODE}

# ========== 5. 一元高斯模型训练 ==========
mu = X.mean(axis=0)
sigma2 = X.var(axis=0)

print("\n" + "=" * 60)
print("一元高斯模型参数")
print("=" * 60)
for j in range(n):
    print(f"  {feature_names[j]:>12}: mu={mu[j]:.4f}, sigma2={sigma2[j]:.4f}")

# ========== 6. 计算 p(x) ==========
def univariate_gaussian_pdf(X, mu, sigma2):
    """计算一元高斯概率密度（逐特征乘积）"""
    m, n = X.shape
    # 用 log 空间避免下溢
    log_p = np.zeros(m)
    for j in range(n):
        log_p += -0.5 * np.log(2 * np.pi * sigma2[j]) - (X[:, j] - mu[j])**2 / (2 * sigma2[j])
    return np.exp(log_p)

p_vals = univariate_gaussian_pdf(X, mu, sigma2)

# ========== 7. 选择 ε ==========
{EPSILON_SELECTION_CODE}

print(f"\nε = {epsilon:.6f}（{epsilon_method}）")

# ========== 8. 检测异常 ==========
anomaly_mask = p_vals < epsilon
anomaly_indices = np.where(anomaly_mask)[0]
normal_indices = np.where(~anomaly_mask)[0]

print("\n" + "=" * 60)
print("检测结果")
print("=" * 60)
print(f"  总样本: {m}")
print(f"  异常:   {len(anomaly_indices)} ({len(anomaly_indices)/m*100:.1f}%)")
print(f"  正常:   {len(normal_indices)} ({len(normal_indices)/m*100:.1f}%)")

if len(anomaly_indices) > 0:
    print(f"\n  Top-10 异常样本（按 p(x) 升序）:")
    sorted_anomalies = sorted(anomaly_indices, key=lambda i: p_vals[i])
    header = f"  {'序号':>5} | {'p(x)':>10} |" + "|".join(f"{n:>10}" for n in feature_names)
    print(f"  {header}")
    print("  " + "-" * (20 + 12 * n))
    for idx in sorted_anomalies[:10]:
        vals = " |".join(f"{X[idx, j]:10.3f}" for j in range(n))
        print(f"  {idx:5d} | {p_vals[idx]:10.2e} |{vals}")

    # 异常 vs 正常 特征对比
    print(f"\n  异常 vs 正常 特征均值:")
    print(f"  {'特征':>12} | {'异常组均值':>10} | {'正常组均值':>10} | {'差异':>10}")
    for j in range(n):
        anom_mean = X[anomaly_indices, j].mean()
        norm_mean = X[normal_indices, j].mean()
        print(f"  {feature_names[j]:>12} | {anom_mean:10.3f} | {norm_mean:10.3f} | {anom_mean - norm_mean:+10.3f}")
else:
    print("  ⚠️ 未检测到异常。考虑调小 ε。")

# ========== 9. 可视化 ==========
n_features = n
n_cols_q1 = max(1, n_features)
# 计算总行数
n_rows_hist = 1 if n_features <= 3 else 2
n_rows_qq = 1 if n_features <= 3 else 2
extra_rows = 1  # 热力图 + p(x)直方图行
if n_features == 2:
    extra_rows += 1  # 二维散点+等高线
{HAS_TRANSFORM_ROW}
{HAS_F1_ROW}
total_rows = n_rows_hist + n_rows_qq + extra_rows

fig = plt.figure(figsize=(max(14, n_features * 5), total_rows * 4.5))

current_row = 0

# 子图函数：用 GridSpec 动态布局
gs = fig.add_gridspec(total_rows, max(3, n_features))

# Row: 直方图 + 正态叠加
for j in range(n_features):
    ax = fig.add_subplot(gs[current_row, j])
    data = X[:, j]
    ax.hist(data, bins=min(30, m//3), density=True, alpha=0.6, color='steelblue', edgecolor='white')
    x_range = np.linspace(data.min(), data.max(), 200)
    pdf = (1 / np.sqrt(2 * np.pi * sigma2[j])) * np.exp(-(x_range - mu[j])**2 / (2 * sigma2[j]))
    ax.plot(x_range, pdf, 'r-', linewidth=2, label='拟合正态')
    ax.set_title(f'{feature_names[j]} 分布', fontsize=11, fontweight='bold')
    ax.set_xlabel(feature_names[j])
    ax.set_ylabel('密度')
    ax.legend(fontsize=8)
current_row += 1

# Row: Q-Q 图
for j in range(n_features):
    ax = fig.add_subplot(gs[current_row, j])
    data_sorted = np.sort(X[:, j])
    # 理论分位数
    theoretical = np.random.normal(mu[j], np.sqrt(sigma2[j]), m)
    theoretical_sorted = np.sort(theoretical)
    ax.scatter(theoretical_sorted, data_sorted, alpha=0.6, s=20, c='steelblue')
    # 对角线
    lims = [min(theoretical_sorted.min(), data_sorted.min()),
            max(theoretical_sorted.max(), data_sorted.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='理论正态线')
    ax.set_title(f'{feature_names[j]} Q-Q', fontsize=11, fontweight='bold')
    ax.set_xlabel('理论分位数')
    ax.set_ylabel('实际分位数')
    ax.legend(fontsize=8)
current_row += 1

# Row: 相关性热力图 + p(x) 直方图
ax_corr = fig.add_subplot(gs[current_row, :n_features])
im = ax_corr.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax_corr.set_xticks(range(n_features))
ax_corr.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
ax_corr.set_yticks(range(n_features))
ax_corr.set_yticklabels(feature_names, fontsize=9)
ax_corr.set_title('特征相关性矩阵', fontsize=12, fontweight='bold')
for i in range(n_features):
    for j in range(n_features):
        ax_corr.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                     color='white' if abs(corr[i,j]) > 0.5 else 'black', fontsize=9)
plt.colorbar(im, ax=ax_corr, shrink=0.8)

ax_hist = fig.add_subplot(gs[current_row, n_features:])
ax_hist.hist(p_vals, bins=min(50, m//2), color='steelblue', alpha=0.7, edgecolor='white')
ax_hist.axvline(x=epsilon, color='red', linestyle='--', linewidth=2, label=f'epsilon = {epsilon:.2e}')
ax_hist.set_title('p(x) 分布 + epsilon 阈值', fontsize=12, fontweight='bold')
ax_hist.set_xlabel('p(x)')
ax_hist.set_ylabel('频次')
ax_hist.legend(fontsize=9)
# 标注异常区域
ylim = ax_hist.get_ylim()
ax_hist.fill_betweenx([0, ylim[1]], 0, epsilon, alpha=0.15, color='red')
ax_hist.text(epsilon/2, ylim[1]*0.95, '异常', ha='center', fontsize=9, color='red')
ax_hist.text((epsilon + p_vals.max())/2, ylim[1]*0.95, '正常', ha='center', fontsize=9, color='green')
current_row += 1

# Row（条件）: 二维散点 + 高斯等高线
if n_features == 2:
    ax = fig.add_subplot(gs[current_row, :3])
    # 正常样本
    ax.scatter(X[normal_indices, 0], X[normal_indices, 1], c='steelblue', s=40,
               alpha=0.6, edgecolors='white', linewidth=0.5, label=f'正常 ({len(normal_indices)})')
    # 异常样本
    if len(anomaly_indices) > 0:
        ax.scatter(X[anomaly_indices, 0], X[anomaly_indices, 1], c='red', s=120,
                   edgecolors='darkred', linewidth=1.5, marker='X', zorder=5,
                   label=f'异常 ({len(anomaly_indices)})')

    # 高斯等高线（一元：椭圆主轴上对齐坐标轴）
    from matplotlib.patches import Ellipse
    for std_level, alpha_val, style in [(1, 0.3, '--'), (2, 0.15, ':'), (3, 0.05, '-.')]:
        ellipse = Ellipse(xy=(mu[0], mu[1]),
                          width=2 * std_level * np.sqrt(sigma2[0]),
                          height=2 * std_level * np.sqrt(sigma2[1]),
                          edgecolor='gray', facecolor='none', linestyle=style, linewidth=1.2,
                          label=f'{std_level}sigma')
        ax.add_patch(ellipse)
    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title('特征空间 + 异常样本标注', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    current_row += 1

# Row（条件）: 变换前后偏度对比
{TRANSFORM_CHART_CODE}

# Row（条件）: F1 vs epsilon 曲线
{F1_CHART_CODE}

plt.suptitle(f'异常检测结果：{title_topic}', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('{FILENAME_BASE}_anomaly_detection.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {FILENAME_BASE}_anomaly_detection.png")
plt.close()
```

### 模板 B：多元高斯异常检测

适用于：特征间显著相关 且 m > n

模板 B 的结构与模板 A **95% 相同**，仅在以下部分不同：

```python
# ========== 5. 多元高斯模型训练（替代一元部分） ==========
mu = X.mean(axis=0)
# np.cov(bias=True) 使用 1/m 而非 1/(m-1)，匹配吴恩达课程
Sigma = np.cov(X.T, bias=True)
# 小正则化防止奇异
Sigma += {COV_REG} * np.eye(n)

print("\n" + "=" * 60)
print("多元高斯模型参数")
print("=" * 60)
print(f"  均值向量 mu: {mu}")
print(f"  协方差矩阵 Sigma ({n}x{n}):")
print(Sigma)

# ========== 6. 计算 p(x) — 多元版本 ==========
def multivariate_gaussian_log_p(X, mu, Sigma):
    """多元高斯 log-p(x) — log 空间避免下溢"""
    m, n = X.shape
    sign, logdet = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    diff = X - mu
    # Mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu) per sample
    mahalanobis = np.sum(diff @ Sigma_inv * diff, axis=1)
    log_p = -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)
    return log_p

log_p_vals = multivariate_gaussian_log_p(X, mu, Sigma)
p_vals = np.exp(log_p_vals)
# 后续 epsilon 选择和检测完全相同

# 可视化部分 — 二维散点图增加多元高斯等高线（替代一元椭圆）
if n == 2:
    # 生成网格
    x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    log_p_grid = multivariate_gaussian_log_p(grid, mu, Sigma)
    p_grid = np.exp(log_p_grid).reshape(X1.shape)
    # 绘制等高线（倾斜椭圆！）
    ax.contour(X1, X2, p_grid, levels=8, colors='gray', alpha=0.5, linewidths=0.8)
    ax.contour(X1, X2, p_grid, levels=[epsilon], colors='red', linewidths=2, linestyles='--')
```

> **注意**：模板 B 完整代码在生成时基于模板 A + 上述差异拼接而成，避免重复维护。

---

## 占位符说明

| 占位符 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `{X_DATA}` | 数组 | 特征矩阵 | `np.array([[14.2, 2.8], [13.8, 2.9], ...])` |
| `{M}` | 整数 | 样本数 | `150` |
| `{N}` | 整数 | 特征数 | `3` |
| `{FEATURE_NAMES}` | 列表 | 特征名 | `["cpu_temp", "mem_usage", "disk_io"]` |
| `{TITLE_TOPIC}` | 字符串 | 检测主题 | `"服务器监控异常检测"` |
| `{HAS_LABELS_CODE}` | 代码 | 标签定义（如果有） | `y = np.array([0, 0, 1, 0, ...])` 或 `# 无标签` |
| `{TRANSFORM_CODE}` | 代码块 | 变换代码 | `X[:,1] = np.log(X[:,1] + 0.01)` |
| `{TRANSFORM_DESC}` | 字符串 | 变换描述 | `"mem_usage: log(x+0.01), 偏度 2.5→0.3"` |
| `{FEATURE_ENGINEERING_CODE}` | 代码块 | 构造新特征代码 | `X_new = X[:,0] / (X[:,1] + 1e-8)` |
| `{EPSILON_SELECTION_CODE}` | 代码块 | ε 选择代码块 | （见下方 ε 选择逻辑） |
| `{EPSILON_VALUE}` | 浮点数 | 最终 ε 值 | `0.000342` |
| `{EPSILON_METHOD}` | 字符串 | ε 选择方法说明 | `"F1-score最大化"` 或 `"1%分位数"` |
| `{COV_REG}` | 浮点数 | 协方差正则化 | `1e-6`（多元高斯专用） |
| `{ANOMALY_COUNT}` | 整数 | 异常数量 | `5` |
| `{FILENAME_BASE}` | 字符串 | 文件前缀 | `"server_monitor_20260802_143000"` |
| `{HAS_TRANSFORM_ROW}` | 整数 | 是否有变换图行 | `1` 或 `0` |
| `{TRANSFORM_CHART_CODE}` | 代码块 | 变换前后对比图 | （条件生成） |
| `{HAS_F1_ROW}` | 整数 | 是否有 F1 图行 | `1` 或 `0` |
| `{F1_CHART_CODE}` | 代码块 | F1 vs ε 曲线图 | （条件生成） |
| `{TIMESTAMP}` | 字符串 | 生成时间 | `"2026-08-02 14:30:00"` |

### ε 选择代码块占位符 `{EPSILON_SELECTION_CODE}`

**有标签版（F1 最大化）**：
```python
epsilon_method = "F1-score最大化"
best_eps, best_f1 = 0, 0
f1_history = []
eps_candidates = np.linspace(p_vals.min(), p_vals.max(), 1000)
for eps in eps_candidates:
    y_pred = (p_vals < eps).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1_history.append(f1)
    if f1 > best_f1:
        best_f1, best_eps = f1, eps
epsilon = best_eps
print(f"最佳 epsilon = {epsilon:.6f} (F1={best_f1:.4f})")
```

**无标签版（分位数法）**：
```python
epsilon_method = "1%分位数"
percentile = 1.0
epsilon = np.percentile(p_vals, percentile)
print(f"epsilon = {epsilon:.6f} (p(x) 第 {percentile}% 分位数)")
```

---

## 关键注意事项

### 编码与兼容性

- **Windows GBK 终端**：打印语句中避免使用 Unicode 特殊字符
  - `μ` → `mu`，`σ²` → `sigma2`，`Σ` → `Sigma`，`Π` → 直接展开公式
- **无头环境**：使用 `matplotlib.use('Agg')`，`.py` 文件末尾加 `plt.close()`
- **中文字体**：`rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']`

### 数值稳定性

- **一元高斯 p(x)**：n 个特征的概率密度连乘会趋近 0，用 log 空间计算再 `np.exp()`
- **多元高斯 p(x)**：`np.linalg.slogdet()` 替代 `np.linalg.det()`，避免行列式下溢
- **协方差正则化**：`Sigma += 1e-6 * np.eye(n)` 防止奇异

### 数据质量

- `np.random.seed(42)` 保证模拟数据和切分可复现
- 缺失值用中位数填充（`np.nanmedian`）
- 变换时加 offset 保证全正值（log/Box-Cox 要求）

### 输出文件

- `.py` 文件命名：`{prefix}_anomaly_{YYYYMMDD}_{HHMMSS}.py`
- `.png` 文件命名：同前缀 + `_anomaly_detection.png`
- 保存到当前工作目录

---

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|---------|
| [references/gaussian-transform.md](references/gaussian-transform.md) | log/sqrt/Box-Cox 变换方法 + 代码 | 任何特征不符合高斯分布时 |
| [references/multivariate-gaussian.md](references/multivariate-gaussian.md) | 多元高斯模型原理 + 公式 + 实现 | m>n 且有特征相关性时 |
| [references/feature-engineering.md](references/feature-engineering.md) | 构造组合特征的方法与代码 | m≤n 且有特征相关性时 |



