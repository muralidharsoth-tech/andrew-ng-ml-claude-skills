---
name: logistic-regression
description: |
  逻辑回归分类预测 —— 支持二分类与多分类，内置正则化梯度下降。
  当用户提到"逻辑回归"、"logistic regression"、"分类预测"、"二分类"、
  "多分类"、"帮我分类"、"预测类别"、"是否"、"会不会"等类似场景时触发。
  适用于任何"给一堆 x,y 分类标签，想预测新样本类别"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 逻辑回归预测（Logistic Regression）

## 概述

此技能支持**二分类**和**多分类**的逻辑回归完整流程：
收集数据 → 判断分类类型 → 检测非线性 → 确认预测目标 → 建议参数 → 训练模型 → 预测新值 → 可视化输出。

核心原则：
- **一律使用梯度下降**（逻辑回归无闭式解）
- **内置特征缩放**（z-score 标准化，梯度下降必需）
- **默认启用 L2 正则化**（防止过拟合，λ 可调为 0 关闭）
- 自动检测是否需要多项式特征（弯曲决策边界）
- 自动判断二分类还是多分类

代码会自动保存到工作目录并运行，每次生成带时间戳的新文件。

## 交互流程

### 环境预检（生成代码前必须执行）

先确认当前 Python 解释器可以导入 numpy 和 matplotlib。

缺少依赖时暂停执行并提示：python -m pip install -r requirements.txt，然后运行 python check_env.py；预检未通过时不要声称代码可运行。


严格按以下 8 步执行，不要跳过任何一步：

### 步骤 1：收集数据

首先确认特征数量和标签含义：

> 你有几个特征（x 变量）？每个 x 叫什么名字、单位是什么？
> 标签 y 代表什么类别？有几个类别？分别叫什么？

然后让用户提供数据：

> 请提供你的训练数据。可以用任意格式发给我，例如：
>
> 二分类：
> `保温厚度 3cm, 窗墙比 0.20 → 不节能`
> `保温厚度 8cm, 窗墙比 0.40 → 节能`
> `...`
>
> 多分类：
> `保温厚度 3cm, 窗墙比 0.20 → A级`
> `保温厚度 8cm, 窗墙比 0.40 → B级`
> `...`
>
> 同时告诉我：**预测的主题是什么**（用于图表标题，英文）。

从用户回复中提取：
- `n_features`（特征数量）
- `x_names`（list，如 `["保温厚度", "窗墙比"]`）
- `x_units`（list，如 `["cm", ""]`）
- `X_raw`（list of lists，每行 n 个 x 值）
- `y_raw`（list，标签值）
- `y_classes`（dict，如 `{0: "不节能", 1: "节能"}` 或 `{0: "A级", 1: "B级", 2: "C级"}`）
- `title_topic`（如 `"Building Energy Efficiency Classification"`，英文）

如果用户提供的标签是字符串（如 "节能"/"不节能"），自动编码为 0, 1, 2... 并确认映射关系。

如果用户没有给出某个信息，根据上下文合理推断并确认。

### 步骤 2：判断分类类型

检查 y 的唯一值数量 `K = len(set(y_raw))`：

| K | 分类类型 | 后续路径 |
|---|---------|---------|
| = 2 | **二分类** | 继续步骤 3（非线性检测） |
| > 2 | **多分类** | 跳至步骤 4（确认预测目标），代码走 `references/multi-class-logistic.md` |

告知用户：

> 检测到 y 有 {K} 个类别，这是**{二分类/多分类}**问题。

如果是多分类，标记 `use_multiclass = True`。

### 步骤 3：检测非线性关系（仅二分类）

拿到数据后，判断两个类之间是否可能由**非线性决策边界**分开。根据以下信号综合判断：

| 信号类型 | 具体表现 | 结论 |
|----------|----------|------|
| 用户描述 | "先好后差再转好"、"分类边界是曲线"、"直线分不开" | 疑似非线性 |
| 数据分布 | 如果画出散点图，两个类交错分布，直线无法分开 | 疑似非线性 |
| 领域知识 | 如"保温越厚越节能，但超过一定值后又变差" | 可主动建议 |

**如果疑似非线性，主动询问用户确认：**

> 从你的数据来看，两个类之间的决策边界可能不是**直线**。使用标准逻辑回归（线性边界 θ₀+θ₁x₁+θ₂x₂=0）可能无法很好地区分两类。
> 建议使用**多项式特征**（默认 degree=2）来构建弯曲的决策边界。
>
> - **degree=2**：增加 x₁², x₂² 项，能形成椭圆/双曲线边界
> - **degree=3**：更复杂的曲线形状（需要更多数据支撑）
> - 建议先用 degree=2 试试
>
> 是否使用多项式特征？（直接回复"可以"使用 degree=2，或指定 degree 值）

如果用户确认使用多项式：

- 设置标志 `use_polynomial = True`
- 记录 `degree`（用户指定或默认 2）
- 后续步骤将走 `references/polynomial-logistic.md` 模板

如果用户认为直线够用（或数据无明显非线性特征），设置 `use_polynomial = False`，走标准二分类流程。

### 步骤 4：确认预测目标

> 你想预测哪个新样本的类别？请给出对应的 x 值：
> {x1_name} = ?, {x2_name} = ?, ...

记录为 `predict_x`（list，长度等于 n_features）。

### 步骤 5：建议学习率 α 和正则化 λ

特征已做 z-score 标准化。根据当前场景选择建议参数：

**α 建议规则：**

| 场景 | 建议 α | 原因 |
|------|--------|------|
| 标准二分类（任意特征数） | 0.1 | 特征缩放后梯度稳定 |
| 多项式二分类 | 0.05 | 特征空间更大，梯度更复杂 |
| 多分类（Softmax） | 0.05 | K 组参数同时更新 |
| 多分类 + 多项式 | 0.01 | 最复杂场景 |

**λ 建议规则（λ=0 关闭正则化）：**

计算有效特征数 `n_eff`（不含偏置项）和比率 `ratio = m / n_eff`：
- 标准二分类：`n_eff = n_features`
- 多项式二分类：`n_eff = n_base × degree`
- 多分类：`n_eff = n_features × K`

| 条件 | 建议 λ |
|------|--------|
| ratio < 3 | 5.0（强力正则化，样本极少） |
| 3 ≤ ratio < 5 | 1.0（中度正则化） |
| 5 ≤ ratio < 10 | 0.5（轻度正则化） |
| ratio ≥ 10 | 0.1（默认轻度正则化） |

**iters 建议：**

| 场景 | 建议 iters |
|------|-----------|
| 标准二分类 | 2000 |
| 多项式二分类 | 3000 |
| 多分类 | 3000 |
| 多分类 + 多项式 | 5000 |

**询问话术：**

> 当前场景：{场景描述}。特征已做 z-score 标准化。
> 建议参数：
> - 学习率 α = **{suggested_alpha}**
> - 正则化 λ = **{suggested_lambda}**（正则化默认开启，λ=0 则关闭）
> - 迭代次数 = **{suggested_iters}**
>
> 是否需要调整？（直接回复"可以"使用建议值，或指定要修改的参数）

如果用户要求调整，使用用户指定的值。确认后记录 `alpha`, `lambda_val`, `iters`。

**无论是否正则化（λ 是否为 0），都使用相同的梯度下降函数。** λ=0 时正则化项自动退化消失。

### 步骤 6：生成代码

在当前工作目录生成代码文件。文件名规则：

- 统一使用 `logistic_regression_{YYYYMMDD}_{HHMMSS}.py`

模板选择（决策树）：

```
是逻辑回归？
  │
  ├── K > 2（多分类）
  │     └── 读取 references/multi-class-logistic.md
  │
  └── K = 2（二分类）
        ├── use_polynomial = True？
        │     └── 是 → 读取 references/polynomial-logistic.md
        │
        └── 否（标准二分类）
              ├── n_features = 1 → 使用本文件「模板 B：1 特征」
              └── n_features ≥ 2 → 使用本文件「模板 A：多特征」（最常用）
```

将所有 `{PLACEHOLDER}` 替换为实际值。

### 步骤 7：运行代码

```bash
cd "e:\py 代码编辑（机器学习）" && python logistic_regression_{timestamp}.py
```

如果运行报错，分析错误并修复代码后重新运行。

### 步骤 8：输出结果

向用户报告：
- **模型公式**（原始空间参数）
  - 二分类：`g(x) = sigmoid(θ₀ + θ₁x₁ + θ₂x₂)`
  - 决策边界：`θ₀ + θ₁x₁ + θ₂x₂ = 0`
- **训练集准确率**（及多分类的各类别指标）
- **预测结果**：概率值 + 预测类别
- **最终代价 J(θ)** 和使用的 λ 值
- 图片保存路径和代码文件路径

---

## 代码模板 A：标准二分类 + 多特征（≥ 2 个特征）

> 这是最常用的模板。3 张图输出（决策边界 + 收敛曲线 + 3D 概率曲面）。
> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 特征矩阵 (m, n)
y = np.array([{Y_DATA}]).reshape(-1, 1)   # 标签 (m, 1)，值为 0 或 1

m, n = X_raw.shape
x_names = {X_NAMES}     # 特征名列表（中文）
x_units = {X_UNITS}     # 特征单位列表
y_classes = {Y_CLASSES}  # 类别映射，如 {0: "Not Efficient", 1: "Efficient"}

print(f"样本数: {m}, 特征数: {n}")
print(f"X_raw:\n{X_raw}")
print(f"y:\n{y.flatten()}")

# ========== 2. 特征缩放（z-score）==========
X_mean = X_raw.mean(axis=0, keepdims=True)   # (1, n)
X_std  = X_raw.std(axis=0, keepdims=True)    # (1, n)
X_scaled = (X_raw - X_mean) / X_std           # (m, n)

# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_scaled])   # (m, n+1)

print(f"\nX_mean: {X_mean}")
print(f"X_std: {X_std}")

# ========== 3. Sigmoid 函数 ==========
def sigmoid(z):
    """逻辑函数，将任意实数映射到 (0,1)"""
    return 1 / (1 + np.exp(-z))

# ========== 4. 正则化梯度下降（逻辑回归 + L2）==========
def logisticRidgeGD(X, y, theta, alpha, lam, iters):
    """
    逻辑回归 + L2 正则化梯度下降。

    代价函数：
      J(theta) = -(1/m) * sum(y*log(h) + (1-y)*log(1-h))
                 + (lam/(2m)) * sum(theta[1:]^2)

    梯度：
      grad = (1/m) * X^T @ (h - y)
      grad[1:] += (lam/m) * theta[1:]    （theta0 不参与惩罚）

    参数：
      X      - 特征矩阵 (m, n+1)，第 0 列为全 1（偏置）
      y      - 标签向量 (m, 1)
      theta  - 参数向量 (n+1, 1)
      alpha  - 学习率
      lam    - L2 正则化强度（>0 岭回归，=0 退化为标准梯度下降）
      iters  - 迭代次数
    """
    m_samples = len(y)
    J_history = np.zeros(iters)

    for i in range(iters):
        z = X @ theta                           # (m, n+1) @ (n+1, 1) -> (m, 1)
        h = sigmoid(z)                          # 预测概率 (m, 1)
        errors = h - y                          # 误差 (m, 1)

        # 梯度：数据拟合项
        gradient = (1 / m_samples) * (X.T @ errors)  # (n+1, 1)

        # L2 正则化梯度：跳过偏置项 theta[0]
        gradient[1:] += (lam / m_samples) * theta[1:]

        theta = theta - alpha * gradient

        # 代价函数（含正则化项，用于监控收敛）
        cost = -(1 / m_samples) * np.sum(
            y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10)
        )
        reg_term = (lam / (2 * m_samples)) * np.sum(theta[1:] ** 2)
        J_history[i] = cost + reg_term

    return theta, J_history


# 初始化参数
theta_scaled = np.zeros((n + 1, 1))   # (n+1, 1)
alpha = {ALPHA}                        # 学习率
lam = {LAMBDA}                         # L2 正则化强度
iters = {ITERS}                        # 迭代次数

print(f"\n训练参数: alpha={alpha}, lambda={lam}, iters={iters}")

theta_scaled, J_history = logisticRidgeGD(X, y, theta_scaled, alpha, lam, iters)

print(f"缩放空间参数: theta0={theta_scaled[0,0]:.6f}, "
      + ", ".join([f"theta{j+1}={theta_scaled[j+1,0]:.6f}" for j in range(n)]))
print(f"初始代价: {J_history[0]:.6f}")
print(f"最终代价: {J_history[-1]:.6f}")

# ========== 5. 参数还原（从缩放空间回到原始空间）==========
theta = np.zeros((n + 1, 1))
# 截距: theta0_orig = theta0_scaled - sum(theta_j_scaled * mean_j / std_j)
theta[0, 0] = theta_scaled[0, 0] - np.sum(
    theta_scaled[1:, 0] * X_mean[0] / X_std[0]
)
# 斜率: theta_j_orig = theta_j_scaled / std_j
for j in range(1, n + 1):
    theta[j, 0] = theta_scaled[j, 0] / X_std[0, j - 1]

print(f"\n原始空间参数:")
print(f"  theta0 (截距) = {theta[0,0]:.6f}")
for j in range(n):
    print(f"  theta{j+1} ({x_names[j]}) = {theta[j+1,0]:.6f}")

# 打印模型公式
formula = f"g(x) = sigmoid({theta[0,0]:.6f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign} {theta[j+1, 0]:.6f}*{x_names[j]}"
formula += ")"
print(f"\n模型公式: {formula}")

# 决策边界
if n == 2:
    x2_expr = f"x2 = -({theta[0,0]:.4f} + {theta[1,0]:.4f}*x1) / {theta[2,0]:.4f}" if abs(theta[2,0]) > 1e-10 else f"x1 = {-theta[0,0]/theta[1,0]:.4f}"
elif n == 1:
    x2_expr = f"x = {-theta[0,0]/theta[1,0]:.4f}"
else:
    x2_expr = f"theta0 + sum(theta_j * x_j) = 0"
print(f"决策边界 (h=0.5): {x2_expr}")

# ========== 6. 模型评估 ==========
z_train = X @ theta_scaled
h_train = sigmoid(z_train)
y_pred = (h_train >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100

print(f"\n训练集准确率: {accuracy:.1f}%")
print(f"训练集预测详情:")
for i in range(m):
    print(f"  样本{i+1}: x=({', '.join([f'{X_raw[i,j]:.{2 if j==1 else 0}f}' for j in range(n)])}), "
          f"真实y={int(y[i,0])}, 概率={h_train[i,0]:.4f}, 预测y={y_pred[i,0]}")

# ========== 7. 预测新样本 ==========
new_x = np.array([{PREDICT_X}])          # 新样本特征值，形状 (1, n)
new_x_scaled = (new_x - X_mean) / X_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
prob_new = sigmoid(X_new @ theta_scaled)[0, 0]
pred_new = 1 if prob_new >= 0.5 else 0

print(f"\n{'='*50}")
print(f"  新样本预测")
print(f"  x = ({', '.join([f'{new_x[0,j]}' for j in range(n)])})")
print(f"  预测概率 = {prob_new:.4f} ({prob_new*100:.1f}%)")
print(f"  预测结果: {y_classes[pred_new]}")
print(f"{'='*50}")

# ========== 8. 可视化 ==========
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15],
              hspace=0.35, wspace=0.30)

# ----- 左上：决策边界（2D 等高线 + 训练数据）-----
ax = fig.add_subplot(gs[0, 0])

# 绘制训练数据点
colors = ['#e74c3c' if yi == 0 else '#2ecc71' for yi in y.flatten()]
markers = ['x' if yi == 0 else 'o' for yi in y.flatten()]
labels_seen = set()
for i in range(m):
    lbl = y_classes[0] if y[i, 0] == 0 else y_classes[1]
    if lbl not in labels_seen:
        ax.scatter(X_raw[i, 0], X_raw[i, 1], c=colors[i], marker=markers[i],
                   s=120, label=lbl, zorder=5)
        labels_seen.add(lbl)
    else:
        ax.scatter(X_raw[i, 0], X_raw[i, 1], c=colors[i], marker=markers[i],
                   s=120, zorder=5)

# 决策边界线：theta0 + theta1*x1 + theta2*x2 = 0  ->  x2 = -(theta0 + theta1*x1) / theta2
x1_min, x1_max = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
x1_line = np.linspace(x1_min, x1_max, 200)
if abs(theta[2, 0]) > 1e-10:
    x2_line = -(theta[0, 0] + theta[1, 0] * x1_line) / theta[2, 0]
    # 裁剪 x2 范围，保持在可见区域内
    x2_min_v = X_raw[:, 1].min() - 0.05
    x2_max_v = X_raw[:, 1].max() + 0.05
    mask = (x2_line >= x2_min_v) & (x2_line <= x2_max_v)
    x1_line_vis = x1_line[mask]
    x2_line_vis = x2_line[mask]
    ax.plot(x1_line_vis, x2_line_vis, 'b-', linewidth=2.5,
            label='Decision Boundary (h=0.5)', zorder=3)
else:
    # theta2 ≈ 0，边界接近垂直
    x1_vert = -theta[0, 0] / theta[1, 0]
    ax.axvline(x=x1_vert, color='b', linewidth=2.5,
               label='Decision Boundary (h=0.5)', zorder=3)

# 填充决策区域（半透明背景色）
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05, 300)
)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_scaled = (grid_points - X_mean) / X_std
X_grid = np.hstack([np.ones((grid_points.shape[0], 1)), grid_scaled])
z_grid = sigmoid(X_grid @ theta_scaled).reshape(x1_grid.shape)

# 绘制概率等高线
ax.contourf(x1_grid, x2_grid, z_grid, levels=20, alpha=0.15,
            cmap='RdYlGn', zorder=0)
ax.contour(x1_grid, x2_grid, z_grid, levels=[0.5], colors='blue',
           linewidths=2.5, zorder=3)

# 绘制新预测点
pred_color = '#2ecc71' if pred_new == 1 else '#e74c3c'
pred_label = f'New: ({", ".join([f"{new_x[0,j]}" for j in range(n)])}), P={prob_new:.1%}'
ax.scatter([new_x[0, 0]], [new_x[0, 1]], c=pred_color, s=200, marker='*',
           label=pred_label, zorder=10)

ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
ax.set_ylabel(f'{x_names[1]} ({x_units[1]})', fontsize=11)
ax.set_title(f'(A) Decision Boundary & Data\nLogistic Ridge, lambda={lam}',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05)

# ----- 右上：梯度下降收敛曲线 -----
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(iters), J_history, 'b-', linewidth=1.5)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Cost J(theta)', fontsize=11)
ax2.set_title(f'(B) Ridge GD Convergence\n(alpha={alpha}, lambda={lam})',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 标注初始和最终代价
ax2.annotate(f'Start: {J_history[0]:.4f}',
             xy=(0, J_history[0]), xytext=(iters * 0.08, J_history[0] + 0.08),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             fontsize=9, color='red')
ax2.annotate(f'End: {J_history[-1]:.4f}',
             xy=(iters - 1, J_history[-1]),
             xytext=(iters * 0.55, J_history[-1] + 0.06),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
             fontsize=9, color='green')

# ----- 下方全宽：3D 概率曲面 g(x) = sigmoid(theta^T x) -----
ax3 = fig.add_subplot(gs[1, :], projection='3d')

# 用较粗的网格画曲面（性能考虑）
x1_surf, x2_surf = np.meshgrid(
    np.linspace(x1_min, x1_max, 80),
    np.linspace(X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05, 80)
)
surf_points = np.c_[x1_surf.ravel(), x2_surf.ravel()]
surf_scaled = (surf_points - X_mean) / X_std
X_surf = np.hstack([np.ones((surf_points.shape[0], 1)), surf_scaled])
z_surf = sigmoid(X_surf @ theta_scaled).reshape(x1_surf.shape)

# 绘制概率曲面
surf = ax3.plot_surface(x1_surf, x2_surf, z_surf,
                         cmap='RdYlGn', alpha=0.85,
                         edgecolor='none', antialiased=True)

# 绘制 h=0.5 决策边界线（在曲面上）
if abs(theta[2, 0]) > 1e-10:
    x2_db3d = -(theta[0, 0] + theta[1, 0] * x1_line) / theta[2, 0]
    x2_min_3d = X_raw[:, 1].min() - 0.05
    x2_max_3d = X_raw[:, 1].max() + 0.05
    mask_3d = (x2_db3d >= x2_min_3d) & (x2_db3d <= x2_max_3d)
    x1_db3d = x1_line[mask_3d]
    x2_db3d = x2_db3d[mask_3d]
    z_db3d = np.full_like(x1_db3d, 0.5)
    ax3.plot(x1_db3d, x2_db3d, z_db3d, 'k-', linewidth=3,
             label='h=0.5 Decision Boundary', zorder=10)

# h=0.5 水平参考平面（半透明）
x1_flat, x2_flat = np.meshgrid(
    np.linspace(x1_min, x1_max, 30),
    np.linspace(X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05, 30)
)
z_flat = np.full_like(x1_flat, 0.5)
ax3.plot_surface(x1_flat, x2_flat, z_flat, alpha=0.15, color='gray',
                 edgecolor='none', zorder=1)

# 训练数据点投影到曲面上
for i in range(m):
    xi1, xi2 = X_raw[i, 0], X_raw[i, 1]
    pi = h_train[i, 0]
    c = '#e74c3c' if y[i, 0] == 0 else '#2ecc71'
    mkr = 'x' if y[i, 0] == 0 else 'o'
    ax3.scatter([xi1], [xi2], [pi], c=c, marker=mkr, s=80, zorder=12)

# 新预测点
ax3.scatter([new_x[0, 0]], [new_x[0, 1]], [prob_new],
            c='#3498db', marker='*', s=250, zorder=15,
            label=f'New Building (P={prob_new:.1%})')

# 颜色条
cbar = fig.colorbar(surf, ax=ax3, shrink=0.6, aspect=20, pad=0.10)
cbar.set_label(f'Probability of {y_classes[1]}', fontsize=10)

ax3.set_xlabel(f'\n{x_names[0]}\n({x_units[0]})', fontsize=10, labelpad=8)
ax3.set_ylabel(f'\n{x_names[1]}\n({x_units[1]})', fontsize=10, labelpad=8)
ax3.set_zlabel('g(x) = P(y=1)', fontsize=10, labelpad=8)
ax3.set_title('(C) Probability Surface  g(x) = sigmoid(theta0 + theta1*x1 + theta2*x2)',
              fontsize=12, fontweight='bold', pad=15)

# 调整 3D 视角
ax3.view_init(elev=25, azim=-140)
ax3.legend(loc='upper left', fontsize=8)

# 3D 图与 tight_layout 不兼容，用手动调整
plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.06)
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: {FILENAME_BASE}.png")
print(f"代码文件: {FILENAME_BASE}.py")
```

---

## 代码模板 B：标准二分类 + 1 个特征

> 与模板 A 共用 1-7 段（数据→预测），仅可视化部分不同。
> 1 个特征无法画 2D 决策边界和 3D 概率曲面，降级为 **sigmoid 曲线 + 收敛曲线**。

将模板 A 的「第 8 段：可视化」替换为以下代码：

```python
# ========== 8. 可视化（1 特征降级：2 图）==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ----- 左图：Sigmoid 曲线 + 训练数据 + 预测点 -----
axes[0].scatter(X_raw, y, c=['#e74c3c' if yi == 0 else '#2ecc71' for yi in y.flatten()],
                marker='x' if y[0, 0] == 0 else 'o',  # 仅示意
                s=80, zorder=5)

# 分别绘制两类数据点（不同标记和颜色）
for yi_val in [0, 1]:
    mask = (y.flatten() == yi_val)
    if mask.any():
        c = '#e74c3c' if yi_val == 0 else '#2ecc71'
        mkr = 'x' if yi_val == 0 else 'o'
        axes[0].scatter(X_raw[mask], y[mask], c=c, marker=mkr, s=80,
                        label=y_classes[yi_val], zorder=5)

# 生成平滑 sigmoid 曲线
x_line = np.linspace(X_raw.min() * 0.9, X_raw.max() * 1.1, 200).reshape(-1, 1)
x_line_scaled = (x_line - X_mean) / X_std
X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
y_line = sigmoid(X_line @ theta_scaled)
axes[0].plot(x_line, y_line, 'b-', linewidth=2.5, label='g(x) = sigmoid(theta0 + theta1*x)')

# h=0.5 决策阈值线
axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
                label='Decision Threshold (h=0.5)')

# 决策阈值对应的 x 值
x_threshold = -theta[0, 0] / theta[1, 0]
axes[0].axvline(x=x_threshold, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

# 预测点
axes[0].scatter([new_x[0, 0]], [prob_new], c='#3498db', s=150, marker='*',
                zorder=10, label=f'New: x={new_x[0,0]}, P={prob_new:.1%}')

axes[0].set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
axes[0].set_ylabel('Probability', fontsize=11)
axes[0].set_title(f'(A) Sigmoid Curve: {title_topic}', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ----- 右图：收敛曲线 -----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost J(theta)', fontsize=11)
axes[1].set_title(f'(B) Ridge GD Convergence\n(alpha={alpha}, lambda={lam})',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 标注
axes[1].annotate(f'Start: {J_history[0]:.4f}',
                 xy=(0, J_history[0]), xytext=(iters * 0.08, J_history[0] + 0.08),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                 fontsize=9, color='red')
axes[1].annotate(f'End: {J_history[-1]:.4f}',
                 xy=(iters - 1, J_history[-1]),
                 xytext=(iters * 0.55, J_history[-1] + 0.06),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                 fontsize=9, color='green')

plt.tight_layout()
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: {FILENAME_BASE}.png")
print(f"代码文件: {FILENAME_BASE}.py")
```

---

## 占位符说明

| 占位符 | 规则 | 示例（2 特征） |
|--------|------|------|
| `{X_DATA}` | 特征矩阵，每行一个样本 | `[3, 0.20], [4, 0.25], [5, 0.28], ...` |
| `{Y_DATA}` | 标签列表（0 或 1）| `0, 0, 0, 0, 1, 1, 1, 1, 1, 1` |
| `{X_NAMES}` | Python list，特征中文名 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | Python list，特征单位 | `["cm", ""]` |
| `{Y_CLASSES}` | 类别映射字典 | `{0: "Not Efficient", 1: "Efficient"}` |
| `{y_name}` | 标签中文描述 | `"是否节能"` |
| `{title_topic}` | 英文预测主题 | `"Building Energy Efficiency Classification"` |
| `{ALPHA}` | 学习率 | `0.1` |
| `{LAMBDA}` | L2 正则化强度（0=关闭）| `0.1` |
| `{ITERS}` | 迭代次数 | `2000` |
| `{PREDICT_X}` | 新样本特征值（逗号分隔）| `7, 0.33` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名）| `logistic_regression_20260728_150000` |

---

## 关键注意事项

### 编码问题
Windows GBK 环境下，print 中的 Unicode 必须替换：`θ₀→theta0`, `θ₁→theta1`, `λ→lambda`。
matplotlib 的 label/title 不走 GBK，可以用 Unicode。

### 无头环境
必须 `matplotlib.use('Agg')`，用 `plt.close()` 代替 `plt.show()`。

### 特征缩放
逻辑回归的梯度下降**必须**做特征缩放。不同于线性回归有正规方程可选，逻辑回归只能用迭代法求解。z-score 标准化内置在所有模板中，无需用户确认。

### θ₀ 不参与正则化
截距项只控制决策边界的整体平移，不参与"扭曲"。惩罚它会导致模型偏向某一类（先验偏移）。

### 决策边界计算
当 `theta[2] ≈ 0` 时（第二个特征的系数接近零），决策边界接近垂直。代码中已处理这种特殊情况（`abs(theta[2, 0]) > 1e-10` 判断）。

### 3D 图与 tight_layout 不兼容
包含 3D 子图时使用 `plt.subplots_adjust()` 手动调整间距，不使用 `plt.tight_layout()`。

### 文件名
时间戳 `YYYYMMDD_HHMMSS`，不覆盖历史文件。
无论二分类/多分类/多项式，统一使用 `logistic_regression_` 前缀。

### 数据量检查
- 最少 `n_features + 2` 个数据点
- 样本数 < 5×n 时提醒用户结果可能不够稳健
- 二分类每类至少 3 个样本

### 参考文件
- `references/polynomial-logistic.md`：多项式特征 + 逻辑回归代码模板，**非线性决策边界时使用**。包含 `make_polynomial()` 特征构造函数和弯曲决策边界的可视化
- `references/multi-class-logistic.md`：多分类 Softmax 逻辑回归代码模板，**K > 2 时使用**。包含 Softmax 函数、One-hot 编码、决策区域可视化和 max-probability 3D 曲面



