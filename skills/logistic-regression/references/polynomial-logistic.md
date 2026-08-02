# 多项式逻辑回归（Polynomial Logistic Regression）—— 弯曲的决策边界

## 概述

多项式逻辑回归 = **多项式特征构造 + 特征缩放 + 逻辑回归 + L2 正则化梯度下降**。

当两个类之间的边界不是直线时，标准逻辑回归（θ₀+θ₁x₁+θ₂x₂=0）无法很好地区分。通过构造 x², x³ 等高次特征，决策边界可以弯曲（椭圆、双曲线等形状）。

$$g(\mathbf{x}) = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_2 + \theta_4 x_2^2)$$

> 核心思想：在数据预处理阶段手动构造多项式特征，然后用逻辑回归求解。**logisticRidgeGD 函数一字不改。**

## 何时使用（SKILL.md 步骤 3 触发）

| 条件 | 建议 |
|------|------|
| 数据散点图显示两类无法用直线分开 | **强烈建议使用本模板** |
| 用户描述"先好后差再转好"等非线性分类模式 | **建议使用本模板** |
| 直线逻辑回归准确率 < 70% | **建议尝试多项式** |
| degree 越高越灵活，但也越容易过拟合 | degree 一般 ≤ 3 |
| m/n_poly < 3 | 强烈建议正则化，λ = 5.0 |
| 3 ≤ m/n_poly < 5 | 建议正则化，λ = 1.0 |
| 5 ≤ m/n_poly < 10 | 轻度正则化，λ = 0.5 |
| m/n_poly ≥ 10 | 轻微正则化，λ = 0.1 |

> **本模板默认使用 L2 正则化**。λ = 0 时退化为标准梯度下降多项式逻辑回归（无正则化）。

---

## 代码模板（多项式特征 + 逻辑回归 + L2 正则化）

> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 基础特征矩阵 (m, n_base)
y = np.array([{Y_DATA}]).reshape(-1, 1)

m, n_base = X_raw.shape
x_names = {X_NAMES}     # 基础特征名列表（中文）
x_units = {X_UNITS}     # 基础特征单位列表
y_classes = {Y_CLASSES}  # 类别映射，如 {0: "Not Efficient", 1: "Efficient"}
degree = {DEGREE}        # 多项式最高次数

print(f"样本数: {m}, 基础特征数: {n_base}, degree={degree}")

# ========== 2. 构造多项式特征 ==========
def make_polynomial(X_raw, degree):
    """
    为每个特征构造 x, x^2, ..., x^degree。
    不含交互项（如 x1*x2），保持可解释性。

    输入: X_raw (m, n_base)
    输出: X_poly (m, n_base * degree)
    """
    m_samples, n_base = X_raw.shape
    features = []
    for j in range(n_base):
        for d in range(1, degree + 1):
            features.append((X_raw[:, j] ** d).reshape(-1, 1))
    return np.hstack(features)

X_poly = make_polynomial(X_raw, degree)   # (m, n_poly)
n = X_poly.shape[1]                       # 多项式特征总数

# 打印特征名（帮助理解）
poly_names = []
for base_name in x_names:
    for d in range(1, degree + 1):
        if d == 1:
            poly_names.append(base_name)
        else:
            poly_names.append(f"{base_name}^{d}")
print(f"构造的多项式特征: {poly_names}")
print(f"特征数: {n_base} -> {n}")

# ========== 3. 特征缩放（z-score 在多项式特征上做）==========
X_poly_mean = X_poly.mean(axis=0, keepdims=True)   # (1, n)
X_poly_std  = X_poly.std(axis=0, keepdims=True)    # (1, n)
X_scaled = (X_poly - X_poly_mean) / X_poly_std     # (m, n)

# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_scaled])          # (m, n+1)

# ========== 4. Sigmoid 函数 ==========
def sigmoid(z):
    """逻辑函数，将任意实数映射到 (0,1)"""
    return 1 / (1 + np.exp(-z))

# ========== 5. 逻辑回归 + L2 正则化梯度下降 ==========
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
      lam    - L2 正则化强度
      iters  - 迭代次数
    """
    m_samples = len(y)
    J_history = np.zeros(iters)

    for i in range(iters):
        z = X @ theta
        h = sigmoid(z)
        errors = h - y

        gradient = (1 / m_samples) * (X.T @ errors)
        gradient[1:] += (lam / m_samples) * theta[1:]

        theta = theta - alpha * gradient

        cost = -(1 / m_samples) * np.sum(
            y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10)
        )
        reg_term = (lam / (2 * m_samples)) * np.sum(theta[1:] ** 2)
        J_history[i] = cost + reg_term

    return theta, J_history


theta_scaled = np.zeros((n + 1, 1))
alpha = {ALPHA}
lam = {LAMBDA}
iters = {ITERS}

print(f"\n训练参数: alpha={alpha}, lambda={lam}, iters={iters}")

theta_scaled, J_history = logisticRidgeGD(X, y, theta_scaled, alpha, lam, iters)

print(f"初始代价: {J_history[0]:.6f}")
print(f"最终代价: {J_history[-1]:.6f}")

# ========== 6. 参数还原（从缩放空间回到原始空间）==========
# 重要：使用 X_poly_mean / X_poly_std，不是 X_raw 的！
theta = np.zeros((n + 1, 1))
theta[0, 0] = theta_scaled[0, 0] - np.sum(
    theta_scaled[1:, 0] * X_poly_mean[0] / X_poly_std[0]
)
for j in range(1, n + 1):
    theta[j, 0] = theta_scaled[j, 0] / X_poly_std[0, j - 1]

# 打印模型公式
formula = f"g(x) = sigmoid({theta[0,0]:.6f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign} {theta[j+1, 0]:.6f}*{poly_names[j]}"
formula += ")"
print(f"\n模型公式: {formula}")

# ========== 7. 模型评估 ==========
z_train = X @ theta_scaled
h_train = sigmoid(z_train)
y_pred = (h_train >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100

print(f"\n训练集准确率: {accuracy:.1f}%")
print(f"训练集预测详情:")
for i in range(m):
    x_info = ", ".join([f"{X_raw[i,j]:.{2 if j==1 else 0}f}" for j in range(n_base)])
    print(f"  样本{i+1}: x=({x_info}), "
          f"真实y={int(y[i,0])}, 概率={h_train[i,0]:.4f}, 预测y={y_pred[i,0]}")

# ========== 8. 预测新样本 ==========
new_x_raw = np.array([{PREDICT_X}])   # 新样本基础特征值，形状 (1, n_base)
new_x_poly = make_polynomial(new_x_raw, degree)
new_x_scaled = (new_x_poly - X_poly_mean) / X_poly_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
prob_new = sigmoid(X_new @ theta_scaled)[0, 0]
pred_new = 1 if prob_new >= 0.5 else 0

print(f"\n{'='*50}")
print(f"  新样本预测")
print(f"  x = ({', '.join([f'{new_x_raw[0,j]}' for j in range(n_base)])})")
print(f"  预测概率 = {prob_new:.4f} ({prob_new*100:.1f}%)")
print(f"  预测结果: {y_classes[pred_new]}")
print(f"{'='*50}")

# ========== 9. 可视化 ==========
{VISUALIZATION}
```
```

---

## 可视化分支

### 分支 A：2 个基础特征（n_base == 2）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多项式逻辑回归可视化（2 基础特征）----
# 3 图输出：曲线决策边界 + 收敛曲线 + 3D 概率曲面

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15],
              hspace=0.35, wspace=0.30)

# ----- 左上 (A)：曲线决策边界 -----
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

# 生成网格，计算每个格点的概率（用于等高线）
x1_min, x1_max = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
x2_min_v = X_raw[:, 1].min() - 0.05
x2_max_v = X_raw[:, 1].max() + 0.05

x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min_v, x2_max_v, 300)
)
# 网格点也要做多项式特征
grid_points_raw = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_points_poly = make_polynomial(grid_points_raw, degree)
grid_points_scaled = (grid_points_poly - X_poly_mean) / X_poly_std
X_grid = np.hstack([np.ones((grid_points_raw.shape[0], 1)), grid_points_scaled])
z_grid = sigmoid(X_grid @ theta_scaled).reshape(x1_grid.shape)

# 概率等高线填充
ax.contourf(x1_grid, x2_grid, z_grid, levels=20, alpha=0.15,
            cmap='RdYlGn', zorder=0)
# h=0.5 决策边界等高线（现在是曲线！）
ax.contour(x1_grid, x2_grid, z_grid, levels=[0.5], colors='blue',
           linewidths=2.5, zorder=3, linestyles='-')

# 也可以画 h=0.3 和 h=0.7 的辅助线
ax.contour(x1_grid, x2_grid, z_grid, levels=[0.3, 0.7], colors=['gray', 'gray'],
           linewidths=1, linestyles='--', alpha=0.5, zorder=2)

# 新预测点
pred_color = '#2ecc71' if pred_new == 1 else '#e74c3c'
ax.scatter([new_x_raw[0, 0]], [new_x_raw[0, 1]], c=pred_color, s=200,
           marker='*', zorder=10,
           label=f'New: P={prob_new:.1%} -> {y_classes[pred_new]}')

ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
ax.set_ylabel(f'{x_names[1]} ({x_units[1]})', fontsize=11)
ax.set_title(f'(A) Curved Decision Boundary (degree={degree})\nLogistic Ridge, lambda={lam}',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min_v, x2_max_v)

# ----- 右上 (B)：收敛曲线 -----
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(iters), J_history, 'b-', linewidth=1.5)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Cost J(theta)', fontsize=11)
ax2.set_title(f'(B) Ridge GD Convergence\n(alpha={alpha}, lambda={lam}, degree={degree})',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax2.annotate(f'Start: {J_history[0]:.4f}',
             xy=(0, J_history[0]), xytext=(iters * 0.08, J_history[0] + 0.08),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             fontsize=9, color='red')
ax2.annotate(f'End: {J_history[-1]:.4f}',
             xy=(iters - 1, J_history[-1]),
             xytext=(iters * 0.55, J_history[-1] + 0.06),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
             fontsize=9, color='green')

# ----- 下方全宽 (C)：3D 多项式概率曲面 -----
ax3 = fig.add_subplot(gs[1, :], projection='3d')

# 曲面网格（较粗，性能考虑）
x1_surf, x2_surf = np.meshgrid(
    np.linspace(x1_min, x1_max, 80),
    np.linspace(x2_min_v, x2_max_v, 80)
)
surf_points_raw = np.c_[x1_surf.ravel(), x2_surf.ravel()]
surf_points_poly = make_polynomial(surf_points_raw, degree)
surf_points_scaled = (surf_points_poly - X_poly_mean) / X_poly_std
X_surf = np.hstack([np.ones((surf_points_raw.shape[0], 1)), surf_points_scaled])
z_surf = sigmoid(X_surf @ theta_scaled).reshape(x1_surf.shape)

# 绘制概率曲面（多项式逻辑回归的 sigmoid 曲面也是 S 型的！）
surf = ax3.plot_surface(x1_surf, x2_surf, z_surf,
                         cmap='RdYlGn', alpha=0.85,
                         edgecolor='none', antialiased=True)

# h=0.5 参考平面（半透明）
x1_flat, x2_flat = np.meshgrid(
    np.linspace(x1_min, x1_max, 30),
    np.linspace(x2_min_v, x2_max_v, 30)
)
z_flat = np.full_like(x1_flat, 0.5)
ax3.plot_surface(x1_flat, x2_flat, z_flat, alpha=0.15, color='gray',
                 edgecolor='none', zorder=1)

# h=0.5 决策线（在曲面上，曲线而非直线）
# 采样较密的网格找到 h≈0.5 的点
x1_db = np.linspace(x1_min, x1_max, 200)
x2_db = np.linspace(x2_min_v, x2_max_v, 200)
X1_db, X2_db = np.meshgrid(x1_db, x2_db)
db_points_raw = np.c_[X1_db.ravel(), X2_db.ravel()]
db_points_poly = make_polynomial(db_points_raw, degree)
db_points_scaled = (db_points_poly - X_poly_mean) / X_poly_std
X_db = np.hstack([np.ones((db_points_raw.shape[0], 1)), db_points_scaled])
Z_db = sigmoid(X_db @ theta_scaled).reshape(X1_db.shape)

# 把靠近 h=0.5 的等高线画出来
ax3.contour(X1_db, X2_db, Z_db, levels=[0.5], colors='black',
            linewidths=3, zorder=10)

# 训练数据点投影到曲面上
for i in range(m):
    xi1, xi2 = X_raw[i, 0], X_raw[i, 1]
    pi = h_train[i, 0]
    c = '#e74c3c' if y[i, 0] == 0 else '#2ecc71'
    mkr = 'x' if y[i, 0] == 0 else 'o'
    ax3.scatter([xi1], [xi2], [pi], c=c, marker=mkr, s=80, zorder=12)

# 新预测点
ax3.scatter([new_x_raw[0, 0]], [new_x_raw[0, 1]], [prob_new],
            c='#3498db', marker='*', s=250, zorder=15,
            label=f'New (P={prob_new:.1%})')

# 颜色条
cbar = fig.colorbar(surf, ax=ax3, shrink=0.6, aspect=20, pad=0.10)
cbar.set_label(f'Probability of {y_classes[1]}', fontsize=10)

ax3.set_xlabel(f'\n{x_names[0]}\n({x_units[0]})', fontsize=10, labelpad=8)
ax3.set_ylabel(f'\n{x_names[1]}\n({x_units[1]})', fontsize=10, labelpad=8)
ax3.set_zlabel('g(x) = P(y=1)', fontsize=10, labelpad=8)
ax3.set_title(f'(C) Polynomial Probability Surface (degree={degree})',
              fontsize=12, fontweight='bold', pad=15)

ax3.view_init(elev=25, azim=-140)
ax3.legend(loc='upper left', fontsize=8)

plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.06)
```

### 分支 B：1 个基础特征（n_base == 1）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多项式逻辑回归可视化（1 基础特征）----
# 2 图输出：多项式 sigmoid 曲线 + 收敛曲线

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ----- 左图 (A)：多项式 sigmoid 曲线 -----
# 分别绘制两类数据点
for yi_val in [0, 1]:
    mask = (y.flatten() == yi_val)
    if mask.any():
        c = '#e74c3c' if yi_val == 0 else '#2ecc71'
        mkr = 'x' if yi_val == 0 else 'o'
        axes[0].scatter(X_raw[mask], y[mask], c=c, marker=mkr, s=80,
                        label=y_classes[yi_val], zorder=5)

# 生成平滑多项式 sigmoid 曲线
x_line = np.linspace(X_raw.min() * 0.9, X_raw.max() * 1.1, 200).reshape(-1, 1)
x_line_poly = make_polynomial(x_line, degree)
x_line_scaled = (x_line_poly - X_poly_mean) / X_poly_std
X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
y_line = sigmoid(X_line @ theta_scaled)
axes[0].plot(x_line, y_line, 'b-', linewidth=2.5,
             label=f'g(x) = sigmoid(poly(degree={degree}))')

# h=0.5 决策阈值线
axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
                label='Decision Threshold (h=0.5)')

# 预测点
axes[0].scatter([new_x_raw[0, 0]], [prob_new], c='#3498db', s=150, marker='*',
                zorder=10, label=f'New: x={new_x_raw[0,0]}, P={prob_new:.1%}')

axes[0].set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
axes[0].set_ylabel('Probability', fontsize=11)
axes[0].set_title(f'(A) Polynomial Sigmoid Curve (degree={degree})', fontsize=12,
                  fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ----- 右图 (B)：收敛曲线 -----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost J(theta)', fontsize=11)
axes[1].set_title(f'(B) Ridge GD Convergence\n(alpha={alpha}, lambda={lam}, degree={degree})',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

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
```

### 分支 C：>= 3 个基础特征（n_base >= 3）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多项式逻辑回归可视化（>= 3 基础特征）----
# 无法画 2D/3D 图，降级为真实值 vs 预测概率 + 收敛曲线

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ----- 左图 (A)：预测概率分布 -----
# 按真实类别分组显示预测概率
for yi_val in [0, 1]:
    mask = (y.flatten() == yi_val)
    if mask.any():
        c = '#e74c3c' if yi_val == 0 else '#2ecc71'
        axes[0].scatter(np.arange(m)[mask], h_train[mask],
                        c=c, s=80, zorder=5, label=y_classes[yi_val])
axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
axes[0].set_xlabel('Sample Index', fontsize=11)
axes[0].set_ylabel('Predicted Probability', fontsize=11)
axes[0].set_title(f'(A) Predicted Probabilities\nAccuracy={accuracy:.1f}%', fontsize=12,
                  fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ----- 右图 (B)：收敛曲线 -----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost J(theta)', fontsize=11)
axes[1].set_title(f'(B) Ridge GD Convergence\n(alpha={alpha}, lambda={lam}, n_poly={n})',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
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
```

选择对应分支后，在模板末尾加上保存代码：

```python
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: {FILENAME_BASE}.png")
print(f"代码文件: {FILENAME_BASE}.py")
```

---

## 占位符说明

| 占位符 | 规则 | 示例（2 基础特征 degree=2） |
|--------|------|------|
| `{X_DATA}` | 基础特征的原始值 | `[3, 0.20], [4, 0.25], ...` |
| `{Y_DATA}` | 标签值（0 或 1）| `0, 0, 0, 0, 1, 1, 1, 1, 1, 1` |
| `{X_NAMES}` | Python list，基础特征中文名 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | Python list，基础特征单位 | `["cm", ""]` |
| `{Y_CLASSES}` | 类别映射字典 | `{0: "Not Efficient", 1: "Efficient"}` |
| `{DEGREE}` | 多项式最高次数 | `2` |
| `{ALPHA}` | 学习率 | `0.05` |
| `{LAMBDA}` | L2 正则化强度（0=关闭）| `0.5` |
| `{ITERS}` | 迭代次数 | `3000` |
| `{PREDICT_X}` | 新样本基础特征值 | `7, 0.33` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名）| `logistic_regression_20260728_150000` |
| `{VISUALIZATION}` | 按 n_base 选择可视化分支 | 见上方的分支 A/B/C 代码 |

---

## 关键注意

1. **缩放对象是 X_poly，不是 X_raw**：`X_poly_mean`/`X_poly_std` 的计算和参数还原都要用多项式特征空间的统计量
2. **参数数量 = n_base × degree + 1**：degree=2 时每个基础特征产生 2 个多项式特征
3. **决策边界是曲线**：标准逻辑回归的决策边界是 `θ₀+θ₁x₁+θ₂x₂=0`（直线方程），多项式逻辑回归的边界变成 `θ₀+θ₁x₁+θ₂x₁²+θ₃x₂+θ₄x₂²=0`（二次曲线）
4. **θ₀ 不参与正则化**：截距项只控制整体平移，不参与"扭曲"
5. **iters 默认 3000**：多项式特征空间的梯度下降需要更多迭代才能收敛
6. **α 建议 0.05**：多项式特征量级差异更大（x vs x²），需要较小的学习率
7. **degree 不要太大**：degree=2 覆盖大多数场景；degree=3 可处理更复杂曲线；degree ≥ 4 极易过拟合，必须配合较大 λ
8. **λ 与 m/n_poly 联动**：样本少/特征多时加大 λ，样本充裕时可减小 λ
9. **文件名保持统一**：仍用 `logistic_regression_` 前缀，不由是否多项式来区分文件名
10. **可视化分支选择**：n_base=2 时输出 3 图（弯曲边界 + 收敛 + 3D 曲面）；n_base=1 时输出 2 图（多项式 sigmoid 曲线 + 收敛）；n_base≥3 时降级为预测概率分布图
