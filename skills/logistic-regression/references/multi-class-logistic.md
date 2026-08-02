# 多分类逻辑回归（Multi-Class Logistic Regression）—— Softmax 回归

## 概述

多分类逻辑回归 = **One-hot 编码 + Softmax 函数 + 交叉熵代价 + L2 正则化梯度下降**。

当 y 有 K ≥ 3 个类别时（如能效等级 A/B/C/D），不能再用 sigmoid 函数（只输出一个概率）。Softmax 回归为每个类别训练一组参数，输出 K 个概率（和为 1），取最大概率对应的类别作为预测。

$$P(y=k|\mathbf{x}) = \frac{\exp(\theta_0^k + \theta_1^k x_1 + \theta_2^k x_2)}{\sum_{j=1}^{K}\exp(\theta_0^j + \theta_1^j x_1 + \theta_2^j x_2)}$$

> 参数矩阵 Θ 形状为 `(n+1, K)`——每列对应一个类别的权重向量。偏置行（第 0 行）不参与正则化。

## 何时使用（SKILL.md 步骤 2 触发）

| 条件 | 决策 |
|------|------|
| y 有 3 个或更多类别 | ✅ 使用本模板 |
| y 仅有 2 个类别 | ❌ 退回二分类（SKILL.md 内置模板或 polynomial-logistic.md） |

---

## 代码模板（多分类 Softmax + L2 正则化）

> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 特征矩阵 (m, n)
y_raw = np.array([{Y_DATA}])    # 标签 (m,)，值 0..K-1

m, n = X_raw.shape
K = {K}                          # 类别数
x_names = {X_NAMES}              # 特征名列表（中文）
x_units = {X_UNITS}              # 特征单位列表
class_names = {CLASS_NAMES}      # 类别名列表，如 ["A级", "B级", "C级"]
class_colors = {CLASS_COLORS}    # 类别颜色列表，如 ["#e74c3c", "#2ecc71", "#3498db"]

print(f"样本数: {m}, 特征数: {n}, 类别数: {K}")
print(f"类别分布:")
for k in range(K):
    count = np.sum(y_raw == k)
    print(f"  类别 {k} ({class_names[k]}): {count} 个样本")

# 确保 y 是列向量
y = y_raw.reshape(-1, 1)

# ========== 2. One-hot 编码 ==========
def one_hot(y, K):
    """
    将标签向量转为 one-hot 矩阵。

    输入: y (m, 1), 值 0..K-1
    输出: Y (m, K), 每行只有一个 1
    例: y=[0,2,1], K=3 → [[1,0,0],[0,0,1],[0,1,0]]
    """
    m_samples = len(y)
    Y = np.zeros((m_samples, K))
    for i in range(m_samples):
        Y[i, int(y[i, 0])] = 1
    return Y

Y = one_hot(y, K)   # (m, K)

# ========== 3. 特征缩放（z-score）==========
X_mean = X_raw.mean(axis=0, keepdims=True)   # (1, n)
X_std  = X_raw.std(axis=0, keepdims=True)    # (1, n)
X_scaled = (X_raw - X_mean) / X_std           # (m, n)

# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_scaled])   # (m, n+1)

print(f"\nX_mean: {X_mean}")
print(f"X_std: {X_std}")

# ========== 4. Softmax 函数 ==========
def softmax(z):
    """
    Softmax 函数 —— 将任意实数向量转为概率分布。

    z: (m, K) 矩阵
    返回: (m, K)，每行概率和为 1

    数值稳定技巧：先减去每行的最大值
    （因为 softmax(z) = softmax(z - max)）
    """
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ========== 5. 多分类 Softmax 梯度下降（含 L2 正则化）==========
def softmaxRidgeGD(X, Y, Theta, alpha, lam, iters):
    """
    Softmax 回归 + L2 正则化梯度下降。

    代价函数（交叉熵 + L2）:
      J(Theta) = -(1/m) * sum(sum(Y * log(H)))
                 + (lam/(2m)) * sum(Theta[1:, :]^2)

    梯度:
      grad = (1/m) * X^T @ (H - Y)
      grad[1:, :] += (lam/m) * Theta[1:, :]    （偏置第 0 行不惩罚）

    参数：
      X     - 特征矩阵 (m, n+1)，第 0 列为全 1
      Y     - One-hot 标签矩阵 (m, K)
      Theta - 参数矩阵 (n+1, K)，每列对应一个类别
      alpha - 学习率
      lam   - L2 正则化强度
      iters - 迭代次数
    """
    m_samples = len(Y)
    J_history = np.zeros(iters)

    for i in range(iters):
        z = X @ Theta                             # (m, K)
        H = softmax(z)                            # (m, K)
        errors = H - Y                            # (m, K)

        # 梯度：数据拟合项
        gradient = (1 / m_samples) * (X.T @ errors)  # (n+1, K)

        # L2 正则化梯度：偏置行（第 0 行）不参与
        gradient[1:, :] += (lam / m_samples) * Theta[1:, :]

        Theta = Theta - alpha * gradient

        # 代价：交叉熵 + 正则化
        cost = -(1 / m_samples) * np.sum(Y * np.log(H + 1e-10))
        reg_term = (lam / (2 * m_samples)) * np.sum(Theta[1:, :] ** 2)
        J_history[i] = cost + reg_term

    return Theta, J_history


# 初始化参数矩阵
Theta_scaled = np.zeros((n + 1, K))   # (n+1, K)
alpha = {ALPHA}
lam = {LAMBDA}
iters = {ITERS}

print(f"\n训练参数: alpha={alpha}, lambda={lam}, iters={iters}")

Theta_scaled, J_history = softmaxRidgeGD(X, Y, Theta_scaled, alpha, lam, iters)

print(f"初始代价: {J_history[0]:.6f}")
print(f"最终代价: {J_history[-1]:.6f}")

# ========== 6. 参数还原（从缩放空间回到原始空间）==========
# 对每个类别分别还原
Theta = np.zeros((n + 1, K))
for k in range(K):
    # 截距
    Theta[0, k] = Theta_scaled[0, k] - np.sum(
        Theta_scaled[1:, k] * X_mean[0] / X_std[0]
    )
    # 斜率
    for j in range(1, n + 1):
        Theta[j, k] = Theta_scaled[j, k] / X_std[0, j - 1]

# 打印每个类别的模型公式
print(f"\n各类别参数:")
for k in range(K):
    formula = f"  类别 {k} ({class_names[k]}): z_{k} = {Theta[0,k]:.6f}"
    for j in range(n):
        sign = '+' if Theta[j+1, k] >= 0 else ''
        formula += f" {sign} {Theta[j+1,k]:.6f}*{x_names[j]}"
    print(formula)

# ========== 7. 模型评估 ==========
z_train = X @ Theta_scaled
H_train = softmax(z_train)                     # (m, K) 各类别概率
y_pred = np.argmax(H_train, axis=1).reshape(-1, 1)  # (m, 1) 预测类别
accuracy = np.mean(y_pred == y) * 100

print(f"\n训练集准确率: {accuracy:.1f}%")

# 每类别准确率
print(f"每类别准确率:")
for k in range(K):
    mask = (y.flatten() == k)
    if mask.sum() > 0:
        class_acc = np.mean(y_pred[mask].flatten() == k) * 100
        print(f"  类别 {k} ({class_names[k]}): {class_acc:.1f}% ({mask.sum()} 个样本)")

# 训练集预测详情
print(f"\n训练集预测详情:")
for i in range(m):
    x_info = ", ".join([f"{X_raw[i,j]:.{2 if j==1 else 0}f}" for j in range(n)])
    probs_str = ", ".join([f"{class_names[k]}:{H_train[i,k]:.3f}" for k in range(K)])
    print(f"  样本{i+1}: x=({x_info}), "
          f"真实={class_names[int(y[i,0])]}, 预测={class_names[y_pred[i,0]]}, "
          f"概率=({probs_str})")

# ========== 8. 预测新样本 ==========
new_x = np.array([{PREDICT_X}])          # 新样本特征值，形状 (1, n)
new_x_scaled = (new_x - X_mean) / X_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
z_new = X_new @ Theta_scaled             # (1, K)
prob_new = softmax(z_new)[0]             # (K,) 各类别概率
pred_class = np.argmax(prob_new)         # 预测类别

print(f"\n{'='*50}")
print(f"  新样本预测")
print(f"  x = ({', '.join([f'{new_x[0,j]}' for j in range(n)])})")
for k in range(K):
    print(f"  P({class_names[k]}) = {prob_new[k]:.4f} ({prob_new[k]*100:.1f}%)")
print(f"  预测结果: {class_names[pred_class]}")
print(f"{'='*50}")

# ========== 9. 可视化 ==========
{VISUALIZATION}
```

---

## 可视化分支

### 分支 A：2 个特征（n == 2）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多分类可视化（2 特征）：决策区域 + 收敛 + 3D 曲面 ----
from matplotlib.colors import ListedColormap

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15],
              hspace=0.35, wspace=0.30)

# 创建离散色图（每个类别一种颜色）
class_cmap = ListedColormap(class_colors[:K])

# ----- 左上 (A)：决策区域 -----
ax = fig.add_subplot(gs[0, 0])

# 绘制训练数据点（按真实类别着色）
labels_seen = set()
for i in range(m):
    k = int(y[i, 0])
    if k not in labels_seen:
        ax.scatter(X_raw[i, 0], X_raw[i, 1], c=class_colors[k],
                   marker='o', s=120, label=class_names[k], zorder=5,
                   edgecolors='black', linewidth=0.5)
        labels_seen.add(k)
    else:
        ax.scatter(X_raw[i, 0], X_raw[i, 1], c=class_colors[k],
                   marker='o', s=120, zorder=5,
                   edgecolors='black', linewidth=0.5)

# 生成网格，计算每个格点的预测类别
x1_min, x1_max = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
x2_min_v = X_raw[:, 1].min() - 0.05
x2_max_v = X_raw[:, 1].max() + 0.05

x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min_v, x2_max_v, 300)
)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_scaled = (grid_points - X_mean) / X_std
X_grid = np.hstack([np.ones((grid_points.shape[0], 1)), grid_scaled])
H_grid = softmax(X_grid @ Theta_scaled)         # (90000, K)
y_grid = np.argmax(H_grid, axis=1).reshape(x1_grid.shape)  # 各类格点的预测类别

# 决策区域填充
ax.contourf(x1_grid, x2_grid, y_grid, levels=np.arange(K+1)-0.5,
            colors=class_colors[:K], alpha=0.15, zorder=0)

# 决策边界线（各类别之间的分界线）
ax.contour(x1_grid, x2_grid, y_grid, levels=np.arange(1, K),
           colors='black', linewidths=1.5, linestyles='-', zorder=2, alpha=0.6)

# 新预测点
ax.scatter([new_x[0, 0]], [new_x[0, 1]], c=class_colors[pred_class],
           marker='*', s=250, zorder=10,
           edgecolors='black', linewidth=1.5,
           label=f'New: -> {class_names[pred_class]}')

ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
ax.set_ylabel(f'{x_names[1]} ({x_units[1]})', fontsize=11)
ax.set_title(f'(A) Decision Regions (K={K})\nSoftmax Ridge, lambda={lam}',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min_v, x2_max_v)

# ----- 右上 (B)：交叉熵收敛曲线 -----
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(iters), J_history, 'b-', linewidth=1.5)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Cost J(Theta)', fontsize=11)
ax2.set_title(f'(B) Softmax GD Convergence\n(alpha={alpha}, lambda={lam}, K={K})',
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

# ----- 下方全宽 (C)：3D max-probability 曲面 -----
ax3 = fig.add_subplot(gs[1, :], projection='3d')

# 较粗网格
x1_surf, x2_surf = np.meshgrid(
    np.linspace(x1_min, x1_max, 80),
    np.linspace(x2_min_v, x2_max_v, 80)
)
surf_points = np.c_[x1_surf.ravel(), x2_surf.ravel()]
surf_scaled = (surf_points - X_mean) / X_std
X_surf = np.hstack([np.ones((surf_points.shape[0], 1)), surf_scaled])
H_surf = softmax(X_surf @ Theta_scaled)         # (6400, K)
max_prob_surf = np.max(H_surf, axis=1).reshape(x1_surf.shape)
pred_surf = np.argmax(H_surf, axis=1).reshape(x1_surf.shape)

# 创建面对的颜色数组（每个面按预测类别着色）
face_colors = np.zeros((x1_surf.shape[0] - 1, x1_surf.shape[1] - 1, 4))
for ki in range(K):
    for ri in range(pred_surf.shape[0] - 1):
        for ci in range(pred_surf.shape[1] - 1):
            # 简化：取四个角点最常见类别
            corner_classes = [pred_surf[ri, ci], pred_surf[ri+1, ci],
                            pred_surf[ri, ci+1], pred_surf[ri+1, ci+1]]
            # 取第一个角的类别（简化着色）
            kk = pred_surf[ri, ci]
            hex_color = class_colors[kk].lstrip('#')
            r, g, b = int(hex_color[0:2], 16)/255, int(hex_color[2:4], 16)/255, int(hex_color[4:6], 16)/255
            face_colors[ri, ci] = [r, g, b, 0.8]

# 绘制 max-probability 曲面（高度 = 最大概率，颜色 = 预测类别）
surf = ax3.plot_surface(x1_surf, x2_surf, max_prob_surf,
                         facecolors=face_colors,
                         edgecolor='none', antialiased=True, shade=True)

# 训练数据点投影到曲面上（每个点显示其被预测为真实类别的概率）
for i in range(m):
    xi1, xi2 = X_raw[i, 0], X_raw[i, 1]
    pi = H_train[i, int(y[i, 0])]   # 真实类别的概率
    ax3.scatter([xi1], [xi2], [pi], c=class_colors[int(y[i, 0])],
                marker='o', s=60, zorder=12, edgecolors='black', linewidth=0.3)

# 新预测点（显示预测类别的概率）
ax3.scatter([new_x[0, 0]], [new_x[0, 1]], [prob_new[pred_class]],
            c=class_colors[pred_class], marker='*', s=250, zorder=15,
            edgecolors='black', linewidth=1.5,
            label=f'New -> {class_names[pred_class]} (P={prob_new[pred_class]:.1%})')

# 颜色条（使用离散颜色）
ax3.set_xlabel(f'\n{x_names[0]}\n({x_units[0]})', fontsize=10, labelpad=8)
ax3.set_ylabel(f'\n{x_names[1]}\n({x_units[1]})', fontsize=10, labelpad=8)
ax3.set_zlabel('max_k P(y=k|x)', fontsize=10, labelpad=8)
ax3.set_title(f'(C) Max-Probability Surface (K={K})',
              fontsize=12, fontweight='bold', pad=15)

ax3.view_init(elev=25, azim=-140)
ax3.legend(loc='upper left', fontsize=8)

# 添加手动图例（各类别颜色）
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=class_colors[k], alpha=0.8,
                        label=f'Class {k}: {class_names[k]}') for k in range(K)]
ax3.legend(handles=legend_patches, loc='upper left', fontsize=8)

plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.06)
```

### 分支 B：1 个特征（n == 1）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多分类可视化（1 特征）：K 条概率曲线 + 收敛曲线 ----

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ----- 左图 (A)：K 条概率曲线 vs 特征值 -----
# 绘制训练数据（按类别着色）
for k in range(K):
    mask = (y.flatten() == k)
    if mask.any():
        axes[0].scatter(X_raw[mask], y[mask], c=class_colors[k],
                        marker='o', s=80, label=class_names[k], zorder=5,
                        edgecolors='black', linewidth=0.3)

# 生成平滑概率曲线（每个类别一条）
x_line = np.linspace(X_raw.min() * 0.9, X_raw.max() * 1.1, 200).reshape(-1, 1)
x_line_scaled = (x_line - X_mean) / X_std
X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
H_line = softmax(X_line @ Theta_scaled)   # (200, K)

for k in range(K):
    axes[0].plot(x_line, H_line[:, k], color=class_colors[k],
                 linewidth=2, label=f'P({class_names[k]})')

# 预测点
for k in range(K):
    axes[0].scatter([new_x[0, 0]], [prob_new[k]], c=class_colors[k],
                    marker='*', s=150, zorder=10, alpha=0.7)

axes[0].set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
axes[0].set_ylabel('Probability', fontsize=11)
axes[0].set_title(f'(A) Softmax Probability Curves (K={K})', fontsize=12,
                  fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ----- 右图 (B)：交叉熵收敛曲线 -----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost J(Theta)', fontsize=11)
axes[1].set_title(f'(B) Softmax GD Convergence\n(alpha={alpha}, lambda={lam}, K={K})',
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

### 分支 C：>= 3 个特征（n >= 3）→ 用此段替换 `{VISUALIZATION}`

```python
# ---- 多分类可视化（>= 3 特征）：预测概率分布 + 收敛曲线 ----

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ----- 左图 (A)：预测概率分布 -----
# 按真实类别分组，显示每个样本被预测为真实类别的概率
for k in range(K):
    mask = (y.flatten() == k)
    if mask.any():
        prob_true_class = H_train[mask, k]
        axes[0].scatter(np.arange(m)[mask], prob_true_class,
                        c=class_colors[k], s=80, zorder=5, label=class_names[k])
axes[0].axhline(y=1.0/K, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                label=f'Random Guess (1/{K})')
axes[0].set_xlabel('Sample Index', fontsize=11)
axes[0].set_ylabel('P(True Class)', fontsize=11)
axes[0].set_title(f'(A) Predicted P(True Class)\nAccuracy={accuracy:.1f}%', fontsize=12,
                  fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ----- 右图 (B)：交叉熵收敛曲线 -----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost J(Theta)', fontsize=11)
axes[1].set_title(f'(B) Softmax GD Convergence\n(alpha={alpha}, lambda={lam}, K={K})',
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

## 多项式扩展（可选）

如果 SKILL.md 步骤 3 判定需要多项式特征，在模板的步骤 3（特征缩放）之前插入以下代码：

```python
# ========== 2b. 构造多项式特征（可选）==========
degree = {DEGREE}

def make_polynomial(X_raw, degree):
    """为每个特征构造 x, x^2, ..., x^degree。不含交互项。"""
    m_samples, n_base = X_raw.shape
    features = []
    for j in range(n_base):
        for d in range(1, degree + 1):
            features.append((X_raw[:, j] ** d).reshape(-1, 1))
    return np.hstack(features)

X_raw_poly = make_polynomial(X_raw, degree)
n_orig = n
n = X_raw_poly.shape[1]

# 更新特征名
poly_names = []
for base_name in x_names:
    for d in range(1, degree + 1):
        if d == 1:
            poly_names.append(base_name)
        else:
            poly_names.append(f"{base_name}^{d}")
x_names = poly_names
x_units = x_units * degree   # 单位延续

print(f"多项式特征: {n_orig} -> {n}")
X_raw = X_raw_poly   # 后续代码继续用 X_raw（现在是多项式特征）
```

然后将步骤 3 的 `n` 替换为多项式后的 `n`（实际特征数）。

> 注意：多项式扩展后，可视化仍然是 2D（决策区域仍是弯曲的），但 n_features 含义变了——`X_raw[:, 0]` 不再是原始特征，而是多项式特征的第一列。可视化代码中直接绘制 `X_raw[:, 0]` vs `X_raw[:, 1]` 的含义也会变。对于多项式 + 多分类 + 2 基础特征的情况，决策区域图仍然可用（每个格点也做多项式扩展后判断类别），但轴标签需要调整。

---

## 占位符说明

| 占位符 | 规则 | 示例 |
|--------|------|------|
| `{X_DATA}` | 特征矩阵 | `[3, 0.20], [4, 0.25], ...` |
| `{Y_DATA}` | 标签（0..K-1）| `0, 0, 1, 1, 2, 2, ...` |
| `{K}` | 类别数（整数）| `3` |
| `{X_NAMES}` | Python list，特征中文名 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | Python list，特征单位 | `["cm", ""]` |
| `{CLASS_NAMES}` | Python list，各类别中文名 | `["A级", "B级", "C级"]` |
| `{CLASS_COLORS}` | Python list，各类别颜色（hex）| `["#e74c3c", "#2ecc71", "#3498db"]` |
| `{ALPHA}` | 学习率 | `0.05` |
| `{LAMBDA}` | L2 正则化强度（0=关闭）| `0.1` |
| `{ITERS}` | 迭代次数 | `3000` |
| `{PREDICT_X}` | 新样本特征值 | `7, 0.33` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名）| `logistic_regression_20260728_150000` |
| `{VISUALIZATION}` | 按 n 选择可视化分支 | 见上方分支 A/B/C 代码 |

---

## 关键注意

1. **参数是矩阵不是向量**：Θ 形状为 `(n+1, K)`，每列对应一个类别的权重。这意味着参数数量是二分类的 K 倍
2. **Softmax 数值稳定**：`z - max(z)` 技巧防止 exp 溢出。如果 z 值很大（如 1000），`exp(1000) = inf`，会得到 NaN
3. **交叉熵代价**：`-sum(Y * log(H + 1e-10))` 中的 `+1e-10` 防止 log(0)
4. **偏置行不参与正则化**：`gradient[1:, :] += ...` 跳过第 0 行，K 个类别的截距都不被惩罚
5. **α 建议 0.05**：K 组参数同时更新，需要比二分类（α=0.1）更保守的学习率
6. **iters 默认 3000**：参数多了 K 倍，收敛需要更多迭代
7. **每个类别至少 3 个样本**：否则 Softmax 概率估计极不稳定
8. **决策区域可视化**（分支 A）：使用 `contourf` 填充各区域的预测类别，颜色用离散 `ListedColormap`。黑色轮廓线分隔各类别区域
9. **3D max-probability 曲面**：高度 = `max_k P(y=k|x)`（表示模型对该点分类的"自信度"），颜色 = 预测类别。决策边界附近曲面较低（不确定），远离边界处曲面较高（接近 1）
10. **类别颜色**：K ≤ 5 时默认用红/绿/蓝/橙/紫。K > 5 时使用 tableau 调色板自动扩展
