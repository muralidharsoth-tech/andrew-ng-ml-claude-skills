# 岭回归（Ridge Regression）—— 多项式 + 梯度下降 + L2 正则化

## 概述

岭回归 = 多项式特征构造 + 特征缩放 + 梯度下降 + L2 正则化惩罚项。

$$\text{代价函数：} J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2 \;+\; \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

$$\text{梯度：} \nabla J = \frac{1}{m}X^T(h-y) \;+\; \frac{\lambda}{m}\theta \quad\text{（}\theta_0\text{ 不参与惩罚）}$$

> 核心思想：在代价函数中增加参数平方和的惩罚项，迫使 θ 保持较小值 → 防止高次多项式剧烈扭曲（过拟合）。
>
> **λ = 0 时退化为标准梯度下降多项式回归（无正则化）**。

## 何时使用（SKILL.md 步骤 4b 触发）

| 条件 | 建议 |
|------|------|
| 多项式回归（几乎所有情况）| **强烈建议使用本模板** |
| m / n_poly < 3 | 强烈建议，λ = 5.0 |
| 3 <= m / n_poly < 5 | 建议，λ = 1.0 |
| 5 <= m / n_poly < 10 | 轻度建议，λ = 0.5 |
| m / n_poly >= 10 且 degree <= 2 | 可不正则化（λ=0） |
| 普通线性回归 m/n < 5 | 可选使用 |

---

## 代码模板（多项式特征 + 岭回归梯度下降）

> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。
> 多项式回归文件名前缀：`polynomial_regression_`。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 基础特征矩阵 (m, n_base)
y = np.array([{Y_DATA}]).reshape(-1, 1)

m, n_base = X_raw.shape
x_names = {X_NAMES}   # 基础特征名列表（中文）
x_units = {X_UNITS}   # 基础特征单位列表
degree = {DEGREE}     # 多项式最高次数

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

# ========== 3. 特征缩放 + 构造 X（加偏置列）==========
# z-score 标准化：只在多项式特征上做，偏置列除外
X_poly_mean = X_poly.mean(axis=0, keepdims=True)   # (1, n)
X_poly_std  = X_poly.std(axis=0, keepdims=True)    # (1, n)
X_scaled = (X_poly - X_poly_mean) / X_poly_std     # (m, n)

# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_scaled])          # (m, n+1)

# ========== 4. 岭回归梯度下降 ==========
def ridgeGradientDescent(X, y, theta, alpha, iters, lam):
    """
    岭回归梯度下降（L2 正则化）。

    - 偏置项 theta[0] 不参与正则化（截距只控制平移，不参与"扭曲"）
    - 正则化梯度: grad[1:] += (lam/m) * theta[1:]
    - lam=0 时退化为标准梯度下降
    """
    m_samples = len(y)
    J_history = np.zeros(iters)

    for i in range(iters):
        predictions = X @ theta                         # (m, n+1) @ (n+1, 1) -> (m, 1)
        errors = predictions - y                        # (m, 1)
        gradient = (1 / m_samples) * (X.T @ errors)    # (n+1, 1)

        # L2 正则化梯度：跳过偏置项 theta[0]
        gradient[1:] += (lam / m_samples) * theta[1:]

        theta = theta - alpha * gradient

        # 代价函数（含正则化项，用于监控收敛）
        mse = (1 / (2 * m_samples)) * np.sum(errors ** 2)
        reg_term = (lam / (2 * m_samples)) * np.sum(theta[1:] ** 2)
        J_history[i] = mse + reg_term

    return theta, J_history


theta_scaled = np.zeros((n + 1, 1))
alpha = {ALPHA}
iters = {ITERS}
lam = {LAMBDA}    # λ=0 时等价于标准梯度下降（无正则化）

theta_scaled, J_history = ridgeGradientDescent(X, y, theta_scaled, alpha, iters, lam)

# ========== 5. 参数还原（从缩放空间回到原始空间）==========
theta = np.zeros((n + 1, 1))
# 截距: theta0_orig = theta0_scaled - sum(theta_j_scaled * mean_j / std_j)
theta[0, 0] = theta_scaled[0, 0] - np.sum(
    theta_scaled[1:, 0] * X_poly_mean[0] / X_poly_std[0]
)
# 斜率: theta_j_orig = theta_j_scaled / std_j
for j in range(1, n + 1):
    theta[j, 0] = theta_scaled[j, 0] / X_poly_std[0, j - 1]

# 打印模型公式
formula = f"{y_name} = {theta[0,0]:.6f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign} {theta[j+1, 0]:.6f} * {poly_names[j]}"
print(f"模型公式: {formula}")

# ========== 6. 预测 ==========
new_x_raw = np.array([{PREDICT_X}])   # 新样本基础特征值，形状 (1, n_base)
new_x_poly = make_polynomial(new_x_raw, degree)
new_x_scaled = (new_x_poly - X_poly_mean) / X_poly_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
prediction = (X_new @ theta_scaled)[0, 0]
print(f"预测{y_name}: {prediction:.2f} {y_unit}")

# ========== 7. 拟合优度指标 ==========
y_pred_all = X @ theta_scaled            # 所有样本的预测值
residuals = y - y_pred_all               # 残差
ss_res = np.sum(residuals ** 2)          # 残差平方和
ss_tot = np.sum((y - y.mean()) ** 2)     # 总离差平方和
r2 = 1 - ss_res / ss_tot                # R² 决定系数
rmse = np.sqrt(np.mean(residuals ** 2))  # 均方根误差

print(f"R^2 = {r2:.4f} (越接近1拟合越好)")
print(f"RMSE = {rmse:.4f} {y_unit}")
if lam > 0:
    theta_norm = np.sqrt(np.sum(theta_scaled[1:] ** 2))
    print(f"正则化: lambda={lam}, 最终代价 J={J_history[-1]:.6f}, L2参数范数={theta_norm:.4f}")
else:
    print(f"最终代价 J={J_history[-1]:.6f} (无正则化)")

# ========== 8. 可视化 ==========
{N_SINGLE_VIS}
{N_MULTI_VIS}

plt.tight_layout()
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print("图片已保存: {FILENAME_BASE}.png")
```

---

## 可视化分支

### 分支 A：单变量基础特征（n_base == 1）→ 用此段替换 `{N_SINGLE_VIS}`

```python
# ---- 单变量多项式可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：数据散点 + 多项式拟合曲线 + 预测点
axes[0].scatter(X_raw, y, c='red', marker='x', s=80, label='Training Data')

# 生成平滑曲线
x_line = np.linspace({X_LINE_MIN}, {X_LINE_MAX}, 200).reshape(-1, 1)
x_line_poly = make_polynomial(x_line, degree)
x_line_scaled = (x_line_poly - X_poly_mean) / X_poly_std
X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
y_line = X_line @ theta_scaled
axes[0].plot(x_line, y_line, 'b-', linewidth=2, label=f'Ridge Fit (deg={degree})')

axes[0].scatter(new_x_raw[:, 0], [prediction], c='green', s=120, marker='o',
                label=f'Prediction: {prediction:.1f}')
axes[0].set_xlabel('{X_LABEL}')
axes[0].set_ylabel('{Y_LABEL}')
axes[0].set_title('Polynomial Ridge Regression: {TITLE_TOPIC}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：梯度下降收敛曲线（含正则化信息）
reg_label = f'Ridge GD (lam={lam})' if lam > 0 else 'GD (no regularization)'
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cost J(theta)')
axes[1].set_title(reg_label)
axes[1].grid(True, alpha=0.3)
```

同时 `{N_MULTI_VIS}` 替换为 `# 单变量，无需多变量图`。

### 分支 B：多变量基础特征（n_base > 1）→ 用此段替换 `{N_MULTI_VIS}`

```python
# ---- 多变量多项式可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：真实值 vs 预测值
axes[0].scatter(y, y_pred_all, c='blue', s=80, zorder=5)
y_min = min(y.min(), y_pred_all.min()) * 0.9
y_max = max(y.max(), y_pred_all.max()) * 1.1
axes[0].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=1.5, label='Perfect Fit')
axes[0].set_xlabel(f'Actual {y_unit}')
axes[0].set_ylabel(f'Predicted {y_unit}')
axes[0].set_title(f'Polynomial Ridge Regression: {TITLE_TOPIC}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：残差图（残差 vs 预测值）
axes[1].scatter(y_pred_all, residuals, c='green', s=80, zorder=5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
axes[1].set_xlabel(f'Predicted {y_unit}')
axes[1].set_ylabel(f'Residuals {y_unit}')
axes[1].set_title(f'Residual Plot (R^2={r2:.3f}, RMSE={rmse:.2f})')
axes[1].grid(True, alpha=0.3)
```

同时 `{N_SINGLE_VIS}` 替换为 `# 多变量，无需单变量图`。

---

## 占位符说明

| 占位符 | 规则 | 示例（单变量 degree=2） |
|--------|------|------|
| `{X_DATA}` | 基础特征的原始值 | `[500], [800], [1200]` |
| `{Y_DATA}` | y 值列表 | `3.1, 5.0, 7.8, 10.5` |
| `{X_NAMES}` | Python list，特征中文名 | `["面积"]` |
| `{X_UNITS}` | Python list，特征单位 | `["m^2"]` |
| `{DEGREE}` | 多项式最高次数（整数） | `2` |
| `{ALPHA}` | 学习率，默认 0.01 | `0.01` |
| `{ITERS}` | 迭代次数，多项式回归建议 1000 | `1000` |
| `{LAMBDA}` | L2 正则化强度（0 = 无正则化） | `0.1` 或 `0` |
| `{PREDICT_X}` | 新样本基础特征值 | `1100` |
| `{y_name}` / `{y_unit}` | 中文 / 单位 | `能耗` / `万 kWh` |
| `{X_LABEL}` / `{Y_LABEL}` | 英文轴标签（仅 n_base=1） | `Area (m^2)` / `Energy (10k kWh)` |
| `{TITLE_TOPIC}` | 英文预测主题 | `Building Energy Prediction` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名） | `polynomial_regression_20260728_093000` |
| `{X_LINE_MIN/MAX}` | 仅 n_base=1，x 轴范围 | `400` / `2600` |

---

## 关键注意

1. **θ₀ 不参与正则化**：截距项只控制整体平移，不造成曲线"扭曲"。惩罚它没有意义
2. **λ = 0 时等价于标准梯度下降**：代码仍然走 `ridgeGradientDescent`，只是 `gradient[1:] += 0`，行为完全一致
3. **iters 默认 1000**：多项式特征的梯度下降需要比普通线性回归（500 次）更多迭代，因为特征空间维度更高
4. **必须做特征缩放**：x 和 x² 的量级差异极大（如 1000 vs 1000000），不缩放会导致梯度下降无法收敛
5. **参数还原公式**：与普通梯度下降模板一致。截距还原 `θ₀ = θ₀ˢ - Σ(θⱼˢ × μⱼ / σⱼ)`，斜率还原 `θⱼ = θⱼˢ / σⱼ`
6. **可视化左图是曲线**：多项式回归的拟合线不是直线，需要通过 `make_polynomial` + 缩放 + 预测来画平滑曲线
7. **R² 和 RMSE 始终输出**：与正规方程保持一致的评估指标，方便对比
8. **代价函数包含正则化项**：`J_history` 记录了 `MSE + reg_term`，收敛曲线的终值不为 0（除非 λ=0 且完美拟合）
9. **文件命名**：多项式回归统一用 `polynomial_regression_{timestamp}.py`，不区分是否启用正则化
10. **degree 不要太大**：degree=2 覆盖大多数场景；degree=3 可处理 S 型；degree >= 4 极易过拟合，必须配合较大 λ
