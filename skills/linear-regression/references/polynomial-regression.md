# 多项式回归（Polynomial Regression）—— 直线不够用的时候

> ⚠️ **本文件为历史参考。多项式回归的新主路径是 `references/ridge-regression.md`（梯度下降 + L2 正则化），可有效防止过拟合。**

## 概述

多项式回归 = **特征工程 + 线性回归**。核心思想：在数据预处理阶段手动构造 `x², x³, ...` 作为新特征，然后用线性回归求解。**代价函数和梯度下降的原代码一字不改。**

> 本文件使用正规方程求解。新主路径使用梯度下降 + L2 正则化，见 `ridge-regression.md`。

$$\text{原模型：} h_\theta(x) = \theta_0 + \theta_1 \cdot x$$
$$\text{多项式模型 (degree=2)：} h_\theta(x) = \theta_0 + \theta_1 \cdot x + \theta_2 \cdot x^2$$

> 对参数 θ 仍是线性的！只是 x 被扩展成了多个"人工特征"。

## 何时使用（检测规则）

在技能步骤 2 中，根据以下信号判断是否需要多项式回归：

| 信号 | 说明 | 示例 |
|------|------|------|
| 用户明确描述非线性关系 | "先快后慢"、"增长放缓"、"边际递减" | "面积越大，能耗增长越来越慢" |
| 数据形状描述 | "U型"、"倒U型"、"S型"、"曲线" | "风机能耗随转速先降后升" |
| 散点图明显弯曲 | 画出数据后肉眼可见不是直线 | 数据点呈弧线分布 |
| 简单线性回归 R² 低 | 做完普通线性回归后 R² < 0.7 | 直线只解释了不到 70% 的变异 |

确认后询问用户：

> 从你的数据和描述来看，x 和 y 的关系可能是**非线性的**（{具体形状描述}）。
> 建议使用**多项式回归**处理。请确认：
> - **最高次数（degree）**：默认 **2**（加 x²），也可以选 3（加 x² 和 x³）
> - 次数越高曲线越灵活，但也越容易过拟合。一般 degree ≤ 3

## 多项式构造方式

### 单变量基础特征（最常见）

```python
# 原始：只有 面积 这一个 x
X_raw = [[500], [800], [1200]]     # (m, 1)

# 构造后（degree=2）：多了 面积²
X_poly = [[500, 250000],
          [800, 640000],
          [1200, 1440000]]          # (m, 2)

# 构造后（degree=3）：多了 面积² 和 面积³
X_poly = [[500, 250000, 125000000],
          [800, 640000, 512000000],
          [1200, 1440000, 1728000000]]  # (m, 3)
```

### 多变量基础特征

每个基础特征各自扩展多项式（不含交互项，保持可解释性）：

```python
# 原始：面积 + 房间数 两个 x
X_raw = [[500, 5], [800, 8], [1200, 10]]   # (m, 2)

# 构造后（degree=2）：
# [面积, 面积², 房间数, 房间数²]
X_poly = [[500, 250000, 5, 25],
          [800, 640000, 8, 64],
          [1200, 1440000, 10, 100]]          # (m, 4)
```

---

## 算法选择

> ⚠️ **重要变更（2026-07-28）**：多项式回归的主路径已改为 **梯度下降 + L2 正则化（岭回归）**，详见 `references/ridge-regression.md`。
>
> 正规方程 + 多项式 = 极高过拟合风险（高次多项式能完美穿过所有训练点，训练误差=0 但测试误差爆炸）。
> 梯度下降 + L2 正则化能够有效控制参数大小、保持曲线平滑。
>
> 本文件中的正规方程模板**保留作为历史参考**，仅在用户明确要求"多项式 + 正规方程 + 不要正则化"时使用。

| 条件 | 决策 |
|------|------|
| 多项式回归 | 🔴 主路径 → `references/ridge-regression.md`（梯度下降 + L2 正则化） |
| 多项式 + 明确不需要正则化 | 🟡 备选 → 本文件正规方程模板（有风险，需提醒用户） |

> 绝大多数多项式回归场景 n_poly 很小（degree=2, n_base=1~5），但即使如此也建议走梯度下降+正则化路径。

---

## 代码模板（正规方程 + 多项式特征）

> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 基础特征矩阵 (m, n_base)
y = np.array([{Y_DATA}]).reshape(-1, 1)

m, n_base = X_raw.shape
x_names = {X_NAMES}   # 基础特征名列表
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
    m, n_base = X_raw.shape
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

# ========== 3. 正规方程 —— 一步求解 ==========
# 不需要特征缩放！正规方程直接解线性方程组。
X = np.hstack([np.ones((m, 1)), X_poly])   # (m, n+1)
theta = np.linalg.inv(X.T @ X) @ X.T @ y   # (n+1, 1)

# 打印模型公式
formula = f"{y_name} = {theta[0,0]:.6f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign} {theta[j+1, 0]:.6f} * {poly_names[j]}"
print(f"模型公式: {formula}")

# ========== 4. 预测 ==========
new_x_raw = np.array([{PREDICT_X}])   # 新样本基础特征值，形状 (1, n_base)
new_x_poly = make_polynomial(new_x_raw, degree)
X_new = np.hstack([np.ones((1, 1)), new_x_poly])
prediction = (X_new @ theta)[0, 0]
print(f"预测{y_name}: {prediction:.2f} {y_unit}")

# ========== 5. 拟合优度指标 ==========
y_pred_all = X @ theta
residuals = y - y_pred_all
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean(residuals ** 2))

print(f"R^2 = {r2:.4f} (越接近1拟合越好)")
print(f"RMSE = {rmse:.4f} {y_unit}")

# ========== 6. 可视化 ==========
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

# 左图：数据散点 + 多项式曲线 + 预测点
axes[0].scatter(X_raw, y, c='red', marker='x', s=80, label='Training Data')

# 生成平滑曲线（非直线！）
x_line = np.linspace({X_LINE_MIN}, {X_LINE_MAX}, 200).reshape(-1, 1)
x_line_poly = make_polynomial(x_line, degree)
X_line = np.hstack([np.ones((200, 1)), x_line_poly])
y_line = X_line @ theta
axes[0].plot(x_line, y_line, 'b-', linewidth=2, label=f'Polynomial Fit (degree={degree})')

axes[0].scatter(new_x_raw[:, 0], [prediction], c='green', s=120, marker='o',
                label=f'Prediction: {prediction:.1f}')
axes[0].set_xlabel('{X_LABEL}')
axes[0].set_ylabel('{Y_LABEL}')
axes[0].set_title('Polynomial Regression: {TITLE_TOPIC}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：真实值 vs 预测值
axes[1].scatter(y, y_pred_all, c='blue', s=80, zorder=5)
y_min = min(y.min(), y_pred_all.min()) * 0.9
y_max = max(y.max(), y_pred_all.max()) * 1.1
axes[1].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=1.5, label='Perfect Fit')
axes[1].set_xlabel(f'Actual {y_unit}')
axes[1].set_ylabel(f'Predicted {y_unit}')
axes[1].set_title(f'Actual vs Predicted (R^2={r2:.3f}, RMSE={rmse:.2f})')
axes[1].legend()
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
axes[0].set_title(f'Polynomial Regression: {TITLE_TOPIC}')
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
| `{X_NAMES}` | Python list，基础特征中文名 | `["面积"]` |
| `{X_UNITS}` | Python list，基础特征单位 | `["m^2"]` |
| `{DEGREE}` | 多项式最高次数（整数） | `2` |
| `{PREDICT_X}` | 新样本基础特征值 | `1100` |
| `{y_name}` / `{y_unit}` | 中文 / 单位 | `能耗` / `万 kWh` |
| `{X_LABEL}` / `{Y_LABEL}` | 英文轴标签（仅 n_base=1） | `Area (m^2)` / `Energy (10k kWh)` |
| `{TITLE_TOPIC}` | 英文预测主题 | `Building Energy Prediction` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名） | `polynomial_regression_20260727_140000` |
| `{X_LINE_MIN/MAX}` | 仅 n_base=1，x 轴范围 | `400` / `2600` |

> 文件名前缀改为 `polynomial_regression_` 以区别于普通线性回归。

---

## 关键注意

1. **不需要特征缩放**：正规方程直接解线性方程组，即便 x 和 x² 差了几个数量级也不影响结果
2. **不需要 alpha、iters**：无超参数，和普通正规方程完全一样
3. **degree 不要太大**：degree=2 覆盖大多数场景（∪ 型 / ⋂ 型）；degree=3 可处理 S 型。degree 再高容易过拟合
4. **数据量要求**：最少 `n_poly + 2` 个数据点。degree=2 时至少 4 个点
5. **可视化**：单变量基础特征时，左图是**曲线**（不是直线），因为 x 出现在了多项式中
6. **θ 符号含义**：θ₂ < 0 → ⋂ 型（增长放缓），θ₂ > 0 → ∪ 型（先降后升），θ₂ ≈ 0 → 近似直线，多项式不必要
7. **特征构造不含交互项**：为了可解释性，`make_polynomial` 只做 `x, x², x³`，不做 `x₁×x₂` 这种交叉项
