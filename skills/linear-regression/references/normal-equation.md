# 正规方程（Normal Equation）—— 线性回归精确解

## 概述

当特征数 n < 1000 时，使用正规方程。**不需要特征缩放、不需要学习率、不需要迭代 —— 一步求解精确 θ。**

$$\theta = (X^T X)^{-1} X^T y$$

- 单变量 (n=1)：左图为数据散点+拟合线+预测点，右图为真实值 vs 预测值诊断图
- 多变量 (n>1)：左图为真实值 vs 预测值，右图为残差图

## 算法选择

| 条件 | 决策 |
|------|------|
| n < 1000 | ✅ 正规方程（本文件） |
| n >= 1000 | ❌ 退回梯度下降 |

---

## 代码模板

> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代（`theta0` 不用 `θ₀`），Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
# X_raw: (m, n) 矩阵，每行一个样本，每列一个特征
# y: (m, 1) 向量
X_raw = np.array([{X_DATA}])
y = np.array([{Y_DATA}]).reshape(-1, 1)

m, n = X_raw.shape   # 样本数, 特征数
x_names = {X_NAMES}  # 特征名列表（中文，用于打印）
x_units = {X_UNITS}  # 特征单位列表

# ========== 2. 构造 X（加偏置列，无需缩放）==========
X = np.hstack([np.ones((m, 1)), X_raw])   # (m, n+1)

# ========== 3. 正规方程 —— 一步求解 ==========
# θ = (XᵀX)⁻¹ Xᵀ y      复杂度 O(n³)，n<1000 无感
theta = np.linalg.inv(X.T @ X) @ X.T @ y   # (n+1, 1)

# 打印模型公式
formula = f"{y_name} = {theta[0,0]:.6f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign} {theta[j+1, 0]:.6f} * {x_names[j]}"
print(f"模型公式: {formula}")

# ========== 4. 预测 ==========
new_x = np.array([{PREDICT_X}])   # 新样本特征值，形状 (1, n)
X_new = np.hstack([np.ones((1, 1)), new_x])
prediction = (X_new @ theta)[0, 0]
print(f"预测{y_name}: {prediction:.2f} {y_unit}")

# ========== 5. 拟合优度指标 ==========
y_pred_all = X @ theta                       # 所有样本的预测值
residuals = y - y_pred_all                    # 残差
ss_res = np.sum(residuals ** 2)               # 残差平方和
ss_tot = np.sum((y - y.mean()) ** 2)          # 总离差平方和
r2 = 1 - ss_res / ss_tot                      # R² 决定系数
rmse = np.sqrt(np.mean(residuals ** 2))       # 均方根误差

print(f"R^2 = {r2:.4f} (越接近1拟合越好)")
print(f"RMSE = {rmse:.4f} {y_unit}")

# ========== 6. 可视化 ==========
{N_SINGLE_VISUALIZATION}
{N_MULTI_VISUALIZATION}

plt.tight_layout()
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print("图片已保存: {FILENAME_BASE}.png")
```

---

## 可视化分支

### 分支 A：单变量（n == 1）→ 用此段替换 `{N_SINGLE_VISUALIZATION}`

```python
# ---- 单变量可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：数据散点 + 拟合直线 + 预测点
axes[0].scatter(X_raw, y, c='red', marker='x', s=80, label='Training Data')
x_line = np.linspace({X_LINE_MIN}, {X_LINE_MAX}, 100).reshape(-1, 1)
X_line = np.hstack([np.ones((100, 1)), x_line])
y_line = X_line @ theta
axes[0].plot(x_line, y_line, 'b-', linewidth=2, label='Fitted Line')
axes[0].scatter(new_x[:, 0], [prediction], c='green', s=120, marker='o',
                label=f'Prediction: {prediction:.1f}')
axes[0].set_xlabel('{X_LABEL}')
axes[0].set_ylabel('{Y_LABEL}')
axes[0].set_title('Linear Regression: {TITLE_TOPIC}')
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

同时 `{N_MULTI_VISUALIZATION}` 替换为 `# 单变量，无需多变量图`。

### 分支 B：多变量（n > 1）→ 用此段替换 `{N_MULTI_VISUALIZATION}`

```python
# ---- 多变量可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：真实值 vs 预测值
axes[0].scatter(y, y_pred_all, c='blue', s=80, zorder=5)
y_min = min(y.min(), y_pred_all.min()) * 0.9
y_max = max(y.max(), y_pred_all.max()) * 1.1
axes[0].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=1.5, label='Perfect Fit')
axes[0].set_xlabel(f'Actual {y_unit}')
axes[0].set_ylabel(f'Predicted {y_unit}')
axes[0].set_title(f'Linear Regression: {TITLE_TOPIC}')
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

同时 `{N_SINGLE_VISUALIZATION}` 替换为 `# 多变量，无需单变量图`。

---

## 占位符说明

| 占位符 | 规则 | 示例 |
|--------|------|------|
| `{X_DATA}` | 每行 n 个 x 值，行间逗号分隔 | 单变量: `[5], [8], [10]`；多变量: `[500, 5], [800, 8]` |
| `{Y_DATA}` | y 值列表 | `3.0, 5.5, 6.0, 8.0` |
| `{X_NAMES}` | Python list，特征中文名 | `["面积", "房间数"]` |
| `{X_UNITS}` | Python list，特征单位 | `["m^2", "间"]` |
| `{PREDICT_X}` | 新样本特征值 | 单变量: `17`；多变量: `1100, 7` |
| `{y_name}` / `{y_unit}` | 中文 / 单位 | `能耗` / `万 kWh` |
| `{X_LABEL}` / `{Y_LABEL}` | 仅单变量左图，英文轴标签 | `Area (m^2)` / `Energy (10k kWh)` |
| `{TITLE_TOPIC}` | 英文预测主题 | `Building Energy Prediction` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名） | `linear_regression_20260727_113000` |
| `{X_LINE_MIN/MAX}` | 仅单变量，x 轴范围 | `3` / `26` |

## 关键注意

1. **不需要特征缩放**：正规方程直接解线性方程组，特征量级不影响结果
2. **不需要 alpha、iters**：无超参数
3. **单变量时左图保留传统散点+拟合线**：更直观，与用户之前的认知一致
4. **多变量无法画 2D 拟合线**：改用「真实值 vs 预测值」诊断图 + 残差图
5. **R² 和 RMSE**：每次输出，帮助判断模型好坏
6. **O(n³) 限制**：n >= 1000 时矩阵求逆变慢，应退回梯度下降
