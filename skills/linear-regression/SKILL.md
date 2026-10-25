---
name: linear-regression
description: |
  线性回归预测 —— 智能选择正规方程或梯度下降，支持单变量与多变量。
  当用户提到"线性回归"、"线性拟合"、"线性预测"、"直线拟合"、"回归预测"、
  "帮我做回归"、"预测一下"、"拟合一条线"、"多变量回归"等类似场景时触发。
  适用于任何"给一堆 x,y 数据，想拟合并预测新值"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 线性回归预测（正规方程 + 梯度下降）

## 概述

此技能支持**单变量**和**多变量**的**线性回归**和**多项式回归**完整流程：
收集数据 → 检测非线性 → 算法选择 → 训练模型 → 预测新值 → 可视化输出。

核心原则：
- **多项式回归一律使用梯度下降**（可集成 L2 正则化防止过拟合），不管特征数多少
- **普通线性回归**：n < 1000 使用正规方程（一步求解、精确），n >= 1000 使用梯度下降
自动检测数据是否需要多项式回归（曲线拟合），避免"直线拟合曲线数据"效果差的情况。

代码会自动保存到工作目录并运行，每次生成带时间戳的新文件。

## 交互流程

严格按以下 7 步执行，不要跳过任何一步：

### 步骤 0：运行环境预检

在生成或运行 Python 代码前，先确认当前解释器可以导入 `numpy` 和 `matplotlib`。如果任一依赖缺失，暂停训练并提示：

```bash
python -m pip install -r requirements.txt
python check_env.py
```

如果用户没有仓库根目录中的 `requirements.txt`，给出等价命令 `python -m pip install numpy matplotlib`，并要求用户重新运行环境检查。不要继续输出“可运行”代码，直到预检通过。

### 步骤 1：收集数据

首先确认是单变量还是多变量：

> 你有几个特征（x 变量）？每个 x 叫什么名字、单位是什么？

然后让用户提供数据：

> 请提供你的训练数据。可以用表格格式发给我，例如：
>
> 单变量：
> `面积 500, 能耗 3.1；面积 800, 能耗 4.8；...`
>
> 多变量：
> `面积 500, 房间数 5, 能耗 3.0`
> `面积 800, 房间数 8, 能耗 5.5`
> `...`
>
> 同时告诉我：**y 代表什么（含单位）**，以及**预测的主题是什么**（用于图表标题）。

从用户回复中提取：
- `n_features`（特征数量）
- `x_names`（list，如 `["面积", "房间数"]`）
- `x_units`（list，如 `["m^2", "间"]`）
- `x_values`（list of lists，每行 n 个 x 值，或单变量时就是一维 list）
- `y_values`（list）
- `y_name`（如 `"能耗"`）
- `y_unit`（如 `"万 kWh"`）
- `title_topic`（如 `"Building Energy Prediction"`，英文）

如果用户没有给出某个信息，根据上下文合理推断并确认。

### 步骤 2：检测非线性关系（多项式回归判断）

拿到数据后，判断 x 和 y 之间是否可能是**非线性关系**。根据以下信号综合判断：

| 信号类型 | 具体表现 | 结论 |
|----------|----------|------|
| 用户描述 | "先快后慢"、"增长放缓"、"边际递减"、"U型"、"S型"、"先降后升" | 疑似非线性 |
| 数据趋势 | 如果画出散点图，肉眼可见弯曲 | 疑似非线性 |
| 领域知识 | 建筑能耗随面积"边际递减"是普遍规律 | 可主动建议 |

**如果疑似非线性，主动询问用户确认：**

> 从你的数据和描述来看，x 和 y 的关系可能是**非线性的**（{具体描述，如"面积越大，能耗增量逐渐变小"}）。
> 建议使用**多项式回归**来处理。请确认以下参数：
>
> - **最高次数（degree）**：默认 **2**（在模型中加入 x² 项）
>   - degree=2：能拟合 ∪ 型或 ⋂ 型曲线（最常见）
>   - degree=3：能拟合 S 型曲线（更灵活，但需要更多数据）
>   - 建议先用 degree=2 试试
> - 是否需要调整？（直接回复"可以"使用 degree=2）

如果用户确认使用多项式回归：

- 设置标志 `use_polynomial = True`
- 记录 `degree`（用户指定或默认 2）
- 后续步骤将走梯度下降 + 正则化路径（详见步骤 4）
- 文件名前缀改为 `polynomial_regression_`（区别于普通线性回归）

如果用户明确表示是线性关系（或数据无明显非线性特征），设置 `use_polynomial = False`，走普通线性回归流程。

### 步骤 3：确认预测目标

> 你想预测哪个样本的 {y_name}？请给出对应的 x 值：
> {x1_name} = ?, {x2_name} = ?, ...

记录为 `predict_x`（list，单变量时长度为 1）。

### 步骤 4：算法选择

首先判断是否为多项式回归：

```
判断回归类型
    │
    ├── 多项式回归（use_polynomial = True）
    │     │
    │     └── 一律使用梯度下降（不管构造后特征数 n_poly 多少）
    │           原因：正规方程 + 多项式 = 极高过拟合风险；
    │                 梯度下降可以方便集成 L2 正则化。
    │           │
    │           ├── 进入步骤 4b（正则化判断 + λ 确认）
    │           └── 然后进入步骤 4c（学习率确认）
    │
    └── 普通线性回归（use_polynomial = False）
          │
          ├── n < 1000 ──→ 正规方程 → 读取 references/normal-equation.md
          │                 不需要特征缩放、不需要学习率、一步求解
          │                 跳过步骤 4b、4c，直接到步骤 5
          │
          └── n >= 1000 ──→ 梯度下降 → 使用本文件「梯度下降代码模板」
                            跳过步骤 4b（正则化判断），进入步骤 4c
```

---

#### 步骤 4b：正则化判断（仅多项式回归分支）

计算 `n_poly = n_base × degree`（构造后的特征总数），`ratio = m / n_poly`。

根据以下条件综合判断是否建议正则化，以及建议的 λ 值：

| 条件 | 建议 |
|------|------|
| m / n_poly < 3 | 强烈建议正则化，建议 λ = 5.0 |
| 3 <= m / n_poly < 5 | 建议正则化，建议 λ = 1.0 |
| 5 <= m / n_poly < 10，或 degree=3 且 ratio < 10 | 轻度建议正则化，建议 λ = 0.5 |
| m / n_poly >= 10 且 degree <= 2 | 可不正则化（λ = 0），数据充裕 |

询问用户：

> 当前数据：{m} 个样本，构造后 {n_poly} 个特征（m/n_poly = {ratio:.1f}），degree={degree}。
>
> {根据上表生成具体建议} 建议使用**岭回归（梯度下降 + L2 正则化，λ={suggested_lambda}）**来控制模型复杂度，防止过拟合。
>
> 是否启用正则化？（直接回复"可以"使用建议，或回复"不需要"关闭正则化）

如果用户同意正则化，进一步确认 λ：

> 建议正则化强度 λ = **{suggested_lambda}**。
> - λ 越大 → 惩罚越强 → 模型越简单、越不容易过拟合
> - λ = 0 → 无正则化 = 标准梯度下降
> - 常用范围：0.1（轻度）~ 10（强力）
>
> 是否需要调整 λ？（直接回复"可以"使用建议值，或输入新值如 0.5、2.0）

确认后记录：
- `use_regularization = True`，`lambda_val = 用户确认的 λ`
- 如果用户拒绝正则化：`use_regularization = False`，`lambda_val = 0`

**无论是否正则化，都使用 `references/ridge-regression.md` 模板**（λ=0 时等价于标准梯度下降多项式回归）。

---

#### 步骤 4c：学习率确认（仅梯度下降分支）

> 特征已做 z-score 标准化，建议学习率为 **0.01**。
> 是否需要调整？（直接回复"可以"使用建议值）

- 普通线性回归：`iters` 默认 500
- 多项式回归：`iters` 默认 1000（特征空间更复杂，需要更多迭代）

### 步骤 5：生成代码

在当前工作目录生成代码文件。文件名规则：

- **普通线性回归**：`linear_regression_{YYYYMMDD}_{HHMMSS}.py`
- **多项式回归**：`polynomial_regression_{YYYYMMDD}_{HHMMSS}.py`

模板选择：

- **正规方程（普通线性，n < 1000）**：参照 `references/normal-equation.md` 中的代码模板
- **梯度下降（普通线性，n >= 1000）**：使用下方「梯度下降代码模板」
- **多项式回归（任何 n，含/不含正则化）**：参照 `references/ridge-regression.md` 中的代码模板
  - `{LAMBDA}` 占位符：有正则化填建议值，无正则化填 0

将所有 `{PLACEHOLDER}` 替换为实际值。

### 步骤 6：运行代码

```bash
cd "{工作目录}" && python {FILENAME_BASE}.py
```

> 普通线性: `linear_regression_{timestamp}.py`；多项式: `polynomial_regression_{timestamp}.py`

Error → 分析修复后重试。

### 步骤 7：输出结果

向用户报告：
- 模型公式
  - 普通线性：`y = θ₀ + θ₁x₁ + θ₂x₂ + ...`
  - 多项式：`y = θ₀ + θ₁·x + θ₂·x² + ...`（展示曲线形状：θ₂ < 0 → ⋂ 型增长放缓；θ₂ > 0 → ∪ 型）
- 预测结果
- **正规方程额外输出**：R² 和 RMSE
- **岭回归额外输出**：R²、RMSE、最终代价 J(θ)、使用的 λ 值
- 图片保存路径和代码文件路径

---

## 正规方程代码模板（主模板，n < 1000）

> 完整模板见 `references/normal-equation.md`。

关键特点：
- 不需特征缩放、不需学习率、不需迭代
- 左图：真实值 vs 预测值诊断图
- 右图：残差图（用于检查拟合质量）
- 输出 R² 和 RMSE 指标

核心代码：

```python
# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_raw])

# 正规方程 —— 一步求解（O(n³)，n<1000 完全无感）
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

---

## 梯度下降代码模板（备用，n >= 1000）

> 多变量兼容。`{PLACEHOLDER}` 需要替换。中文字符一律用 ASCII（`theta0` 不用 `θ₀`）。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
# X_raw: (m, n) 矩阵，每行一个样本，每列一个特征
# y: (m, 1) 向量
X_raw = np.array([{X_DATA}])   # 每行 [{x1_i}, {x2_i}, ...]
y = np.array([{Y_DATA}]).reshape(-1, 1)

m, n = X_raw.shape  # 样本数, 特征数

# ========== 2. 特征缩放（逐列 z-score）==========
X_mean = X_raw.mean(axis=0, keepdims=True)   # (1, n)
X_std  = X_raw.std(axis=0, keepdims=True)    # (1, n)
X_scaled = (X_raw - X_mean) / X_std           # (m, n)

# 构造 X（加偏置列）
X = np.hstack([np.ones((m, 1)), X_scaled])   # (m, n+1)

# ========== 3. 梯度下降 ==========
def gradientDescent(X, y, theta, alpha, iters):
    """批量梯度下降（多变量兼容）"""
    m = len(y)
    J_history = np.zeros(iters)
    for i in range(iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta = theta - alpha * gradient
        J_history[i] = (1 / (2 * m)) * np.sum(errors ** 2)
    return theta, J_history

theta_scaled = np.zeros((n + 1, 1))
alpha = {ALPHA}
iters = {ITERS}

theta_scaled, J_history = gradientDescent(X, y, theta_scaled, alpha, iters)

# 恢复原始尺度的参数
theta = np.zeros((n + 1, 1))
theta[0, 0] = theta_scaled[0, 0] - np.sum(theta_scaled[1:, 0] * X_mean[0] / X_std[0])
for j in range(1, n + 1):
    theta[j, 0] = theta_scaled[j, 0] / X_std[0, j - 1]

# 打印模型公式
formula = f"{y_name} = {theta[0,0]:.4f}"
for j in range(n):
    sign = '+' if theta[j+1, 0] >= 0 else ''
    formula += f" {sign}{theta[j+1, 0]:.6f} * {x_names[j]}"
print(f"训练结果：{formula}")

# ========== 4. 预测 ==========
new_x = np.array([{PREDICT_X}])   # 新样本的特征值
new_x_scaled = (new_x - X_mean) / X_std
X_new = np.hstack([np.ones((1,1)), new_x_scaled])
prediction = (X_new @ theta_scaled)[0, 0]
print(f"预测{y_name}：{prediction:.2f} {y_unit}")

# ========== 5. 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---- 左图：真实值 vs 预测值 ----
y_pred_all = X @ theta_scaled
axes[0].scatter(y, y_pred_all, c='blue', s=80, zorder=5)
y_min = min(y.min(), y_pred_all.min()) * 0.9
y_max = max(y.max(), y_pred_all.max()) * 1.1
axes[0].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=1.5, label='Perfect Fit')
axes[0].set_xlabel('Actual {Y_LABEL}')
axes[0].set_ylabel('Predicted {Y_LABEL}')
axes[0].set_title('Linear Regression: {TITLE_TOPIC}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ---- 右图：梯度下降收敛曲线 ----
axes[1].plot(range(iters), J_history, 'b-', linewidth=1.5)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cost J(theta)')
axes[1].set_title('Gradient Descent Convergence')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print("图片已保存: {FILENAME_BASE}.png")
```

---

## 占位符填写规则

| 占位符 | 规则 | 示例 |
|--------|------|------|
| `{X_DATA}` | 每行一组 x 值，逗号分隔行 | `[500, 5], [800, 8], [1000, 10]` |
| `{Y_DATA}` | y 值数组 | `3.0, 5.5, 6.0, 8.0` |
| `{x_names}` | Python list 形式 | `["Area", "Rooms"]` |
| `{y_name}` / `{y_unit}` | 中文变量名 / 单位 | `能耗` / `万 kWh` |
| `{Y_LABEL}` | 英文 y 轴标签 | `Energy (10k kWh)` |
| `{TITLE_TOPIC}` | 英文预测主题 | `Building Energy Prediction` |
| `{ALPHA}` | 仅梯度下降，默认 0.01 | `0.01` |
| `{ITERS}` | 仅梯度下降，默认 500 | `500` |
| `{PREDICT_X}` | 新样本所有特征值 | `1100, 7`（多变量逗号分隔） |
| `{FILENAME_BASE}` | 时间戳文件名 | `linear_regression_20260727_113000` |

## 关键注意事项

### 编码问题
Windows GBK 环境下，print 中的 Unicode 必须替换：`θ₀→theta0`, `θ₁→theta1`, `²→^2`, `×→*`。
matplotlib 的 label/title 不走 GBK，可以用 Unicode。

### 无头环境
必须 `matplotlib.use('Agg')`，用 `plt.close()` 代替 `plt.show()`。

### 正规方程 vs 梯度下降

| | 正规方程（默认） | 梯度下降（备用） | 岭回归（多项式专用） |
|---|---|---|---|
| 条件 | 普通线性，n < 1000 | 普通线性，n >= 1000 | 多项式回归（任何 n） |
| 特征缩放 | 不需要 | 必须 z-score | 必须 z-score |
| 学习率 | 不需要 | 需要调 | 需要调 |
| 迭代 | 1 步精确解 | ~500 步近似 | ~1000 步近似 |
| 正则化 | 不支持 | 不支持 | 支持 L2（λ 可调） |
| 右图 | 残差图 | 收敛曲线 | 收敛曲线（含 λ 标记） |
| 额外指标 | R², RMSE | J_history | R², RMSE, J_history, λ |

### 参数还原（仅梯度下降）

梯度下降在缩放后的特征上训练，必须还原到原始尺度：
- `θⱼᵒʳⁱᵍ = θⱼˢᶜᵃˡᵉᵈ / σⱼ`（斜率）
- `θ₀ᵒʳⁱᵍ = θ₀ˢᶜᵃˡᵉᵈ - Σ(θⱼˢᶜᵃˡᵉᵈ × μⱼ / σⱼ)`（截距）

### 数据量检查
- 最少 `n+2` 个数据点（否则欠定）
- 样本数 < 5×n 时提醒用户结果可能不够稳健

### 文件名
时间戳 `YYYYMMDD_HHMMSS`，不覆盖历史文件。

### 参考文件
- `references/normal-equation.md`：正规方程完整代码模板，**普通线性回归 n < 1000 时使用**
- `references/ridge-regression.md`：多项式回归 + 梯度下降 + L2 正则化代码模板，**多项式回归时使用（主路径）**。λ=0 时等价于纯梯度下降多项式回归
- `references/polynomial-regression.md`：[历史参考] 多项式 + 正规方程（过拟合风险高，主流程不再使用。保留作为备选）
