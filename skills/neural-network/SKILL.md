---
name: neural-network
description: |
  神经网络预测 —— 支持二分类与多分类，纯 NumPy 从零实现反向传播。
  当用户提到"神经网络"、"neural network"、"NN预测"、"深度学习"、
  "帮我进行神经网络预测"、"用神经网络分类"、"用NN"等类似场景时触发。
  适用于任何"给一堆 x,y 数据，想用神经网络预测新样本类别"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 神经网络预测（Neural Network）

## 概述

此技能支持**二分类**和**多分类**的神经网络完整流程：
收集数据 → 数据切分（可选）→ 判断分类类型 → 推荐网络结构 → 确认预测目标 → 建议参数 → 梯度检验（可选）→ 训练模型 → 预测新值 → 可视化输出。

核心原则：
- **纯 NumPy 从零实现**反向传播（不依赖 TensorFlow/PyTorch）
- **内置特征缩放**（z-score 标准化，梯度下降必需）
- **默认启用 L2 正则化**（防止过拟合，λ 可调为 0 关闭）
- **Xavier 随机初始化**（打破对称性，避免神经元同质化）
- **首次训练建议运行梯度检验**（验证反向传播正确性，通过后关闭）
- 自动判断二分类还是多分类

代码会自动保存到工作目录并运行，每次生成带时间戳的新文件。

## 交互流程

### 环境预检（生成代码前必须执行）

先确认当前 Python 解释器可以导入 
缺少依赖时暂停执行并提示：python -m pip install -r requirements.txt，然后运行 python check_env.py；预检未通过时不要声称代码可运行。


严格按以下 10 步执行，不要跳过任何一步：

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

### 步骤 1.5：确认是否预留验证数据

在训练前，询问用户是否需要切分数据，为后续模型诊断做准备：

> ### 📊 数据切分设置
>
> 你提供了 **{m}** 条数据。在训练前，建议预留一部分数据作为**验证集**，
> 用于客观评估模型真实效果（避免"考试题就是练习题"）。
>
> 推荐方案：**60% 训练 / 20% 验证 / 20% 测试**
> - 训练集（**{train_size}** 条）：用来训练模型参数
> - 验证集（**{val_size}** 条）：用来调参和诊断过拟合/欠拟合
> - 测试集（**{test_size}** 条）：最终评估（本次训练暂不使用）
>
> 是否预留验证数据？
> - **A. 是（推荐）**——切分数据，可得到真实泛化评估，后续还能用 `neural-network-diagnosis` 技能诊断模型
> - **B. 否**——全部数据用于训练（简单直接，但无法客观评估泛化效果）

**边界情况：**

| 情况 | 处理 |
|------|------|
| m < 15 | 告知用户数据太少不建议切分，默认走"全部数据训练"。如用户坚持切分，用 70/30（训练/验证，无测试集） |
| 某类别样本数 < 3 | 告知用户该类样本不足以支撑切分，建议不切分或收集更多数据 |

**用户选 A（切分）时：**

1. 执行 60/20/20 随机切分（`np.random.seed(42)` 保证可复现）
2. 打印切分结果摘要：
   ```
   ✅ 数据已切分（随机种子=42）：
      - 训练集: {train_size} 条（用于训练模型参数）
      - 验证集: {val_size} 条（用于调参和诊断）
      - 测试集: {test_size} 条（预留，本次训练不使用）
   
   各类别分布:
      {class_0}: 训练{t0_train}条, 验证{t0_val}条, 测试{t0_test}条
      {class_1}: 训练{t1_train}条, 验证{t1_val}条, 测试{t1_test}条
   ```
3. 检查每类在验证集中至少 1 条 → 如某类为 0，重新 shuffle 并告知用户
4. 设置 `do_split = True`，记录 `train_idx`, `val_idx`, `test_idx`

**用户选 B（不切分）时：**

```
⚠️ 使用全部 {m} 条数据训练。注意：后续如需诊断模型泛化效果，
   需要用 neural-network-diagnosis 技能重新划分数据并重新训练。
```

设置 `do_split = False`。

### 步骤 3：判断分类类型

检查 y 的唯一值数量 `K = len(set(y_raw))`：

| K | 分类类型 | 输出层激活 | 输出神经元数 | 后续路径 |
|---|---------|----------|------------|---------|
| = 2 | **二分类** | Sigmoid | 1 | 继续步骤 4 |
| > 2 | **多分类** | Softmax | K | 继续步骤 4（网络结构推荐逻辑相同），代码走 `references/multi-class-nn.md` |

告知用户：

> 检测到 y 有 {K} 个类别，这是**{二分类/多分类}**问题。
> 输出层将使用 **{Sigmoid / Softmax}** 激活函数，{1 / K} 个输出神经元。

如果是多分类，标记 `use_multiclass = True`。

### 步骤 4：推荐网络结构

计算 `m`（样本数）, `n`（特征数）, `K`（类别数）。

**推荐策略：**

```
隐藏层数：
  ├── m < 50 → 1 个隐藏层（数据少，层数多极易过拟合）
  ├── 50 ≤ m < 200 且 n > 5 → 可考虑 2 个隐藏层
  └── m ≥ 200 → 可考虑 2 个隐藏层（数据充裕）

隐藏层神经元数：
  ├── 第 1 层 = max(3, min(n * 3, 20))（取 n×3，最少 3，最多 20）
  └── 如有第 2 层 = max(2, 第1层 // 2)（逐层减半）
```

总参数量计算：
```
# 1 隐藏层
total_params = (n+1)*h1 + (h1+1)*K

# 2 隐藏层
total_params = (n+1)*h1 + (h1+1)*h2 + (h2+1)*K
```

询问话术：

> 根据你的数据（{m} 个样本，{n} 个特征），建议网络结构：
> - 输入层：**{n}** 个神经元（{x_names 列表}）
> - 隐藏层：**{num_layers}** 层，每层 {hidden_neurons} 个神经元（Sigmoid 激活）
> - 输出层：**{K}** 个神经元（{Sigmoid/Softmax}）
>
> 总参数量约 **{total_params}** 个。
>
> 是否需要调整网络结构？（直接回复"可以"使用推荐结构，或指定你想要的层数和神经元数，如"2层，8和4"）

记录：`hidden_layer_sizes`（list，如 `[6]` 表示 1 隐藏层 6 神经元，`[8, 4]` 表示 2 隐藏层 8 和 4 神经元）

### 步骤 5：确认预测目标

> 你想预测哪个新样本的类别？请给出对应的 x 值：
> {x1_name} = ?, {x2_name} = ?, ...

记录为 `predict_x`（list，长度等于 n_features）。

### 步骤 6：建议学习率 α 和正则化 λ

特征已做 z-score 标准化。根据当前场景选择建议参数：

**α 建议规则：**

| 场景 | 建议 α | 原因 |
|------|--------|------|
| 二分类，1 隐藏层 | 0.1 | 特征缩放后梯度稳定 |
| 二分类，2 隐藏层 | 0.05 | 更深网络，梯度传播路径更长 |
| 多分类，1 隐藏层 | 0.05 | Softmax + 多个输出神经元 |
| 多分类，2 隐藏层 | 0.03 | 最复杂场景，保守步长 |

**λ 建议规则（基于数据/参数比）：**

计算 `ratio = m / total_params`：

| ratio | 建议 λ | 说明 |
|-------|--------|------|
| < 1 | 1.0 | 参数比样本还多，强力正则化 |
| 1 ~ 3 | 0.1 | 参数较多，中度正则化 |
| 3 ~ 10 | 0.01 | 数据充裕，轻度正则化（默认情况） |
| > 10 | 0.001 | 数据很充足，微弱正则化 |

**iters 建议：**

| 场景 | 建议 iters |
|------|-----------|
| 二分类，1 隐藏层 | 3000 |
| 二分类，2 隐藏层 | 5000 |
| 多分类，1 隐藏层 | 5000 |
| 多分类，2 隐藏层 | 8000 |

询问话术：

> 当前场景：{场景描述}（{m} 样本，{total_params} 参数，ratio={ratio:.1f}）。
> 特征已做 z-score 标准化。建议训练参数：
> - 学习率 α = **{suggested_alpha}**
> - 正则化 λ = **{suggested_lambda}**（L2 正则化，λ=0 则关闭）
> - 迭代次数 = **{suggested_iters}**
>
> 是否需要调整？（直接回复"可以"使用建议值，或指定要修改的参数）

如果用户要求调整，使用用户指定的值。确认后记录 `alpha`, `lambda_val`, `iters`。

### 步骤 7：确认是否运行梯度检验

计算总参数量 `total_params`，估算梯度检验耗时（每个参数约需 2 次代价函数计算，每次约 0.5-2ms）：

> 网络共有 **{total_params}** 个参数，梯度检验需要计算 **{total_params × 2}** 次代价函数。
> 预估耗时约 **{estimated_time} 秒**。
>
> **建议首次使用此网络结构时运行一次梯度检验**，验证反向传播代码的梯度计算是否正确。
>
> 原理：数值梯度 ≈ [J(θ+ε) − J(θ−ε)] / (2ε)，与反向传播的解析梯度比对。
> - 差异 < 1e-7 → ✅ 反向传播梯度计算无误
> - 差异在 1e-4 ~ 1e-7 → ⚠️ 可能有浮点误差，通常可接受
> - 差异 > 1e-4 → ❌ 反向传播存在 bug，需要检查
>
> **注意**：同一网络结构只需检验一次。之前已验证过的结构可以跳过。
>
> 是否运行梯度检验？（直接回复"可以"运行，或回复"跳过"直接开始训练）

- 如果用户确认运行 → 生成代码中 `ENABLE_GRADIENT_CHECK = True`，先跑梯度检验，通过后再训练
- 如果用户跳过 → `ENABLE_GRADIENT_CHECK = False`，直接训练

### 步骤 8：生成代码

在当前工作目录生成代码文件。文件名规则：

- 统一使用 `neural_network_{YYYYMMDD}_{HHMMSS}.py`

模板选择（决策树）：

```
K > 2（多分类）？
  │
  ├── 是 → 读取 references/multi-class-nn.md 生成代码
  │
  └── 否（二分类）
        ├── n_features = 1 → 使用本文件「模板 B：1 特征」
        └── n_features ≥ 2 → 使用本文件「模板 A：2 特征」（最常用）
```

将所有 `{PLACEHOLDER}` 替换为实际值。

**代码自适应规则：**
- 如果 `hidden_layer_sizes` 为 `[h1]` → 生成 1 隐藏层代码（Theta1 + Theta2）
- 如果 `hidden_layer_sizes` 为 `[h1, h2]` → 生成 2 隐藏层代码（Theta1 + Theta2 + Theta3）
- 如果 `ENABLE_GRADIENT_CHECK = True` → 梯度检验代码不注释，`if True:` 块执行

### 步骤 9：运行代码

```bash
cd "e:\py 代码编辑（机器学习）" && python neural_network_{timestamp}.py
```

如果运行报错，分析错误并修复代码后重新运行。

### 步骤 10：输出结果

向用户报告：
- **网络结构**：{n} → {hidden_layer_sizes} → {K}
- **总参数量**：{total_params}
- **训练集准确率**
- （如果切分了）**验证集准确率**
- **预测结果**：概率 + 预测类别
- **最终代价 J** 和使用的 λ 值
- **梯度检验结果**（如运行了的话）
- 图片保存路径和代码文件路径

如果 `do_split = True`，额外输出：

> ✅ 模型已预留验证数据。训练时只用训练集（{train_size} 条），验证集（{val_size} 条）未参与训练。
> 如需进一步诊断模型效果（偏差/方差分析、学习曲线、λ 网格搜索），可使用 **`neural-network-diagnosis`** 技能。

如果 `do_split = False`：

> ⚠️ 模型使用全部 {m} 条数据训练，未预留验证集。无法客观评估泛化效果。
> 如需诊断模型，可使用 **`neural-network-diagnosis`** 技能（会重新划分数据并重新训练）。

---

## 代码模板 A：二分类 + 2 特征 + 1 隐藏层

> 最常用模板。2 张图输出（决策边界填充 + 收敛曲线）。1×2 布局。
> 替换所有 `{PLACEHOLDER}`。中文字符用 ASCII 替代。Windows GBK 兼容。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ===================================================================
#                        神经网络二分类预测
#  纯 NumPy 实现：前向传播 + 反向传播 + 梯度下降 + L2 正则化
# ===================================================================

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 特征矩阵 (m, n)
y = np.array([{Y_DATA}]).reshape(-1, 1)   # 标签 (m, 1)，值为 0 或 1

m, n = X_raw.shape                               # 样本数, 特征数
x_names = {X_NAMES}                               # 特征名列表（中文）
x_units = {X_UNITS}                               # 特征单位列表
y_classes = {Y_CLASSES}                           # 类别映射

print("=" * 60)
print(f"  样本数: {m}, 特征数: {n}")
print(f"  特征: {x_names}")
print(f"  类别: {y_classes}")
print("=" * 60)

# ========== 2. 数据切分 ==========
DO_SPLIT = {DO_SPLIT}

if DO_SPLIT:
    train_idx = {TRAIN_IDX}
    val_idx   = {VAL_IDX}
    test_idx  = {TEST_IDX}

    X_train_raw = X_raw[train_idx]
    y_train = y[train_idx]
    X_val_raw   = X_raw[val_idx]
    y_val   = y[val_idx]
    X_test_raw  = X_raw[test_idx]
    y_test  = y[test_idx]

    m_train = len(y_train)
    m_val   = len(y_val)
    m_test  = len(y_test)

    print(f"\n数据切分: 训练{m_train}条 / 验证{m_val}条 / 测试{m_test}条")

    # 只在训练集上计算标准化参数（避免信息泄漏到验证/测试集）
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std  = X_train_raw.std(axis=0, keepdims=True)

    # 用训练集的参数标准化所有集合
    X_train_scaled = (X_train_raw - X_mean) / X_std
    X_val_scaled   = (X_val_raw - X_mean) / X_std
    X_test_scaled  = (X_test_raw - X_mean) / X_std

    # 加偏置列
    X = np.hstack([np.ones((m_train, 1)), X_train_scaled])      # (m_train, n+1)
    X_val_full = np.hstack([np.ones((m_val, 1)), X_val_scaled]) # (m_val, n+1)
    X_test_full = np.hstack([np.ones((m_test, 1)), X_test_scaled])  # (m_test, n+1)

    print(f"X_mean: {X_mean}")
    print(f"X_std:  {X_std}")
else:
    # 全部数据用于训练（不切分）
    X_mean = X_raw.mean(axis=0, keepdims=True)
    X_std  = X_raw.std(axis=0, keepdims=True)
    X_scaled = (X_raw - X_mean) / X_std
    X = np.hstack([np.ones((m, 1)), X_scaled])                  # (m, n+1)

    print(f"\nX_mean: {X_mean}")
    print(f"X_std:  {X_std}")

# ========== 3. 网络结构 + 随机初始化 ==========
input_size  = n                                    # 输入特征数（不含偏置）
hidden_size = {HIDDEN_SIZE}                        # 隐藏层神经元数
output_size = 1                                    # 二分类输出

total_params = (input_size + 1) * hidden_size + (hidden_size + 1) * output_size
print(f"\n网络结构: {input_size} → {hidden_size} → {output_size}")
print(f"总参数量: {total_params}")

# Xavier 初始化（打破对称性，适应 sigmoid 激活函数）
np.random.seed(42)
Theta1 = np.random.randn(hidden_size, input_size + 1) * np.sqrt(1.0 / (input_size + 1))
Theta2 = np.random.randn(output_size, hidden_size + 1) * np.sqrt(1.0 / (hidden_size + 1))

print(f"Theta1 形状: {Theta1.shape}")
print(f"Theta2 形状: {Theta2.shape}")

# ========== 4. 核心函数 ==========

def sigmoid(z):
    """Sigmoid 激活函数 g(z) = 1/(1+e^(-z))"""
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """Sigmoid 导数 g'(z) = g(z)(1-g(z))"""
    g = sigmoid(z)
    return g * (1 - g)


def forward_propagation(X, Theta1, Theta2):
    """
    前向传播，计算各层激活值。

    参数:
        X      - 输入矩阵 (m, n+1)，已含偏置列
        Theta1 - 隐藏层权重 (hidden_size, n+1)
        Theta2 - 输出层权重 (1, hidden_size+1)

    返回:
        a1      - 输入层激活 (m, n+1)
        z2      - 隐藏层线性组合 (m, hidden_size)
        a2      - 隐藏层激活 (m, hidden_size)
        a2_bias - 隐藏层激活+偏置 (m, hidden_size+1)
        z3      - 输出层线性组合 (m, 1)
        a3      - 输出层激活 = 最终预测概率 (m, 1)
    """
    m_samples = X.shape[0]

    a1 = X                                               # (m, n+1)
    z2 = a1 @ Theta1.T                                   # (m, hidden_size)
    a2 = sigmoid(z2)                                     # (m, hidden_size)
    a2_bias = np.hstack([np.ones((m_samples, 1)), a2])   # (m, hidden_size+1)
    z3 = a2_bias @ Theta2.T                              # (m, 1)
    a3 = sigmoid(z3)                                     # (m, 1)

    return a1, z2, a2, a2_bias, z3, a3


def cost_function(X, y, Theta1, Theta2, lam):
    """
    交叉熵代价函数 + L2 正则化。

    J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        + (λ/(2m)) * [Σ(Theta1_[:,1:])² + Σ(Theta2_[:,1:])²]

    注意：偏置对应的权重不参与正则化（[:, 1:] 跳过第 0 列）。
    """
    m_samples = len(y)
    _, _, _, _, _, h = forward_propagation(X, Theta1, Theta2)

    # 交叉熵
    J = -(1 / m_samples) * np.sum(
        y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10)
    )

    # L2 正则化项（跳过偏置列 theta0）
    reg = (lam / (2 * m_samples)) * (
        np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)
    )

    return J + reg


def backpropagation(X, y, Theta1, Theta2, lam):
    """
    反向传播算法，计算代价函数关于每个参数的梯度。

    参数:
        X, y, Theta1, Theta2, lam - 同 forward_propagation

    返回:
        Theta1_grad - (hidden_size, n+1)
        Theta2_grad - (1, hidden_size+1)

    算法步骤:
        1. 前向传播，保存中间值
        2. 输出层误差: δ³ = a³ - y
        3. 隐藏层误差: δ² = (δ³ @ Theta2_[:,1:]) ⊙ g'(z²)
        4. 梯度: ∂J/∂Θ = (1/m) * δ @ a_prev
        5. 正则化梯度: ∂J/∂Θ[:,1:] += (λ/m) * Θ[:,1:]
    """
    m_samples = len(y)

    # 前向传播
    a1, z2, a2, a2_bias, z3, a3 = forward_propagation(X, Theta1, Theta2)

    # 输出层误差 δ³ = h - y    (m, 1)
    delta3 = a3 - y

    # 隐藏层误差 δ² = (δ³ @ Theta2_no_bias) ⊙ g'(z²)    (m, hidden_size)
    Theta2_no_bias = Theta2[:, 1:]                     # 去掉偏置列
    delta2 = (delta3 @ Theta2_no_bias) * sigmoid_gradient(z2)

    # 累积梯度
    Theta2_grad = (1 / m_samples) * (delta3.T @ a2_bias)     # (1, hidden_size+1)
    Theta1_grad = (1 / m_samples) * (delta2.T @ a1)          # (hidden_size, n+1)

    # L2 正则化梯度（偏置 theta0 不参与惩罚）
    Theta2_grad[:, 1:] += (lam / m_samples) * Theta2[:, 1:]
    Theta1_grad[:, 1:] += (lam / m_samples) * Theta1[:, 1:]

    return Theta1_grad, Theta2_grad


def gradient_descent(X, y, Theta1, Theta2, alpha, lam, iters):
    """
    批量梯度下降训练神经网络。

    每轮迭代:
        1. 反向传播计算梯度
        2. 更新参数: Θ = Θ - α * ∇Θ
        3. 记录代价
    """
    J_history = np.zeros(iters)

    for i in range(iters):
        Theta1_grad, Theta2_grad = backpropagation(X, y, Theta1, Theta2, lam)
        Theta1 -= alpha * Theta1_grad
        Theta2 -= alpha * Theta2_grad
        J_history[i] = cost_function(X, y, Theta1, Theta2, lam)

    return Theta1, Theta2, J_history


# ========== 5. 梯度检验（调试用）==========
ENABLE_GRADIENT_CHECK = {ENABLE_GC}   # True=运行检验, False=跳过

if ENABLE_GRADIENT_CHECK:
    print("\n" + "=" * 60)
    print("  梯度检验 (Gradient Checking)")
    print("  验证反向传播梯度是否正确...")
    print("=" * 60)

    def gradient_check(X, y, Theta1, Theta2, lam, epsilon=1e-4):
        """
        数值梯度 vs 解析梯度比较。

        数值梯度（双侧差分）:
          ∂J/∂θ ≈ [J(θ+ε) - J(θ-ε)] / (2ε)

        与反向传播的解析梯度比对:
          diff = ||num_grad - bp_grad|| / ||num_grad + bp_grad||
          diff < 1e-7  → 反向传播正确
        """
        # 展开所有参数为一维数组
        params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
        Theta1_grad_bp, Theta2_grad_bp = backpropagation(X, y, Theta1, Theta2, lam)
        grad_bp = np.concatenate([Theta1_grad_bp.ravel(), Theta2_grad_bp.ravel()])

        num_grad = np.zeros_like(params)
        total = len(params)

        t_start = time.time()
        for i in range(total):
            # θ + ε
            params_plus = params.copy()
            params_plus[i] += epsilon
            T1_p = params_plus[:Theta1.size].reshape(Theta1.shape)
            T2_p = params_plus[Theta1.size:].reshape(Theta2.shape)
            J_plus = cost_function(X, y, T1_p, T2_p, lam)

            # θ - ε
            params_minus = params.copy()
            params_minus[i] -= epsilon
            T1_m = params_minus[:Theta1.size].reshape(Theta1.shape)
            T2_m = params_minus[Theta1.size:].reshape(Theta2.shape)
            J_minus = cost_function(X, y, T1_m, T2_m, lam)

            num_grad[i] = (J_plus - J_minus) / (2 * epsilon)

            # 进度显示（每 10 个参数打印一次）
            if (i + 1) % max(1, total // 10) == 0:
                print(f"  进度: {i+1}/{total} 参数已检验...")

        t_elapsed = time.time() - t_start
        print(f"  检验完成，耗时 {t_elapsed:.1f} 秒")

        # 计算差异
        diff = np.linalg.norm(num_grad - grad_bp) / np.linalg.norm(num_grad + grad_bp)
        return diff

    # 执行梯度检验
    gc_diff = gradient_check(X, y, Theta1, Theta2, {LAMBDA})

    print(f"\n  梯度检验差异: {gc_diff:.2e}")
    if gc_diff < 1e-7:
        print("  ✅ 反向传播梯度计算正确！可以放心训练。")
    elif gc_diff < 1e-4:
        print("  ⚠️  差异略大（可能在浮点误差范围内），建议检查学习率后继续训练。")
    else:
        print("  ❌ 反向传播可能存在问题！请检查代码后再训练。")
        print("     常见原因: 正则化梯度未正确添加、sigmoid_gradient 计算错误。")
    print("=" * 60)


# ========== 6. 训练 ==========
alpha = {ALPHA}
lam   = {LAMBDA}
iters = {ITERS}

print(f"\n训练参数: alpha={alpha}, lambda={lam}, iters={iters}")
print("训练中...")

Theta1, Theta2, J_history = gradient_descent(X, y, Theta1, Theta2, alpha, lam, iters)

print(f"初始代价: {J_history[0]:.6f}")
print(f"最终代价: {J_history[-1]:.6f}")
print(f"代价下降: {J_history[0] - J_history[-1]:.6f}")

# ========== 7. 模型评估 ==========
_, _, _, _, _, h_train = forward_propagation(X, Theta1, Theta2)
y_pred_train = (h_train >= 0.5).astype(int)

if DO_SPLIT:
    y_ref = y_train
    n_eval = m_train
    accuracy = np.mean(y_pred_train == y_ref) * 100
    print(f"\n训练集准确率 ({m_train} 条): {accuracy:.1f}%")

    # 验证集评估
    _, _, _, _, _, h_val = forward_propagation(X_val_full, Theta1, Theta2)
    y_pred_val = (h_val >= 0.5).astype(int)
    val_accuracy = np.mean(y_pred_val == y_val) * 100
    print(f"验证集准确率 ({m_val} 条): {val_accuracy:.1f}%")
else:
    y_ref = y
    n_eval = m
    accuracy = np.mean(y_pred_train == y_ref) * 100
    print(f"\n训练集准确率: {accuracy:.1f}%")

print(f"训练集预测详情:")
for i in range(n_eval):
    if DO_SPLIT:
        xi = X_train_raw[i]
        yi = int(y_train[i, 0])
    else:
        xi = X_raw[i]
        yi = int(y[i, 0])
    print(f"  样本{i+1}: x=({', '.join([f'{xi[j]:.2f}' for j in range(n)])}), "
          f"真实y={yi} ({y_classes[yi]}), "
          f"概率={h_train[i,0]:.4f}, 预测y={y_pred_train[i,0]} ({y_classes[y_pred_train[i,0]]})")

if DO_SPLIT:
    print(f"\n验证集预测详情:")
    for i in range(m_val):
        xi = X_val_raw[i]
        yi = int(y_val[i, 0])
        print(f"  验证样本{i+1}: x=({', '.join([f'{xi[j]:.2f}' for j in range(n)])}), "
              f"真实y={yi} ({y_classes[yi]}), "
              f"概率={h_val[i,0]:.4f}, 预测y={y_pred_val[i,0]} ({y_classes[y_pred_val[i,0]]})")

# ========== 8. 预测新样本 ==========
new_x = np.array([{PREDICT_X}])              # 新样本特征值，形状 (1, n)
new_x_scaled = (new_x - X_mean) / X_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
_, _, _, _, _, prob_new_arr = forward_propagation(X_new, Theta1, Theta2)
prob_new = prob_new_arr[0, 0]
pred_new = 1 if prob_new >= 0.5 else 0

print(f"\n{'='*60}")
print(f"  新样本预测")
print(f"  x = ({', '.join([f'{new_x[0,j]}' for j in range(n)])})")
print(f"  预测概率 P(y=1) = {prob_new:.4f} ({prob_new*100:.1f}%)")
print(f"  预测结果: {y_classes[pred_new]}")
print(f"{'='*60}")

# ========== 9. 可视化（2 图，1×2 布局）==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----- 左图：决策边界 + 数据分布（俯视图）-----
ax = axes[0]

# 创建网格点
x1_min, x1_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
x2_min, x2_max = X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

# 对每个网格点做前向传播 → 概率
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_scaled = (grid_points - X_mean) / X_std
X_grid = np.hstack([np.ones((grid_points.shape[0], 1)), grid_scaled])
_, _, _, _, _, h_grid = forward_propagation(X_grid, Theta1, Theta2)
h_grid = h_grid.reshape(x1_grid.shape)

# 填充决策区域：P < 0.5 → 红色（0 类），P ≥ 0.5 → 绿色（1 类）
ax.contourf(x1_grid, x2_grid, h_grid, levels=[0, 0.5, 1],
            alpha=0.25, colors=['#e74c3c', '#2ecc71'])

# h=0.5 决策边界线
ax.contour(x1_grid, x2_grid, h_grid, levels=[0.5],
           colors='blue', linewidths=2.5, zorder=3)

# 额外概率等高线（0.3, 0.7）
ax.contour(x1_grid, x2_grid, h_grid, levels=[0.3, 0.7],
           colors='gray', linewidths=0.8, linestyles='--', alpha=0.5)

# 训练数据散点
if DO_SPLIT:
    plot_X = X_train_raw
    plot_y = y_train
else:
    plot_X = X_raw
    plot_y = y

for yi_val in [0, 1]:
    mask = (plot_y.flatten() == yi_val)
    if mask.any():
        c = '#e74c3c' if yi_val == 0 else '#2ecc71'
        mkr = 'x' if yi_val == 0 else 'o'
        ax.scatter(plot_X[mask, 0], plot_X[mask, 1], c=c, marker=mkr,
                   s=120, label=y_classes[yi_val], zorder=5,
                   edgecolors='black', linewidth=0.5)

# 验证集数据点（如果切分了，用方形标记显示）
if DO_SPLIT:
    for yi_val in [0, 1]:
        mask = (y_val.flatten() == yi_val)
        if mask.any():
            c = '#e74c3c' if yi_val == 0 else '#2ecc71'
            ax.scatter(X_val_raw[mask, 0], X_val_raw[mask, 1], c=c, marker='s',
                       s=80, label=f'{y_classes[yi_val]} (val)', zorder=5,
                       edgecolors='black', linewidth=0.8, alpha=0.6)

# 新预测点
pred_color = '#2ecc71' if pred_new == 1 else '#e74c3c'
ax.scatter([new_x[0, 0]], [new_x[0, 1]], c='#3498db', marker='*',
           s=250, zorder=10, edgecolors='black', linewidth=0.5,
           label=f'New: P={prob_new:.1%} → {y_classes[pred_new]}')

train_label = f'训练集({m_train}条)' if DO_SPLIT else f'{m}条全量数据'
ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
ax.set_ylabel(f'{x_names[1]} ({x_units[1]})', fontsize=11)
ax.set_title(f'(A) Decision Boundary & Data [{train_label}]\nNN [{input_size}-{hidden_size}-{output_size}], '
             f'lambda={lam}', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

# ----- 右图：梯度下降收敛曲线 -----
ax2 = axes[1]
ax2.plot(range(iters), J_history, 'b-', linewidth=1.2)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Cost J(Theta)', fontsize=11)
ax2.set_title(f'(B) Convergence Curve\n'
              f'alpha={alpha}, lambda={lam}, iters={iters}',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 标注起始和终止代价
ax2.annotate(f'Start: {J_history[0]:.4f}',
             xy=(0, J_history[0]),
             xytext=(iters * 0.05, J_history[0] + (J_history[0] - J_history[-1]) * 0.15),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             fontsize=9, color='red')
ax2.annotate(f'End: {J_history[-1]:.4f}',
             xy=(iters - 1, J_history[-1]),
             xytext=(iters * 0.55, J_history[-1] + (J_history[0] - J_history[-1]) * 0.25),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
             fontsize=9, color='green')

plt.tight_layout()
plt.savefig('{FILENAME_BASE}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: {FILENAME_BASE}.png")
print(f"代码文件: {FILENAME_BASE}.py")
```

---

## 代码模板 B：二分类 + 1 特征 + 1 隐藏层

> 与模板 A 共用 1-8 段（数据→预测），仅可视化部分不同。
> 1 个特征无法画 2D 决策边界，降级为 **NN 概率曲线 + 收敛曲线**。

将模板 A 的「第 9 段：可视化」替换为以下代码：

```python
# ========== 9. 可视化（1 特征降级：2 图，1×2 布局）==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----- 左图：NN 概率曲线 + 训练数据 + 预测点 -----
ax = axes[0]

# 生成平滑曲线
x_line = np.linspace(X_raw.min() * 0.9, X_raw.max() * 1.1, 200).reshape(-1, 1)
x_line_scaled = (x_line - X_mean) / X_std
X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
_, _, _, _, _, h_line = forward_propagation(X_line, Theta1, Theta2)
ax.plot(x_line, h_line, 'b-', linewidth=2.5,
        label=f'NN Probability (hidden={hidden_size})')

# h=0.5 决策阈值线
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
           label='Decision Threshold (h=0.5)')

# 训练数据点（y=0 贴底，y=1 贴顶）
if DO_SPLIT:
    plot_X_1d = X_train_raw
    plot_y_1d = y_train
else:
    plot_X_1d = X_raw
    plot_y_1d = y

for yi_val in [0, 1]:
    mask = (plot_y_1d.flatten() == yi_val)
    if mask.any():
        c = '#e74c3c' if yi_val == 0 else '#2ecc71'
        mkr = 'x' if yi_val == 0 else 'o'
        y_pos = np.zeros_like(plot_X_1d[mask]) if yi_val == 0 else np.ones_like(plot_X_1d[mask])
        ax.scatter(plot_X_1d[mask], y_pos, c=c, marker=mkr, s=120,
                   label=y_classes[yi_val], zorder=5, edgecolors='black', linewidth=0.5)

# 验证集数据点（如果切分了，用方形标记）
if DO_SPLIT:
    for yi_val in [0, 1]:
        mask = (y_val.flatten() == yi_val)
        if mask.any():
            c = '#e74c3c' if yi_val == 0 else '#2ecc71'
            y_pos = np.zeros_like(X_val_raw[mask]) if yi_val == 0 else np.ones_like(X_val_raw[mask])
            ax.scatter(X_val_raw[mask], y_pos, c=c, marker='s', s=80,
                       label=f'{y_classes[yi_val]} (val)', zorder=5,
                       edgecolors='black', linewidth=0.8, alpha=0.6)

# 新预测点
ax.scatter([new_x[0, 0]], [prob_new], c='#3498db', marker='*',
           s=250, zorder=10, edgecolors='black', linewidth=0.5,
           label=f'New: P={prob_new:.1%} → {y_classes[pred_new]}')

train_label_1d = f'训练集({m_train}条)' if DO_SPLIT else f'{m}条全量数据'
ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
ax.set_ylabel('P(y=1)', fontsize=11)
ax.set_title(f'(A) NN Probability Curve [{train_label_1d}]\n[{input_size}-{hidden_size}-{output_size}], '
             f'lambda={lam}', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# ----- 右图：收敛曲线（同模板 A）-----
ax2 = axes[1]
ax2.plot(range(iters), J_history, 'b-', linewidth=1.2)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Cost J(Theta)', fontsize=11)
ax2.set_title(f'(B) Convergence Curve\n'
              f'alpha={alpha}, lambda={lam}, iters={iters}',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax2.annotate(f'Start: {J_history[0]:.4f}',
             xy=(0, J_history[0]),
             xytext=(iters * 0.05, J_history[0] + (J_history[0] - J_history[-1]) * 0.15),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             fontsize=9, color='red')
ax2.annotate(f'End: {J_history[-1]:.4f}',
             xy=(iters - 1, J_history[-1]),
             xytext=(iters * 0.55, J_history[-1] + (J_history[0] - J_history[-1]) * 0.25),
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

| 占位符 | 规则 | 示例 |
|--------|------|------|
| `{X_DATA}` | 特征矩阵，每行一个样本 | `[3, 0.20], [4, 0.25], [5, 0.28], ...` |
| `{Y_DATA}` | 标签列表（0 或 1）| `0, 0, 0, 0, 1, 1, 1, 1, 1, 1` |
| `{X_NAMES}` | Python list，特征中文名 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | Python list，特征单位 | `["cm", ""]` |
| `{Y_CLASSES}` | 类别映射字典 | `{0: "不节能", 1: "节能"}` |
| `{DO_SPLIT}` | 是否切分数据（True/False）| `True` |
| `{TRAIN_IDX}` | 训练集索引列表（仅 DO_SPLIT=True 时）| `[0, 1, 3, 5, 6, 8, 10, 12, 14]` |
| `{VAL_IDX}` | 验证集索引列表（仅 DO_SPLIT=True 时）| `[2, 7, 13]` |
| `{TEST_IDX}` | 测试集索引列表（仅 DO_SPLIT=True 时）| `[4, 9, 11]` |
| `{HIDDEN_SIZE}` | 第 1 隐藏层神经元数 | `6` |
| `{HIDDEN_LAYER_SIZES}` | 隐藏层结构（Python list）| `[6]` |
| `{ALPHA}` | 学习率 | `0.1` |
| `{LAMBDA}` | L2 正则化强度（0=关闭）| `0.01` |
| `{ITERS}` | 迭代次数 | `3000` |
| `{ENABLE_GC}` | 梯度检验开关（True/False）| `True` 或 `False` |
| `{PREDICT_X}` | 新样本特征值（逗号分隔）| `7, 0.33` |
| `{FILENAME_BASE}` | 时间戳（不含扩展名）| `neural_network_20260730_150000` |
| `{TITLE_TOPIC}` | 英文预测主题 | `Building Energy Efficiency` |
| `{TOTAL_PARAMS}` | 总参数量（仅用于报告）| `31` |

---

## 关键注意事项

### 编码问题
Windows GBK 环境下，print 中的 Unicode 必须替换：`θ₀→theta0`, `λ→lambda`, `δ→delta`。
matplotlib 的 label/title 不走 GBK，可以用 Unicode。

### 无头环境
必须 `matplotlib.use('Agg')`，用 `plt.close()` 代替 `plt.show()`。

### 特征缩放
神经网络的梯度下降**必须**做特征缩放。不同量级的特征会让 sigmoid 饱和、梯度消失或爆炸。z-score 标准化内置在所有模板中。

### 随机初始化
神经网络不能全零初始化（对称性问题）。使用 Xavier 初始化：`randn(...) * sqrt(1.0 / fan_in)`，让每层的输出方差接近 1。

### θ₀ 不参与正则化
偏置项只控制激活函数的平移，不参与"扭曲"决策边界。惩罚它会导致模型偏向某一类。

### 2 隐藏层扩展
当用户指定 2 个隐藏层时（`hidden_layer_sizes = [h1, h2]`），在模板 A 基础上：
1. 增加 `Theta2` 形状改为 `(h2, h1+1)`
2. 增加 `Theta3` 形状 `(1, h2+1)`
3. `forward_propagation` 增加一层：`a2 → z3 → a3 → z4 → a4`
4. `backpropagation` 增加一层误差：`δ⁴ → δ³ → δ²`
5. `cost_function` 的 reg 项增加 `Theta3[:, 1:]²`
6. `gradient_check` 的 reshape 增加 Theta3

### 梯度检验耗时估算
每个参数约 1-3ms（取决于 m 和隐藏层大小）。总耗时 ≈ `total_params × 2 × 2ms`。
建议 `total_params > 200` 时提醒用户可能很慢。

### 文件名
时间戳 `YYYYMMDD_HHMMSS`，不覆盖历史文件。
统一使用 `neural_network_` 前缀。

### 数据量检查
- 最少 `n_features + 2` 个数据点
- 建议 `m ≥ total_params`（否则极易过拟合）
- 二分类每类至少 3 个样本
- 样本数 < 5×n 时提醒用户神经网络可能不是最佳选择（逻辑回归可能更合适）

### 参考文件
- `references/multi-class-nn.md`：多分类 Softmax 神经网络代码模板，**K > 2 时使用**。包含 Softmax 函数、One-hot 编码、多类交叉熵代价、K 色决策区域可视化


