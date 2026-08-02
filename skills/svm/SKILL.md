---
name: svm
description: |
  支持向量机(SVM)分类预测 —— 基于 sklearn 实现，智能判断是否需要高斯核，支持二分类与多分类。
  当用户提到"SVM"、"支持向量机"、"svm"、"Support Vector Machine"、
  "帮我用SVM分类"、"SVM预测"、"用支持向量机"、"高斯核"、"核函数分类"、
  "最大间隔分类器"、"支持向量"等类似场景时触发。
  适用于任何"给一堆 x,y 分类标签，想用 SVM 预测新样本类别"的需求。
  不要手动写代码——必须调用此技能来统一处理。
---

# 支持向量机（SVM）分类预测

## 概述

此技能基于 sklearn 实现 SVM 分类器，核心特点：

| 特性 | 说明 |
|------|------|
| 🔍 **自动核函数判断** | 先训练线性核探针 → 准确率低则建议升级高斯核 |
| ⚙️ **智能参数推荐** | 根据数据特征自动推荐 C 和 gamma，每次推荐都征求确认 |
| 📊 **支持向量可视化** | 在决策边界图中高亮标注支持向量，直观展示"谁在决定边界" |
| 🎯 **二分类 + 多分类** | sklearn SVC 自动 One-vs-One 处理多分类 |

## 核心决策逻辑

```
用户提供数据
    │
    ▼
Step 3: 训练线性核探针 → 看准确率
    │
    ├── 准确率 ≥ 阈值（默认 85%）
    │   └── 推荐线性核 → 跳过 gamma 选择
    │
    └── 准确率 < 阈值
        └── 建议高斯核(RBF) → 进入 gamma 推荐
              │
              ▼
        Step 4: 推荐 C 值 → 询问确认
        Step 5: 推荐 gamma → 询问确认
              │
              ▼
        Step 7: 读取 references/gaussian-kernel-svm.md 生成代码
```

---

## 交互流程

严格按以下 9 步执行，不要跳过任何一步。

---

### Step 1：收集数据

向用户提问：

> ### 📊 请提供你的训练数据
>
> 1. **特征（X）**：每个样本有几个特征？每个特征叫什么名字、有什么单位？
>    - 例如："2 个特征——保温厚度(cm) 和 窗墙比(无单位)"
> 2. **标签（y）**：要分几类？每类叫什么名字？
>    - 例如："2 类——节能 和 不节能"
> 3. **数据**：请把你的训练样本贴过来（任意格式，我来解析）
>    - 例如：
>      ```
>      保温厚度 3, 窗墙比 0.20 → 不节能
>      保温厚度 8, 窗墙比 0.40 → 节能
>      ...
>      ```
> 4. **预测主题**：给这个任务起个英文标题（用于图表显示）

从用户回复中提取并确认以下变量：

| 变量 | 含义 | 示例 |
|------|------|------|
| `n_features` | 特征数量 | `2` |
| `x_names` | 特征名列表 | `["保温厚度", "窗墙比"]` |
| `x_units` | 特征单位列表 | `["cm", ""]` |
| `X_raw` | 特征矩阵（list of lists） | `[[3, 0.20], [8, 0.40], ...]` |
| `y_raw` | 原始标签列表 | `["不节能", "节能", "不节能", ...]` |
| `title_topic` | 英文预测主题 | `"Building Energy Efficiency"` |

**字符串标签自动编码：** 检测到字符串标签时，自动编码为 0, 1, 2... 并打印映射表确认：

> 检测到标签 {原始值}，已自动编码为：
> - 0 → {类别0名称}
> - 1 → {类别1名称}
>
> 是否正确？（直接回复"可以"确认）

**数据校验（发现问题时立即警告）：**
- 最少 `n_features + 2` 条样本 → 不满足则拒绝继续
- 二分类每类至少 2 个样本 → 不满足则拒绝继续
- `m < 5 × n_features` → ⚠️ 提醒数据偏少，SVM 可能过拟合
- `n_features / m > 3` → 标记 `p_gt_n = True`，这是 SVM 优势场景

---

### Step 2：判断分类类型

```python
K = len(set(y_raw))
```

> 检测到 y 有 **{K}** 个类别（{类别列表}），这是**{二分类 / 多分类}**问题。

- K = 2 → `is_binary = True`
- K > 2 → `is_binary = False`，告知用户：
  > SVM 将使用 sklearn SVC 的 **One-vs-One** 策略，共训练 **{K×(K-1)/2}** 个二分类器。

---

### Step 3：核函数判断（线性优先策略）

这是本技能的核心差异化步骤。**生成一个轻量探针脚本并运行**，根据实际训练结果做判断。

#### 3.1 确定自适应阈值

根据数据情况选择判断阈值：

| 场景 | 阈值 | 原因 |
|------|------|------|
| 默认 | **85%** | 通用标准 |
| 用户明确要求高准确率 | 90% | 严格标准 |
| 多分类 K > 5 | 70% | 多类天然更难 |
| 数据量 m < 20 | 85% | 小数据该阈值防止过早换核 |
| p >> n（特征数/样本数 > 3） | 80% | SVM 经典优势场景，线性核通常够用 |

#### 3.2 生成探针脚本

生成 `svm_probe_{YYYYMMDD}_{HHMMSS}.py`，内容精简（只含数据加载 + 标准化 + 线性 SVC 训练 + 输出准确率）：

```python
import numpy as np
from sklearn.svm import SVC

# 数据
X_raw = np.array({X_DATA})
y = np.array({Y_DATA}).ravel()

# z-score 标准化
X_mean = X_raw.mean(axis=0, keepdims=True)
X_std = X_raw.std(axis=0, keepdims=True)
X = (X_raw - X_mean) / X_std

# 线性核 SVC
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X, y)
train_acc = svm_linear.score(X, y)

print(f"ACCURACY: {train_acc:.4f}")
print(f"ACC_PCT: {train_acc*100:.1f}%")
print(f"N_SV: {len(svm_linear.support_vectors_)}")
print(f"N_SAMPLE: {len(y)}")
```

#### 3.3 运行探针并解读

运行 `python svm_probe_{timestamp}.py`，解析输出获取 `train_acc`。

#### 3.4 决策逻辑

```
train_acc ≥ 阈值？
├── 是 → 推荐线性核，use_gaussian = False
├── 阈值×0.8 ≤ train_acc < 阈值 → 建议高斯核，让用户决定
├── train_acc < 阈值×0.8 → 强烈建议高斯核，默认 use_gaussian = True
└── train_acc < 50% → ⚠️ 警告数据质量，但仍建议高斯核试试
```

#### 3.5 告知用户结果

**线性够用时：**

> ### 🔍 线性可分性检测结果
>
> - 线性核训练准确率：**{acc_pct}%**
> - 判断：数据大致线性可分 ✅
> - 建议：**使用线性核**（简单、快速、可解释、过拟合风险低）
>
> 是否使用线性核继续？（直接回复"可以"，或回复"换高斯核"尝试更复杂的非线性边界）

**线性不够用时：**

> ### 🔍 线性可分性检测结果
>
> - 线性核训练准确率：**{acc_pct}%**（低于 {threshold}% 阈值）
> - 判断：数据可能**非线性可分** 🔄
> - 建议：**升级到高斯核(RBF)**——通过核函数将数据映射到高维空间，构建弯曲的决策边界
>
> 是否使用高斯核？（直接回复"可以"确认，或回复"用线性核"保持线性）

---

### Step 4：推荐 C 值

C 控制"分类准确"和"边界宽度"的平衡：
- **C 大** → 惩罚大 → 边界窄 → 尽力分对每个样本 → 过拟合风险
- **C 小** → 惩罚小 → 边界宽 → 容忍误分类 → 欠拟合风险

**C 值推荐查表（从上到下匹配，命中即停）：**

| 场景 | 条件 | 推荐 C | 原因 |
|------|------|--------|------|
| 高维小样本 | 线性核 + p>>n | **0.5** | 高维需加强正则化防过拟合 |
| 小样本 | m < 30 | **5.0** | 数据少时需强拟合 |
| 数据充裕 | m ≥ 200 + 高斯核 | **0.5** | 数据多时保守正则化 |
| 高度非线性 | 线性探针准确率 < 60% | **10.0** | 数据复杂，需要更强拟合 |
| 用户提到噪声 | 用户描述数据含噪 | **0.1** | 宽边界容忍噪声 |
| 默认（线性核） | 以上都不匹配 + 线性核 | **1.0** | sklearn 默认，通用性好 |
| 默认（高斯核） | 以上都不匹配 + 高斯核 | **1.0** | 通用起点 |

> ### ⚙️ C 参数推荐
>
> 当前场景：{场景描述}
> 建议 **C = {recommended_C}**
>
> > - C = {recommended_C}：{一行解释}
> > - C 更大（如 {C×10}）：边界更窄，更贴合训练数据（可能过拟合）
> > - C 更小（如 {C÷10}）：边界更宽，允许一些误分类（可能欠拟合）
>
> 是否需要调整？（直接回复"可以"使用建议值，或输入新值如 `C=5`）

---

### Step 5：（条件）推荐 gamma 值

仅在 `use_gaussian = True` 时执行此步骤，否则直接跳到 Step 6。

gamma 控制高斯核的"影响范围"：
- **gamma 大**（σ 小）→ 钟形曲线窄 → 每个样本影响范围小 → 边界扭曲 → 过拟合风险
- **gamma 小**（σ 大）→ 钟形曲线宽 → 每个样本影响范围大 → 边界平滑 → 欠拟合风险

**gamma 推荐查表：**

| 场景 | 条件 | 推荐 gamma | 原因 |
|------|------|-----------|------|
| 默认 | 通用 | **`'scale'`** | sklearn 默认，= 1/(n×Var(X))，自适应 |
| 小样本 | m < 30 | **`'scale'`** | 默认即可，注意观察过拟合 |
| 高维 | n > 10 | **`'scale'`** | 高维下默认值通常最优 |
| 极度非线性 | 线性探针准确率 < 50% | **`'scale'`** | 先试默认值，效果差再调整 |

> ### 🎯 Gamma（核宽度）参数推荐
>
> 建议 **gamma = 'scale'**（sklearn 默认）
>
> > `'scale'` = 1 / ({n_features} × 特征方差)，会根据你的数据自动调整。
> > - gamma 越小 → 边界越平滑（可能欠拟合）
> > - gamma 越大 → 边界越弯曲（可能过拟合）
> > - `'scale'` 在大多数情况下是最佳默认选择
>
> 是否需要调整？（直接回复"可以"使用 `'scale'`，或输入数值如 `gamma=0.5`）

**记录：** `gamma_value = 'scale'` 或用户指定的数值（作为字符串，如 `'0.5'`）

---

### Step 6：确认预测目标

> ### 🔮 你要预测什么？
>
> 请提供新样本的特征值：
> - {x1_name} = ?
> - {x2_name} = ?
> - ...

提取 `predict_x` 列表。

---

### Step 7：生成代码

#### 模板选择决策树

```
use_gaussian（Step 3 确定）？
│
├── 否（线性核）→ 使用本文件的代码模板
│   ├── n_features ≥ 2 → 「模板 A：线性核（2+ 特征）」
│   └── n_features = 1 → 「模板 B：线性核（1 特征）」
│
└── 是（高斯核）→ 读取 references/gaussian-kernel-svm.md 的代码模板
    （该模板自动适配二分类/多分类、1 特征/2+ 特征）
```

#### 占位符替换

无论使用哪个模板，统一进行以下占位符替换：

| 占位符 | 来源 | 说明 |
|--------|------|------|
| `{X_DATA}` | Step 1 | np.array 格式，如 `[[3, 0.20], [8, 0.40]]` |
| `{Y_DATA}` | Step 1 | np.array 格式的编码后标签，如 `[0, 1, 1, 0]` |
| `{K}` | Step 2 | 类别数量 |
| `{N_FEATURES}` | Step 1 | 特征数量 |
| `{C_VALUE}` | Step 4 | C 参数值（float） |
| `{GAMMA}` | Step 5 | gamma 值（如 `'scale'` 或 `0.5`，仅高斯核模板包含） |
| `{KERNEL}` | Step 3 | `'linear'` 或 `'rbf'` |
| `{PREDICT_X}` | Step 6 | 待预测样本特征值，逗号分隔 |
| `{X_NAMES}` | Step 1 | Python list，如 `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | Step 1 | Python list，如 `["cm", ""]` |
| `{Y_CLASSES}` | Step 1 | 类别映射 dict，如 `{0: "不节能", 1: "节能"}` |
| `{TITLE_TOPIC}` | Step 1 | 英文标题 |
| `{FILENAME_BASE}` | — | 时间戳文件名（不含扩展名） |
| `{IS_BINARY}` | Step 2 | `True` 或 `False` |

**文件命名：** `svm_{YYYYMMDD}_{HHMMSS}.py`

**占位符替换规则：**
- `{X_DATA}` → 替换为 numpy 数组字面量，如 `np.array([[3, 0.20], [8, 0.40], [5, 0.28]])`
- `{Y_DATA}` → 替换为列表字面量，如 `[0, 1, 1, 0]`
- `{Y_CLASSES}` → 替换为 Python dict 字面量，如 `{0: "不节能", 1: "节能"}`
- `{PREDICT_X}` → 替换为列表，如 `[7, 0.33]`
- 所有字符串占位符（`{TITLE_TOPIC}` 等）→ 直接替换为字符串字面量

---

### Step 8：运行代码

```bash
cd "e:\py 代码编辑（机器学习）" && python svm_{timestamp}.py
```

错误处理：
1. 如果报 `ModuleNotFoundError: No module named 'sklearn'` → 提示用户 `pip install scikit-learn`
2. 如果 SVC 收敛警告 → 不影响结果，告知用户即可
3. 其他错误 → 分析原因、修复代码、重试

---

### Step 9：输出结果

从运行输出中提取关键指标，按以下格式报告：

> ### ✅ SVM 训练完成
>
> | 指标 | 值 |
> |------|-----|
> | 核函数 | {线性核 / 高斯核(RBF)} |
> | C 参数 | {C} |
> | {gamma（仅高斯核）} | {gamma} |
> | 训练准确率 | {accuracy}% |
> | 支持向量数 | {n_sv} / {m}（{ratio}%） |
> | 预测结果 | 样本 {predict_x} → **{预测类别}** |
> | 决策边界图 | `{filename}.png` |

**训练后诊断（从输出中提取并判断）：**

| 信号 | 条件 | 提示 |
|------|------|------|
| 🟢 健康 | 准确率 80%~95% + SV 占比 10%~50% | 模型状态良好 |
| 🟡 可能过拟合 | 准确率 100% 或 SV 占比 > 80% | 建议减小 C 或增大 gamma 后重试 |
| 🟡 可能欠拟合 | 准确率 < 70% | 建议增大 C 或（高斯核时）增大 gamma 后重试 |
| 🔴 数据问题 | SV 占比 > 90% | 数据可能极度混乱或标签有误 |

---

## 代码模板

---

### 模板 A：线性核（2+ 特征）

适用于 `n_features ≥ 2` 的线性核场景，自动适配二分类和多分类。

```python
# -*- coding: utf-8 -*-
"""
SVM 线性核分类 —— {TITLE_TOPIC}
核函数: linear | C: {C_VALUE} | 类别数: {K}
生成时间: {FILENAME_BASE}
"""

# ============================================================
# 段 1：加载数据
# ============================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X_raw = np.array({X_DATA})
y_raw = np.array({Y_DATA}).ravel()

n_features = {N_FEATURES}
K = {K}

x_names = {X_NAMES}
x_units = {X_UNITS}
y_classes = {Y_CLASSES}
title_topic = "{TITLE_TOPIC}"

print("=" * 60)
print(f"SVM 线性核分类 —— {title_topic}")
print(f"样本数: {{len(X_raw)}}, 特征数: {{n_features}}, 类别数: {{K}}")
print(f"C = {C_VALUE}")
print("=" * 60)

# ============================================================
# 段 2：特征缩放
# ============================================================
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
y = y_raw

print(f"\n特征均值: {{scaler.mean_}}")
print(f"特征标准差: {{scaler.scale_}}")

# ============================================================
# 段 3：训练线性 SVM
# ============================================================
model = SVC(kernel='linear', C={C_VALUE}, random_state=42)
model.fit(X, y)

# ============================================================
# 段 4：模型评估
# ============================================================
accuracy = model.score(X, y)
n_sv = sum(model.n_support_)
n_sample = len(y)
sv_ratio = n_sv / n_sample * 100

print(f"\n{'='*60}")
print(f"训练准确率: {{accuracy*100:.1f}}%")
print(f"支持向量数: {{n_sv}} / {{n_sample}} ({{sv_ratio:.1f}}%)")

# 线性核的权重（可解释性）
if K == 2:
    # 二分类：coef_ 是 (1, n_features) 的形状
    weights = model.coef_[0]
    intercept = model.intercept_[0]
    print(f"\n决策超平面:")
    terms = []
    for j in range(n_features):
        unit_str = f" ({{x_units[j]}})" if x_units[j] else ""
        terms.append(f"{{weights[j]:.4f}} * {{x_names[j]}}{{unit_str}}")
    print(f"  score = {{' + '.join(terms)}} + {{intercept:.4f}}")
    print(f"  score ≥ 0 → {{y_classes[1]}}")
    print(f"  score < 0 → {{y_classes[0]}}")

# 每类支持向量数
for i in range(K):
    print(f"  {{y_classes[i]}} 类支持向量: {{model.n_support_[i]}} 个")

# 逐样本预测
print(f"\n逐样本预测详情:")
print(f"  {{'序号':<4}} {{'真实':<8}} {{'预测':<8}} {{'到边界距离':<12}} {{'支持向量?'}}")
print(f"  {{'-'*50}}")
for i in range(n_sample):
    dist = model.decision_function(X[i:i+1])
    if K == 2:
        dist_val = dist[0]
    else:
        # 多分类：取最大投票的绝对值
        dist_val = np.max(np.abs(dist))
    is_sv = "★" if i in model.support_ else ""
    true_name = y_classes[y[i]]
    pred_name = y_classes[model.predict(X[i:i+1])[0]]
    print(f"  {{i:<4}} {{true_name:<8}} {{pred_name:<8}} {{dist_val:+.4f}}       {{is_sv}}")

print(f"{{'='*60}}")

# ============================================================
# 段 5：预测新样本
# ============================================================
x_new_raw = np.array({PREDICT_X}).reshape(1, -1)
x_new = scaler.transform(x_new_raw)
pred = model.predict(x_new)[0]
pred_name = y_classes[pred]

print(f"\n新样本预测:")
for j in range(n_features):
    unit_str = f" {{x_units[j]}}" if x_units[j] else ""
    print(f"  {{x_names[j]}} = {{x_new_raw[0][j]}}{{unit_str}}")
print(f"  预测类别: {{pred_name}}")

if K == 2:
    # 二分类输出 decision_function
    dist_new = model.decision_function(x_new)[0]
    print(f"  到超平面距离: {{dist_new:+.4f}} (正值→{{y_classes[1]}}, 负值→{{y_classes[0]}})")
else:
    # 多分类输出 predict_proba 需要 probability=True, 这里用 decision_function
    dec = model.decision_function(x_new)[0]
    print(f"  One-vs-One 决策值: {{dec}} (共 {{K*(K-1)//2}} 个二分类器)")

# ============================================================
# 段 6：可视化
# ============================================================
# 配色
if K == 2:
    class_colors = ['#e74c3c', '#3498db']   # 红 / 蓝
    region_colors = ['#fadbd8', '#d4e6f1']    # 浅红 / 浅蓝
else:
    # 多分类：使用 tab10 色板
    import matplotlib.cm as cm
    class_colors = [cm.tab10(i) for i in range(K)]
    region_colors = [cm.tab10(i, alpha=0.3) for i in range(K)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- 左图：决策边界 ----
# 只用前 2 个特征画 meshgrid
x0 = X[:, 0]
x1 = X[:, 1]
margin = 0.3
x0_min, x0_max = x0.min() - margin * (x0.max() - x0.min()), x0.max() + margin * (x0.max() - x0.min())
x1_min, x1_max = x1.min() - margin * (x1.max() - x1.min()), x1.max() + margin * (x1.max() - x1.min())

# 构造 meshgrid（只在 2D 上评估——其余维度用均值填充）
xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                       np.linspace(x1_min, x1_max, 200))
# 将 mesh 补齐到 n_features 维
mesh_mean = np.tile(scaler.mean_[2:], (xx0.ravel().shape[0], 1)) if n_features > 2 else None
if n_features == 2:
    grid_full = np.c_[xx0.ravel(), xx1.ravel()]
else:
    grid_2d = np.c_[xx0.ravel(), xx1.ravel()]
    grid_full = np.hstack([grid_2d, mesh_mean])

Z = model.predict(grid_full).reshape(xx0.shape)

if K == 2:
    ax1.contourf(xx0, xx1, Z, levels=[-0.5, 0.5, 1.5], colors=region_colors, alpha=0.6)
    # 决策边界线
    if n_features == 2:
        # 仅当只有 2 特征时能画准确的决策线
        w = model.coef_[0]
        b = model.intercept_[0]
        x1_boundary = -(w[0] * xx0[:, 0] + b) / w[1]
        ax1.plot(xx0[:, 0], x1_boundary, 'k-', linewidth=1.5, label='Decision boundary')
else:
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(class_colors[:K])
    ax1.contourf(xx0, xx1, Z, levels=np.arange(-0.5, K), colors=region_colors[:K], alpha=0.4)
    ax1.contour(xx0, xx1, Z, levels=np.arange(0.5, K-0.5), colors='gray', linewidths=0.5, linestyles='--')

# 训练样本散点
for i in range(K):
    mask = y == i
    ax1.scatter(X[mask, 0], X[mask, 1], c=[class_colors[i]], s=50,
                edgecolors='white', linewidth=0.5, label=f"{{y_classes[i]}} (n={{sum(mask)}})",
                zorder=5)

# 支持向量高亮
ax1.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=180, facecolors='none', edgecolors='black', linewidth=2,
            label=f'Support Vectors (n={{n_sv}})', zorder=10)

# 新预测点
ax1.scatter(x_new[0, 0], x_new[0, 1], s=250, c='#f1c40f', marker='*',
            edgecolors='black', linewidth=1.5, label=f'Predict: {{pred_name}}', zorder=15)

x0_label = f"{{x_names[0]}}{' (' + x_units[0] + ')' if x_units[0] else ''}"
x1_label = f"{{x_names[1]}}{' (' + x_units[1] + ')' if x_units[1] else ''}"
ax1.set_xlabel(x0_label)
ax1.set_ylabel(x1_label)
plot_title = "Linear SVM Decision Boundary"
if n_features > 2:
    plot_title += f"\n(first 2 of {{n_features}} features shown)"
ax1.set_title(plot_title)
ax1.legend(loc='best', fontsize=7)

# ---- 右图：到边界距离分析（二分类）或 各类准确率（多分类）----
if K == 2:
    # 每个样本到超平面的距离
    distances = model.decision_function(X)
    colors_bar = [class_colors[0] if d < 0 else class_colors[1] for d in distances]
    edge_colors = ['black' if i in model.support_ else 'none' for i in range(n_sample)]
    bars = ax2.bar(range(n_sample), distances, color=colors_bar, edgecolor=edge_colors, linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Distance to Hyperplane (decision_function)')
    ax2.set_title(f'Distance to Boundary\n(black edge = Support Vector, {n_sv}/{n_sample})')
else:
    # 各类别准确率
    class_acc = []
    class_names = []
    for i in range(K):
        mask_i = y == i
        if sum(mask_i) > 0:
            acc_i = (model.predict(X[mask_i]) == y[mask_i]).mean() * 100
            class_acc.append(acc_i)
            class_names.append(y_classes[i])
    bars = ax2.bar(range(K), class_acc, color=class_colors[:K], edgecolor='white')
    for i, (acc_i, bar) in enumerate(zip(class_acc, bars)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{{acc_i:.1f}}%",
                 ha='center', fontsize=9)
    ax2.set_xticks(range(K))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_ylim(0, 110)

plt.suptitle(f"SVM (Linear) — {{title_topic}}", fontsize=13, fontweight='bold')
plt.tight_layout()

# ============================================================
# 段 7：保存图表
# ============================================================
output_path = "{FILENAME_BASE}.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: {{output_path}}")

# ============================================================
# 段 8：总结输出
# ============================================================
print(f"\n{{'='*60}}")
print(f"训练完成")
print(f"核函数: linear")
print(f"C = {C_VALUE}")
print(f"准确率: {{accuracy*100:.1f}}%")
print(f"支持向量: {{n_sv}}/{{n_sample}} ({{sv_ratio:.1f}}%)")
print(f"图表: {{output_path}}")
print(f"{{'='*60}}")
```

---

### 模板 B：线性核（1 特征）

适用于 `n_features = 1` 的线性核场景。与模板 A 的区别仅在于可视化（1D 决策曲线替代 2D 边界图）。

```python
# -*- coding: utf-8 -*-
"""
SVM 线性核分类（1 特征）—— {TITLE_TOPIC}
核函数: linear | C: {C_VALUE} | 类别数: {K}
"""

# ============================================================
# 段 1-5：与模板 A 完全相同（加载、缩放、训练、评估、预测）
# ============================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X_raw = np.array({X_DATA})
y_raw = np.array({Y_DATA}).ravel()
n_features = 1
K = {K}
x_names = {X_NAMES}
x_units = {X_UNITS}
y_classes = {Y_CLASSES}
title_topic = "{TITLE_TOPIC}"

print("=" * 60)
print(f"SVM 线性核分类（1 特征）—— {title_topic}")
print(f"样本数: {len(X_raw)}, 类别数: {K}, C = {C_VALUE}")
print("=" * 60)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
y = y_raw

# 训练
model = SVC(kernel='linear', C={C_VALUE}, random_state=42)
model.fit(X, y)

# 评估
accuracy = model.score(X, y)
n_sv = sum(model.n_support_)
n_sample = len(y)
sv_ratio = n_sv / n_sample * 100

print(f"\n训练准确率: {accuracy*100:.1f}%")
print(f"支持向量: {n_sv}/{n_sample} ({sv_ratio:.1f}%)")

if K == 2:
    w = model.coef_[0][0]
    b = model.intercept_[0]
    print(f"决策公式: score = {w:.4f} * x_scaled + {b:.4f}")
    print(f"  score ≥ 0 → {y_classes[1]}, score < 0 → {y_classes[0]}")

# 逐样本预测
print(f"\n{'序号':<4} {'真实':<8} {'预测':<8} {'距离':<10} {'SV?'}")
print(f"{'-'*40}")
for i in range(n_sample):
    dist = model.decision_function(X[i:i+1])[0] if K == 2 else 0
    is_sv = "★" if i in model.support_ else ""
    print(f"  {i:<4} {y_classes[y[i]]:<8} {y_classes[model.predict(X[i:i+1])[0]]:<8} {dist:+.4f}     {is_sv}")

# 预测新样本
x_new_raw = np.array({PREDICT_X}).reshape(1, -1)
x_new = scaler.transform(x_new_raw)
pred = model.predict(x_new)[0]
pred_name = y_classes[pred]
print(f"\n新样本 {x_names[0]} = {x_new_raw[0][0]}{x_units[0] if x_units[0] else ''} → {pred_name}")

# ============================================================
# 段 6：可视化（1D 版本）
# ============================================================
if K == 2:
    class_colors = ['#e74c3c', '#3498db']
else:
    import matplotlib.cm as cm
    class_colors = [cm.tab10(i) for i in range(K)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- 左图：1D 决策曲线 ----
x_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 500).reshape(-1, 1)

if K == 2:
    # 二分类：画 decision_function 曲线
    scores = model.decision_function(x_plot)
    ax1.plot(x_plot, scores, 'b-', linewidth=2, label='decision_function')
    ax1.axhline(y=0, color='black', linewidth=1.5, linestyle='-', label='Boundary (score=0)')
    ax1.fill_between(x_plot.ravel(), scores, 0, where=(scores >= 0), color='#d4e6f1', alpha=0.4)
    ax1.fill_between(x_plot.ravel(), scores, 0, where=(scores < 0), color='#fadbd8', alpha=0.4)
    ax1.set_ylabel('score (decision_function)')
else:
    # 多分类：画预测类别区域
    preds = model.predict(x_plot)
    for i in range(K):
        mask = preds == i
        ax1.fill_between(x_plot.ravel(), -0.5, 0.5, where=mask,
                         color=class_colors[i], alpha=0.3)

# 训练样本点
for i in range(K):
    mask = y == i
    ax1.scatter(X[mask], np.zeros(sum(mask)), c=[class_colors[i]], s=60,
                edgecolors='white', linewidth=0.5, label=f"{y_classes[i]}")

# 支持向量 + 新预测点
ax1.scatter(model.support_vectors_, np.zeros(n_sv),
            s=200, facecolors='none', edgecolors='black', linewidth=2, label=f'SV ({n_sv})')
ax1.scatter(x_new[0, 0], 0, s=200, c='#f1c40f', marker='*',
            edgecolors='black', linewidth=1.5, label=f'Predict: {pred_name}')

x_label = f"{x_names[0]}{' (' + x_units[0] + ')' if x_units[0] else ''} (scaled)"
ax1.set_xlabel(x_label)
ax1.set_title(f"1D Decision Boundary — {title_topic}")
ax1.legend(fontsize=7)

# ---- 右图：距离分析 / 准确率 ----
if K == 2:
    distances = model.decision_function(X)
    colors_bar = [class_colors[0] if d < 0 else class_colors[1] for d in distances]
    edge_colors = ['black' if i in model.support_ else 'none' for i in range(n_sample)]
    ax2.bar(range(n_sample), distances, color=colors_bar, edgecolor=edge_colors, linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Distance to Boundary')
    ax2.set_title(f'Distances (black edge=SV, {n_sv}/{n_sample})')
else:
    class_acc = []
    for i in range(K):
        mask_i = y == i
        if sum(mask_i) > 0:
            class_acc.append((model.predict(X[mask_i]) == y[mask_i]).mean() * 100)
    ax2.bar(range(K), class_acc, color=class_colors[:K], edgecolor='white')
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([y_classes[i] for i in range(K)])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_ylim(0, 110)

plt.suptitle(f"SVM (Linear, 1D) — {title_topic}", fontsize=13, fontweight='bold')
plt.tight_layout()
output_path = "{FILENAME_BASE}.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: {output_path}")
print(f"\n{'='*60}")
print(f"训练完成 | 核函数: linear | C={C_VALUE} | 准确率: {accuracy*100:.1f}%")
print(f"支持向量: {n_sv}/{n_sample} ({sv_ratio:.1f}%) | 图表: {output_path}")
print(f"{'='*60}")
```

---

## 占位符说明

所有模板共享的占位符：

| 占位符 | 类型 | 示例替换结果 |
|--------|------|-------------|
| `{X_DATA}` | 数组字面量 | `np.array([[3, 0.20], [8, 0.40]])` |
| `{Y_DATA}` | 列表字面量 | `[0, 1, 1, 0]` |
| `{K}` | 整数 | `2` |
| `{N_FEATURES}` | 整数 | `2` |
| `{C_VALUE}` | 浮点数 | `1.0` |
| `{GAMMA}` | 字符串或浮点数 | `'scale'` 或 `0.5`（仅高斯核模板） |
| `{KERNEL}` | 字符串 | `'linear'` 或 `'rbf'` |
| `{PREDICT_X}` | 列表字面量 | `[7, 0.33]` |
| `{X_NAMES}` | 列表字面量 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | 列表字面量 | `["cm", ""]` |
| `{Y_CLASSES}` | 字典字面量 | `{0: "不节能", 1: "节能"}` |
| `{TITLE_TOPIC}` | 字符串 | `Building Energy Efficiency` |
| `{FILENAME_BASE}` | 字符串 | `svm_20260731_153000` |
| `{IS_BINARY}` | 布尔值 | `True` 或 `False` |

---

## 关键注意事项

### 1. 特征缩放必须做
SVM 对特征尺度敏感。所有模板在训练前统一执行 z-score 标准化：`X = (X - mean) / std`。预测新样本时必须用**同一个** scaler 做 transform。

### 2. 线性核使用 SVC(kernel='linear')
虽然 sklearn 有 LinearSVC 类（更快），但使用 `SVC(kernel='linear')` 可以获得：
- `support_vectors_`：支持向量坐标
- `n_support_`：每类支持向量数
- `coef_` / `intercept_`：线性权重（用于解释模型）
- 多分类时与高斯核模板的 API 保持一致

### 3. 多分类的 decision_function
二分类时 `decision_function` 返回标量（到超平面距离）。多分类时返回形状 `(n_samples, K×(K-1)/2)` 的数组（每对类别一个值），解读复杂。因此多分类模板主要依赖 `predict()`，仅在报告中简化展示。

### 4. 高斯核无 coef_ / intercept_
高斯核的决策边界在高维空间中，不存在简单的"权重 × 特征"公式。高斯核模板不输出 coef_/intercept_，而是通过支持向量统计和决策边界图来理解模型。

### 5. 与 model-recommendation 联动
当 model-recommendation 技能推荐 SVM 且用户有数据时，应自动跳转到本技能。反之，如果本技能判断数据不适合 SVM（如 m > 10,000），应建议用户用 model-recommendation 重新评估。

### 6. 参考文件
- `references/gaussian-kernel-svm.md`：高斯核(RBF)代码模板。当 Step 3 判断需要高斯核时使用。支持二分类和多分类，自动适配 1 特征/2+ 特征可视化。
