# 高斯核(RBF) SVM 代码模板

## 概述

本模板使用 sklearn SVC + RBF（高斯核）训练 SVM。与线性核模板的关键差异：

| 差异点 | 线性核 | 高斯核(RBF) |
|--------|--------|------------|
| 决策边界 | 直线/超平面 | 任意形状的曲线 |
| 模型可解释 | coef_/intercept_ 可读 | 无简单权重公式 |
| 支持向量 | 通常较少 | 可能更多 |
| 参数 | 仅 C | C + gamma |
| 核心 API | `SVC(kernel='linear')` | `SVC(kernel='rbf', gamma=...)` |

## 何时使用此模板

当 [../SKILL.md](../SKILL.md) 的 Step 3 判断 `use_gaussian = True` 时使用。即：
- 线性核探针准确率低于自适应阈值
- 用户确认需要高斯核

---

## 完整代码模板

```python
# -*- coding: utf-8 -*-
"""
SVM 高斯核(RBF)分类 —— {TITLE_TOPIC}
核函数: rbf | C: {C_VALUE} | gamma: {GAMMA} | 类别数: {K}
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
print(f"SVM 高斯核(RBF)分类 —— {title_topic}")
print(f"样本数: {len(X_raw)}, 特征数: {n_features}, 类别数: {K}")
print(f"C = {C_VALUE}, gamma = {GAMMA}")
print("=" * 60)

# ============================================================
# 段 2：特征缩放（SVM 必须）
# ============================================================
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
y = y_raw

print(f"\n特征均值: {scaler.mean_}")
print(f"特征标准差: {scaler.scale_}")

# 如果 gamma='scale'，显示等效值
gamma_val = {GAMMA}
if gamma_val == 'scale':
    equiv = 1.0 / (n_features * X.var())
    print(f"gamma='scale' → 等效值 ≈ {equiv:.4f} (1/({n_features} * var(X)))")

# ============================================================
# 段 3：训练高斯核 SVM
# ============================================================
model = SVC(kernel='rbf', C={C_VALUE}, gamma={GAMMA}, random_state=42)
model.fit(X, y)

# ============================================================
# 段 4：模型评估 + 支持向量分析
# ============================================================
accuracy = model.score(X, y)
n_sv_total = sum(model.n_support_)
n_sample = len(y)
sv_ratio = n_sv_total / n_sample * 100

print(f"\n{'='*60}")
print(f"训练准确率: {accuracy*100:.1f}%")
print(f"支持向量总数: {n_sv_total} / {n_sample} ({sv_ratio:.1f}%)")
print(f"  其中:")

for i in range(K):
    class_name = y_classes[i]
    n_sv_i = model.n_support_[i]
    n_class_i = sum(y == i)
    print(f"    {class_name}: {n_sv_i} 个支持向量 (共 {n_class_i} 个样本, SV 占比 {n_sv_i/n_class_i*100:.1f}%)")

# 高斯核没有 coef_/intercept_，不输出权重公式
print(f"\n说明: 高斯核没有简单的 '权重 x 特征' 公式。")
print(f"      决策由 {n_sv_total} 个支持向量通过核函数共同决定。")
print(f"      score(x) = b + sum(alpha_j * y_j * exp(-gamma * ||x - x_j||^2))")

# 逐样本预测
print(f"\n逐样本预测详情:")
print(f"  {'序号':<4} {'真实':<8} {'预测':<8} {'支持向量?'}")
print(f"  {'-'*40}")
n_correct = 0
for i in range(n_sample):
    pred_i = model.predict(X[i:i+1])[0]
    is_sv = "★" if i in model.support_ else ""
    true_name = y_classes[y[i]]
    pred_name = y_classes[pred_i]
    correct = "✓" if y[i] == pred_i else "✗"
    n_correct += (y[i] == pred_i)
    print(f"  {i:<4} {true_name:<8} {pred_name:<8} {is_sv}  {correct}")

print(f"\n正确: {n_correct}/{n_sample}")
print(f"{'='*60}")

# ============================================================
# 段 5：预测新样本
# ============================================================
x_new_raw = np.array({PREDICT_X}).reshape(1, -1)
x_new = scaler.transform(x_new_raw)
pred = model.predict(x_new)[0]
pred_name = y_classes[pred]

print(f"\n{'='*60}")
print(f"新样本预测:")
for j in range(n_features):
    unit_str = f" {x_units[j]}" if x_units[j] else ""
    print(f"  {x_names[j]} = {x_new_raw[0][j]}{unit_str}")
print(f"  预测类别: {pred_name}")

# 多分类时尝试输出决策值
if K == 2:
    dist = model.decision_function(x_new)[0]
    print(f"  到超平面距离: {dist:+.4f}")
    print(f"    (正值→{y_classes[1]}, 负值→{y_classes[0]})")
else:
    try:
        dec = model.decision_function(x_new)
        print(f"  One-vs-One 决策值形状: {dec.shape} (共 {K*(K-1)//2} 个二分类器)")
    except:
        pass
print(f"{'='*60}")

# ============================================================
# 段 6：可视化
# ============================================================
# 配色
if K == 2:
    class_colors = ['#e74c3c', '#3498db']
    region_colors = ['#fadbd8', '#d4e6f1']
else:
    import matplotlib.cm as cm
    class_colors = [cm.tab10(i) for i in range(K)]
    region_colors = [cm.tab10(i, alpha=0.3) for i in range(K)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

if n_features >= 2:
    # ======== 2D+ 可视化 ========
    x0 = X[:, 0]
    x1 = X[:, 1]
    margin = 0.3
    range0 = x0.max() - x0.min()
    range1 = x1.max() - x1.min()
    x0_min, x0_max = x0.min() - margin * range0, x0.max() + margin * range0
    x1_min, x1_max = x1.min() - margin * range1, x1.max() + margin * range1

    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 250),
                           np.linspace(x1_min, x1_max, 250))

    # 补齐到 n_features 维（其余维度用均值填充）
    if n_features == 2:
        grid_full = np.c_[xx0.ravel(), xx1.ravel()]
    else:
        grid_2d = np.c_[xx0.ravel(), xx1.ravel()]
        mesh_mean = np.tile(scaler.mean_[2:], (xx0.ravel().shape[0], 1))
        grid_full = np.hstack([grid_2d, mesh_mean])

    Z = model.predict(grid_full).reshape(xx0.shape)

    # ---- 左图：决策区域 + 支持向量 ----
    from matplotlib.colors import ListedColormap
    if K == 2:
        ax1.contourf(xx0, xx1, Z, levels=[-0.5, 0.5, 1.5], colors=region_colors, alpha=0.5)
        # 概率等高线（如果支持）
        try:
            prob = model.decision_function(grid_full).reshape(xx0.shape)
            ax1.contour(xx0, xx1, prob, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
                        linewidths=[0.8, 1.5, 0.8], linestyles=['--', '-', '--'])
        except:
            pass
    else:
        cmap = ListedColormap(class_colors[:K])
        ax1.contourf(xx0, xx1, Z, levels=np.arange(-0.5, K), colors=region_colors[:K], alpha=0.4)
        ax1.contour(xx0, xx1, Z, levels=np.arange(0.5, K-0.5),
                    colors='gray', linewidths=0.5, linestyles='--')

    # 训练样本
    for i in range(K):
        mask = y == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=[class_colors[i]], s=50,
                    edgecolors='white', linewidth=0.5, alpha=0.7,
                    label=f"{y_classes[i]} (n={sum(mask)})", zorder=5)

    # 支持向量高亮（高斯核中支持向量角色更重要）
    ax1.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='black', linewidth=2.5,
                label=f'Support Vectors ({n_sv_total})', zorder=10)

    # 新预测点
    ax1.scatter(x_new[0, 0], x_new[0, 1], s=300, c='#f1c40f', marker='*',
                edgecolors='black', linewidth=2, label=f'Predict: {pred_name}', zorder=15)

    x0_label = f"{x_names[0]}{' (' + x_units[0] + ')' if x_units[0] else ''}"
    x1_label = f"{x_names[1]}{' (' + x_units[1] + ')' if x_units[1] else ''}"
    ax1.set_xlabel(x0_label)
    ax1.set_ylabel(x1_label)
    plot_title = f"RBF SVM Decision Boundary\n(gamma={GAMMA})"
    if n_features > 2:
        plot_title += f" [first 2 of {n_features} features]"
    ax1.set_title(plot_title)
    ax1.legend(loc='best', fontsize=7)

    # ---- 右图：支持向量分布分析 ----
    # 用前 2 个特征画支持向量的分布
    sv_indices = model.support_
    non_sv_mask = np.ones(n_sample, dtype=bool)
    non_sv_mask[sv_indices] = False

    # 不同类别用不同颜色
    for i in range(K):
        # 非支持向量（小点）
        mask_i = (y == i) & non_sv_mask
        if sum(mask_i) > 0:
            ax2.scatter(X[mask_i, 0], X[mask_i, 1], c=[class_colors[i]], s=30,
                        edgecolors='none', alpha=0.4, label=f'{y_classes[i]} (non-SV)')

        # 支持向量（大空心圆）
        mask_sv_i = np.isin(sv_indices, np.where((y == i))[0])
        if sum(mask_sv_i) > 0:
            sv_coords = model.support_vectors_[mask_sv_i]
            ax2.scatter(sv_coords[:, 0], sv_coords[:, 1], s=180,
                        facecolors=class_colors[i], edgecolors='black', linewidth=2,
                        label=f'{y_classes[i]} SV ({sum(mask_sv_i)})', zorder=10)

    ax2.set_xlabel(f"{x_names[0]}{' (' + x_units[0] + ')' if x_units[0] else ''}")
    ax2.set_ylabel(f"{x_names[1]}{' (' + x_units[1] + ')' if x_units[1] else ''}")
    ax2.set_title(f'Support Vector Distribution\n({n_sv_total}/{n_sample} = {sv_ratio:.1f}%)')
    ax2.legend(loc='best', fontsize=7)

else:
    # ======== 1D 可视化 ========
    x_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 500).reshape(-1, 1)
    Z_plot = model.predict(x_plot)

    # 左图：1D 决策
    if K == 2:
        scores = model.decision_function(x_plot)
        ax1.plot(x_plot, scores, 'b-', linewidth=2, label='decision_function')
        ax1.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
        ax1.fill_between(x_plot.ravel(), scores, 0, where=(scores >= 0),
                         color='#d4e6f1', alpha=0.3)
        ax1.fill_between(x_plot.ravel(), scores, 0, where=(scores < 0),
                         color='#fadbd8', alpha=0.3)
        ax1.set_ylabel('score')
    else:
        for i in range(K):
            mask = Z_plot == i
            ax1.fill_between(x_plot.ravel(), -0.5, 0.5, where=mask,
                             color=class_colors[i], alpha=0.3)

    for i in range(K):
        mask = y == i
        ax1.scatter(X[mask], np.zeros(sum(mask)), c=[class_colors[i]], s=60,
                    edgecolors='white', linewidth=0.5, label=f"{y_classes[i]} ({sum(mask)})")
    ax1.scatter(model.support_vectors_, np.zeros(n_sv_total),
                s=200, facecolors='none', edgecolors='black', linewidth=2.5, label=f'SV ({n_sv_total})')
    ax1.scatter(x_new[0, 0], 0, s=250, c='#f1c40f', marker='*',
                edgecolors='black', linewidth=1.5, label=f'Predict: {pred_name}')
    x_label = f"{x_names[0]}{' (' + x_units[0] + ')' if x_units[0] else ''} (scaled)"
    ax1.set_xlabel(x_label)
    ax1.set_title(f"RBF SVM (1D) — {title_topic}")
    ax1.legend(fontsize=7)

    # 右图：支持向量 vs 非支持向量
    sv_mask = np.zeros(n_sample, dtype=bool)
    sv_mask[model.support_] = True
    for i in range(K):
        mask_i = y == i
        # 非 SV
        mask_non_sv = mask_i & ~sv_mask
        if sum(mask_non_sv) > 0:
            ax2.scatter(X[mask_non_sv], np.zeros(sum(mask_non_sv)), c=[class_colors[i]],
                        s=30, alpha=0.4, edgecolors='none')
        # SV
        mask_sv = mask_i & sv_mask
        if sum(mask_sv) > 0:
            ax2.scatter(X[mask_sv], np.zeros(sum(mask_sv)), c=[class_colors[i]],
                        s=200, edgecolors='black', linewidth=2,
                        label=f'{y_classes[i]} SV ({sum(mask_sv)})')
    ax2.set_xlabel(x_label)
    ax2.set_title(f'SV Distribution ({n_sv_total}/{n_sample} = {sv_ratio:.1f}%)')
    ax2.legend(fontsize=7)
    ax2.set_ylim(-0.5, 0.5)

plt.suptitle(f"SVM (RBF, gamma={GAMMA}) — {title_topic}", fontsize=13, fontweight='bold')
plt.tight_layout()

# ============================================================
# 段 7：保存图表
# ============================================================
output_path = "{FILENAME_BASE}.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: {output_path}")

# ============================================================
# 段 8：参数总结
# ============================================================
print(f"\n{'='*60}")
print(f"高斯核 SVM 训练完成")
print(f"{'='*60}")
print(f"核函数: RBF (Gaussian Kernel)")
print(f"  K(x, l) = exp(-gamma * ||x - l||^2)")
print(f"  gamma = {GAMMA}")
if gamma_val == 'scale':
    print(f"    (= 1/({n_features} * var(X)) ≈ {equiv:.4f})")
print(f"C = {C_VALUE}")
print(f"  C 大 → 边界窄/过拟合 | C 小 → 边界宽/欠拟合")
print(f"gamma = {GAMMA}")
print(f"  gamma 大 → 影响范围小/边界扭曲 | gamma 小 → 影响范围大/边界平滑")
print(f"准确率: {accuracy*100:.1f}%")
print(f"支持向量: {n_sv_total}/{n_sample} ({sv_ratio:.1f}%)")

# ============================================================
# 段 9：过拟合/欠拟合诊断
# ============================================================
print(f"\n--- 模型诊断 ---")
if accuracy > 0.99:
    print(f"⚠ 准确率 = 100%，可能过拟合")
    print(f"  建议: 减小 C (如 C={C_VALUE/10:.3f}) 或 增大 gamma (如 gamma={C_VALUE*10})")
elif sv_ratio > 80:
    print(f"⚠ 支持向量占比 {sv_ratio:.1f}%（>80%），疑似过拟合")
    print(f"  建议: 减小 C 使边界更平滑，或减小 gamma 增大影响范围")
elif accuracy < 0.70:
    print(f"⚠ 准确率 {accuracy*100:.1f}%（<70%），可能欠拟合")
    print(f"  建议: 增大 C (如 C={C_VALUE*10}) 或 增大 gamma (如 gamma={C_VALUE*5})")
else:
    print(f"✓ 模型状态看起来健康")
    if sv_ratio < 20:
        print(f"  支持向量占比很低 ({sv_ratio:.1f}%)，边界简洁高效")
    elif sv_ratio < 50:
        print(f"  支持向量占比适中 ({sv_ratio:.1f}%)，边界复杂度和泛化能力平衡")
    else:
        print(f"  支持向量占比偏高 ({sv_ratio:.1f}%)，但准确率合理")

print(f"\n图表: {output_path}")
print(f"{'='*60}")
```

---

## 占位符说明

| 占位符 | 类型 | 说明 |
|--------|------|------|
| `{X_DATA}` | 数组字面量 | 特征矩阵 |
| `{Y_DATA}` | 列表字面量 | 标签列表 |
| `{K}` | 整数 | 类别数 |
| `{N_FEATURES}` | 整数 | 特征数量 |
| `{C_VALUE}` | 浮点数 | C 参数值 |
| `{GAMMA}` | 字符串或浮点数 | gamma 值，通常为 `'scale'` |
| `{PREDICT_X}` | 列表字面量 | 待预测样本特征值 |
| `{X_NAMES}` | 列表字面量 | 特征名列表 |
| `{X_UNITS}` | 列表字面量 | 特征单位列表 |
| `{Y_CLASSES}` | 字典字面量 | 类别映射 |
| `{TITLE_TOPIC}` | 字符串 | 英文预测主题 |
| `{FILENAME_BASE}` | 字符串 | 时间戳文件名前缀 |

---

## 注意事项

### 1. gamma = 'scale' 是首选
sklearn 0.22+ 的默认值，自动根据特征数和方差计算：`gamma = 1/(n_features * Var(X))`。在大多数情况下是合理的起点，先试默认值再按诊断提示调整。

### 2. 高斯核训练比线性核慢
高斯核需要计算每个样本对之间的核函数值，时间复杂度 O(m² × n)。样本数超过 5000 时训练会明显变慢。

### 3. 支持向量占比是高斯的"体检报告"
- **< 20%**：边界简洁，泛化好
- **20%~50%**：正常范围
- **50%~80%**：边界偏复杂，观察准确率
- **> 80%**：几乎每个样本都是支持向量 → 过拟合信号

### 4. 特征数 > 100 时审慎使用高斯核
高维空间中所有点之间的距离趋于相等（维度灾难），高斯核的"距离度量"会失效。高维场景优先用线性核。

### 5. 需要概率时
SVC 默认不输出概率。如需概率输出，在代码中修改为：
```python
model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
```
这会启用 Platt Scaling（5-fold 交叉验证拟合 sigmoid），训练时间增加约 5 倍。
