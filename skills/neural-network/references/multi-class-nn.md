# 多分类神经网络（Multi-Class Neural Network with Softmax）

## 概述

此模板用于 **K ≥ 3 个类别** 的多分类问题。核心算法：
- **Softmax 输出层**：K 个神经元，输出 K 个概率（和为 1）
- **多类交叉熵代价**：`J = -(1/m) Σ Σ y_ik * log(h_ik)`
- **L2 正则化**：默认启用，λ 可调
- **纯 NumPy 反向传播**：从零实现

## 何时使用

当 `len(np.unique(y)) > 2` 时使用此模板。不用于二分类（二分类走 SKILL.md 内置的 Sigmoid 模板）。

## 代码模板

> 替换所有 `{PLACEHOLDER}`。Windows GBK 兼容。中文字符用 ASCII 替代。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ===================================================================
#                多分类神经网络预测 (Softmax)
#  纯 NumPy 实现：前向传播 + 反向传播 + 梯度下降 + L2 正则化
# ===================================================================

# ========== 1. 加载数据 ==========
X_raw = np.array([{X_DATA}])    # 特征矩阵 (m, n)
y = np.array([{Y_DATA}]).reshape(-1, 1)   # 标签 (m, 1)，值为 0, 1, ..., K-1

m, n = X_raw.shape                               # 样本数, 特征数
K = {K}                                          # 类别数
x_names = {X_NAMES}                               # 特征名列表（中文）
x_units = {X_UNITS}                               # 特征单位列表
y_classes = {Y_CLASSES}                           # 类别映射，如 {0: "A级", 1: "B级", 2: "C级"}

print("=" * 60)
print(f"  样本数: {m}, 特征数: {n}, 类别数: {K}")
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

    # 只在训练集上计算标准化参数
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std  = X_train_raw.std(axis=0, keepdims=True)

    X_train_scaled = (X_train_raw - X_mean) / X_std
    X_val_scaled   = (X_val_raw - X_mean) / X_std
    X_test_scaled  = (X_test_raw - X_mean) / X_std

    X = np.hstack([np.ones((m_train, 1)), X_train_scaled])
    X_val_full = np.hstack([np.ones((m_val, 1)), X_val_scaled])
    X_test_full = np.hstack([np.ones((m_test, 1)), X_test_scaled])

    print(f"X_mean: {X_mean}")
    print(f"X_std:  {X_std}")
else:
    X_mean = X_raw.mean(axis=0, keepdims=True)
    X_std  = X_raw.std(axis=0, keepdims=True)
    X_scaled = (X_raw - X_mean) / X_std
    X = np.hstack([np.ones((m, 1)), X_scaled])

    print(f"\nX_mean: {X_mean}")
    print(f"X_std:  {X_std}")

# ========== 3. One-hot 编码 ==========
def one_hot(y_input, K):
    """将类别标签 (m, 1) 转为 one-hot 矩阵 (m, K)"""
    m_samples = len(y_input)
    Y_out = np.zeros((m_samples, K))
    for i in range(m_samples):
        Y_out[i, int(y_input[i, 0])] = 1
    return Y_out

if DO_SPLIT:
    Y = one_hot(y_train, K)                          # (m_train, K)
    Y_val = one_hot(y_val, K)                        # (m_val, K)
else:
    Y = one_hot(y, K)                                # (m, K)

# ========== 4. 网络结构 + 随机初始化 ==========
input_size  = n                                    # 输入特征数（不含偏置）
hidden_size = {HIDDEN_SIZE}                        # 隐藏层神经元数
output_size = K                                    # 多分类：K 个输出神经元

total_params = (input_size + 1) * hidden_size + (hidden_size + 1) * output_size
print(f"\n网络结构: {input_size} → {hidden_size} → {output_size} (Softmax)")
print(f"总参数量: {total_params}")

# Xavier 初始化
np.random.seed(42)
Theta1 = np.random.randn(hidden_size, input_size + 1) * np.sqrt(1.0 / (input_size + 1))
Theta2 = np.random.randn(output_size, hidden_size + 1) * np.sqrt(1.0 / (hidden_size + 1))

print(f"Theta1 形状: {Theta1.shape}")
print(f"Theta2 形状: {Theta2.shape}")

# ========== 5. 核心函数 ==========

def sigmoid(z):
    """Sigmoid 激活函数（隐藏层使用）"""
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """Sigmoid 导数 g'(z) = g(z)(1-g(z))"""
    g = sigmoid(z)
    return g * (1 - g)


def softmax(z):
    """
    Softmax 激活函数（输出层使用）。

    参数:
        z - 线性组合 (m, K)

    返回:
        H - 概率矩阵 (m, K)，每行和为 1

    数值稳定性技巧：
        减去每行最大值, 防止 exp 溢出:
        e^(z-max) / Σ e^(z-max) = e^z / Σ e^z
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X, Theta1, Theta2):
    """
    前向传播（输出层用 Softmax）。

    返回:
        a1, z2, a2, a2_bias, z3, a3 (=H, 概率矩阵)
    """
    m_samples = X.shape[0]

    a1 = X                                               # (m, n+1)
    z2 = a1 @ Theta1.T                                   # (m, hidden_size)
    a2 = sigmoid(z2)                                     # (m, hidden_size)
    a2_bias = np.hstack([np.ones((m_samples, 1)), a2])   # (m, hidden_size+1)
    z3 = a2_bias @ Theta2.T                              # (m, K)
    a3 = softmax(z3)                                     # (m, K) ← Softmax!

    return a1, z2, a2, a2_bias, z3, a3


def cost_function(X, Y, Theta1, Theta2, lam):
    """
    多类交叉熵代价 + L2 正则化。

    J = -(1/m) * ΣΣ Y_ik * log(H_ik)
        + (λ/(2m)) * [Σ(Theta1[:,1:])² + Σ(Theta2[:,1:])²]

    参数:
        Y - one-hot 标签矩阵 (m, K)
    """
    m_samples = len(Y)
    _, _, _, _, _, H = forward_propagation(X, Theta1, Theta2)

    # 多类交叉熵
    J = -(1 / m_samples) * np.sum(Y * np.log(H + 1e-10))

    # L2 正则化项
    reg = (lam / (2 * m_samples)) * (
        np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)
    )

    return J + reg


def backpropagation(X, Y, Theta1, Theta2, lam):
    """
    反向传播（多分类版）。

    算法步骤:
        1. 前向传播（输出层 Softmax）
        2. 输出层误差: δ³ = H - Y          (m, K)
        3. 隐藏层误差: δ² = (δ³ @ Theta2_no_bias) ⊙ g'(z²)
        4. 梯度累积
    """
    m_samples = len(Y)

    # 前向传播
    a1, z2, a2, a2_bias, z3, H = forward_propagation(X, Theta1, Theta2)

    # 输出层误差 δ³ = H - Y    (m, K)
    delta3 = H - Y

    # 隐藏层误差 δ² = (δ³ @ Theta2_no_bias) ⊙ g'(z²)    (m, hidden_size)
    Theta2_no_bias = Theta2[:, 1:]                       # 去掉偏置列
    delta2 = (delta3 @ Theta2_no_bias) * sigmoid_gradient(z2)

    # 累积梯度
    Theta2_grad = (1 / m_samples) * (delta3.T @ a2_bias)       # (K, hidden_size+1)
    Theta1_grad = (1 / m_samples) * (delta2.T @ a1)            # (hidden_size, n+1)

    # L2 正则化梯度（偏置 theta0 不参与惩罚）
    Theta2_grad[:, 1:] += (lam / m_samples) * Theta2[:, 1:]
    Theta1_grad[:, 1:] += (lam / m_samples) * Theta1[:, 1:]

    return Theta1_grad, Theta2_grad


def gradient_descent(X, Y, Theta1, Theta2, alpha, lam, iters):
    """批量梯度下降训练"""
    J_history = np.zeros(iters)

    for i in range(iters):
        Theta1_grad, Theta2_grad = backpropagation(X, Y, Theta1, Theta2, lam)
        Theta1 -= alpha * Theta1_grad
        Theta2 -= alpha * Theta2_grad
        J_history[i] = cost_function(X, Y, Theta1, Theta2, lam)

    return Theta1, Theta2, J_history


# ========== 6. 梯度检验（调试用）==========
ENABLE_GRADIENT_CHECK = {ENABLE_GC}

if ENABLE_GRADIENT_CHECK:
    print("\n" + "=" * 60)
    print("  梯度检验 (Gradient Checking)")
    print("=" * 60)

    def gradient_check(X, Y, Theta1, Theta2, lam, epsilon=1e-4):
        """数值梯度 vs 解析梯度比较"""
        params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
        Theta1_grad_bp, Theta2_grad_bp = backpropagation(X, Y, Theta1, Theta2, lam)
        grad_bp = np.concatenate([Theta1_grad_bp.ravel(), Theta2_grad_bp.ravel()])

        num_grad = np.zeros_like(params)
        total = len(params)

        t_start = time.time()
        for i in range(total):
            params_plus = params.copy(); params_plus[i] += epsilon
            params_minus = params.copy(); params_minus[i] -= epsilon
            T1_p = params_plus[:Theta1.size].reshape(Theta1.shape)
            T2_p = params_plus[Theta1.size:].reshape(Theta2.shape)
            T1_m = params_minus[:Theta1.size].reshape(Theta1.shape)
            T2_m = params_minus[Theta1.size:].reshape(Theta2.shape)
            J_plus  = cost_function(X, Y, T1_p, T2_p, lam)
            J_minus = cost_function(X, Y, T1_m, T2_m, lam)
            num_grad[i] = (J_plus - J_minus) / (2 * epsilon)

            if (i + 1) % max(1, total // 10) == 0:
                print(f"  进度: {i+1}/{total} 参数已检验...")

        t_elapsed = time.time() - t_start
        print(f"  检验完成，耗时 {t_elapsed:.1f} 秒")

        diff = np.linalg.norm(num_grad - grad_bp) / np.linalg.norm(num_grad + grad_bp)
        return diff

    gc_diff = gradient_check(X, Y, Theta1, Theta2, {LAMBDA})
    print(f"\n  梯度检验差异: {gc_diff:.2e}")
    if gc_diff < 1e-7:
        print("  ✅ 反向传播梯度计算正确！")
    elif gc_diff < 1e-4:
        print("  ⚠️  差异略大，建议检查学习率后继续。")
    else:
        print("  ❌ 反向传播可能存在问题！请检查代码。")
    print("=" * 60)


# ========== 7. 训练 ==========
alpha = {ALPHA}
lam   = {LAMBDA}
iters = {ITERS}

print(f"\n训练参数: alpha={alpha}, lambda={lam}, iters={iters}")
print("训练中...")

Theta1, Theta2, J_history = gradient_descent(X, Y, Theta1, Theta2, alpha, lam, iters)

print(f"初始代价: {J_history[0]:.6f}")
print(f"最终代价: {J_history[-1]:.6f}")

# ========== 8. 模型评估 ==========
_, _, _, _, _, H_train = forward_propagation(X, Theta1, Theta2)
y_pred_train = np.argmax(H_train, axis=1).reshape(-1, 1)

if DO_SPLIT:
    y_ref = y_train
    n_eval = m_train
    accuracy = np.mean(y_pred_train == y_ref) * 100
    print(f"\n训练集准确率 ({m_train} 条): {accuracy:.1f}%")

    # 验证集评估
    _, _, _, _, _, H_val = forward_propagation(X_val_full, Theta1, Theta2)
    y_pred_val = np.argmax(H_val, axis=1).reshape(-1, 1)
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
    probs_str = ", ".join([f"{y_classes[c]}: {H_train[i,c]:.1%}" for c in range(K)])
    print(f"  样本{i+1}: x=({', '.join([f'{xi[j]:.2f}' for j in range(n)])}), "
          f"真实={y_classes[yi]}, "
          f"预测={y_classes[y_pred_train[i,0]]} | [{probs_str}]")

# 每类准确率
print(f"\n各类别准确率:")
for c in range(K):
    mask = (y_ref.flatten() == c)
    if mask.any():
        class_acc = np.mean(y_pred_train[mask].flatten() == c) * 100
        print(f"  {y_classes[c]}: {class_acc:.1f}% ({mask.sum()} 个样本)")

if DO_SPLIT:
    print(f"\n验证集预测详情:")
    for i in range(m_val):
        xi = X_val_raw[i]
        yi = int(y_val[i, 0])
        probs_str = ", ".join([f"{y_classes[c]}: {H_val[i,c]:.1%}" for c in range(K)])
        print(f"  验证样本{i+1}: x=({', '.join([f'{xi[j]:.2f}' for j in range(n)])}), "
              f"真实={y_classes[yi]}, "
              f"预测={y_classes[y_pred_val[i,0]]} | [{probs_str}]")

    print(f"\n验证集各类别准确率:")
    for c in range(K):
        mask = (y_val.flatten() == c)
        if mask.any():
            class_acc = np.mean(y_pred_val[mask].flatten() == c) * 100
            print(f"  {y_classes[c]}: {class_acc:.1f}% ({mask.sum()} 个样本)")

# ========== 9. 预测新样本 ==========
new_x = np.array([{PREDICT_X}])
new_x_scaled = (new_x - X_mean) / X_std
X_new = np.hstack([np.ones((1, 1)), new_x_scaled])
_, _, _, _, _, H_new = forward_propagation(X_new, Theta1, Theta2)
prob_new = H_new[0]                    # (K,) 各类别概率
pred_class = np.argmax(prob_new)       # 概率最大的类别

print(f"\n{'='*60}")
print(f"  新样本预测")
print(f"  x = ({', '.join([f'{new_x[0,j]}' for j in range(n)])})")
for c in range(K):
    print(f"  P({y_classes[c]}) = {prob_new[c]:.4f} ({prob_new[c]*100:.1f}%)")
print(f"  预测结果: {y_classes[pred_class]} (argmax)")
print(f"{'='*60}")

# ========== 10. 可视化 ==========
# 根据特征数选择不同的可视化方案
if n >= 2:
    # ---- 2+ 特征：K 色决策区域 + 收敛曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- 左图：决策区域（用前 2 个特征画 meshgrid）-----
    ax = axes[0]

    x1_min, x1_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    x2_min, x2_max = X_raw[:, 1].min() - 0.05, X_raw[:, 1].max() + 0.05
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300)
    )

    # 对每个网格点做前向传播 → 决策类别
    grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    # 如果有超过 2 个特征，其余填充为均值
    if n > 2:
        extra_cols = np.tile(X_raw[:, 2:].mean(axis=0), (grid_points.shape[0], 1))
        grid_points = np.hstack([grid_points, extra_cols])
    grid_scaled = (grid_points - X_mean) / X_std
    X_grid = np.hstack([np.ones((grid_points.shape[0], 1)), grid_scaled])
    _, _, _, _, _, H_grid = forward_propagation(X_grid, Theta1, Theta2)
    y_grid = np.argmax(H_grid, axis=1).reshape(x1_grid.shape)

    # K 色决策区域
    import matplotlib.colors as mcolors
    class_colors = {Y_CLASS_COLORS}   # 如 ['#e74c3c', '#3498db', '#2ecc71']
    cmap = mcolors.ListedColormap(class_colors[:K])
    ax.contourf(x1_grid, x2_grid, y_grid, levels=np.arange(-0.5, K + 0.5),
                alpha=0.3, cmap=cmap, zorder=0)

    # 训练数据散点
    if DO_SPLIT:
        plot_X_mc = X_train_raw
        plot_y_mc = y_train
    else:
        plot_X_mc = X_raw
        plot_y_mc = y

    for c in range(K):
        mask = (plot_y_mc.flatten() == c)
        if mask.any():
            ax.scatter(plot_X_mc[mask, 0], plot_X_mc[mask, 1], c=class_colors[c],
                       marker='o', s=80, label=y_classes[c], zorder=5,
                       edgecolors='black', linewidth=0.5)

    # 验证集数据点（如果切分了，用方形标记）
    if DO_SPLIT:
        for c in range(K):
            mask = (y_val.flatten() == c)
            if mask.any():
                ax.scatter(X_val_raw[mask, 0], X_val_raw[mask, 1], c=class_colors[c],
                           marker='s', s=60, label=f'{y_classes[c]} (val)', zorder=5,
                           edgecolors='black', linewidth=0.8, alpha=0.6)

    # 新预测点
    ax.scatter([new_x[0, 0]], [new_x[0, 1]], c=class_colors[pred_class],
               marker='*', s=250, zorder=10, edgecolors='black', linewidth=0.5,
               label=f'New → {y_classes[pred_class]}')

    train_label_mc = f'训练集({m_train}条)' if DO_SPLIT else f'{m}条全量数据'
    ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
    ax.set_ylabel(f'{x_names[1]} ({x_units[1]})', fontsize=11)
    ax.set_title(f'(A) Decision Regions [{train_label_mc}]\nNN [{input_size}-{hidden_size}-{K}], '
                 f'lambda={lam}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    # ----- 右图：收敛曲线 -----
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

else:
    # ---- 1 特征：K 条概率曲线 + 收敛曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- 左图：K 条 Softmax 概率曲线 -----
    ax = axes[0]

    x_line = np.linspace(X_raw.min() * 0.9, X_raw.max() * 1.1, 200).reshape(-1, 1)
    x_line_scaled = (x_line - X_mean) / X_std
    X_line = np.hstack([np.ones((200, 1)), x_line_scaled])
    _, _, _, _, _, H_line = forward_propagation(X_line, Theta1, Theta2)

    class_colors = {Y_CLASS_COLORS}
    for c in range(K):
        ax.plot(x_line, H_line[:, c], linewidth=2,
                color=class_colors[c % len(class_colors)],
                label=f'P({y_classes[c]})')

    # 训练数据点
    if DO_SPLIT:
        plot_X_mc1d = X_train_raw
        plot_y_mc1d = y_train
    else:
        plot_X_mc1d = X_raw
        plot_y_mc1d = y

    for c in range(K):
        mask = (plot_y_mc1d.flatten() == c)
        if mask.any():
            ax.scatter(plot_X_mc1d[mask], np.zeros_like(plot_X_mc1d[mask]) + c/(K-1) if K > 1 else plot_X_mc1d[mask]*0,
                       c=class_colors[c % len(class_colors)], marker='o', s=80,
                       label=f'{y_classes[c]} (train)', zorder=5,
                       edgecolors='black', linewidth=0.5)

    # 验证集数据点（如果切分了，用方形标记）
    if DO_SPLIT:
        for c in range(K):
            mask = (y_val.flatten() == c)
            if mask.any():
                ax.scatter(X_val_raw[mask], np.zeros_like(X_val_raw[mask]) + c/(K-1) if K > 1 else X_val_raw[mask]*0,
                           c=class_colors[c % len(class_colors)], marker='s', s=60,
                           label=f'{y_classes[c]} (val)', zorder=5,
                           edgecolors='black', linewidth=0.8, alpha=0.6)

    ax.scatter([new_x[0, 0]], [prob_new[pred_class]],
               c=class_colors[pred_class % len(class_colors)], marker='*',
               s=250, zorder=10, edgecolors='black', linewidth=0.5,
               label=f'New → {y_classes[pred_class]}')

    train_label_mc1d = f'训练集({m_train}条)' if DO_SPLIT else f'{m}条全量数据'
    ax.set_xlabel(f'{x_names[0]} ({x_units[0]})', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title(f'(A) Softmax Probability Curves [{train_label_mc1d}]\nNN [{input_size}-{hidden_size}-{K}]',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ----- 右图：收敛曲线 -----
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
| `{X_DATA}` | 特征矩阵 | `[3, 0.20], [4, 0.25], ...` |
| `{Y_DATA}` | 标签（0, 1, 2, ...）| `0, 0, 1, 1, 2, 2, 2` |
| `{K}` | 类别数量 | `3` |
| `{X_NAMES}` | 特征名列表 | `["保温厚度", "窗墙比"]` |
| `{X_UNITS}` | 特征单位列表 | `["cm", ""]` |
| `{Y_CLASSES}` | 类别映射字典 | `{0: "A级", 1: "B级", 2: "C级"}` |
| `{Y_CLASS_COLORS}` | 类别颜色列表（K 个）| `['#e74c3c', '#3498db', '#2ecc71']` |
| `{DO_SPLIT}` | 是否切分数据（True/False）| `True` |
| `{TRAIN_IDX}` | 训练集索引列表 | `[0, 1, 3, 5, 6, 8]` |
| `{VAL_IDX}` | 验证集索引列表 | `[2, 7, 9]` |
| `{TEST_IDX}` | 测试集索引列表 | `[4, 10, 11]` |
| `{HIDDEN_SIZE}` | 隐藏层神经元数 | `6` |
| `{ALPHA}` | 学习率 | `0.05` |
| `{LAMBDA}` | L2 正则化强度 | `0.01` |
| `{ITERS}` | 迭代次数 | `5000` |
| `{ENABLE_GC}` | 梯度检验开关 | `True` 或 `False` |
| `{PREDICT_X}` | 新样本特征值 | `7, 0.33` |
| `{FILENAME_BASE}` | 时间戳文件名 | `neural_network_20260730_160000` |

---

## 关键注意

### Softmax 数值稳定性
`softmax` 中必须先减去每行最大值：`exp(z - max(z, axis=1))`。否则当 `z` 中某个值很大时（如 z=100），`e^100` 会溢出 float64 范围。

### One-hot 编码
`one_hot(y, K)` 假设 y 的值是 `0, 1, ..., K-1`。如果用户提供的标签是字符串（如 "A级"），需要先在交互流程中转换为 0, 1, 2。

### 代价函数
多类交叉熵：`J = -(1/m) ΣΣ Y_ik * log(H_ik)`。其中 `Y_ik` 是 one-hot 标签，`H_ik` 是 Softmax 输出概率。对小数据集，代价函数初值应接近 `log(K)`（无信息先验）。

### 输出层误差
多分类的 `δ³ = H - Y` 形状为 `(m, K)`，与二分类的 `δ³ = a³ - y` 形状 `(m, 1)` 不同。核心公式相同，只是维度扩展。

### 预测
使用 `np.argmax(H, axis=1)` 取概率最大的类别，而非阈值比较。

### 可视化颜色
K 个类别需要 K 种颜色。默认提供 3 种（红、蓝、绿），超过 3 个类别时自动使用 matplotlib 的默认颜色循环。

### 编码
Windows GBK 环境下，print 中的 Unicode 替换为 ASCII。matplotlib label/title 不受影响。
