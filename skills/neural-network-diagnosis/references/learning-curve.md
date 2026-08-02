# 学习曲线（Learning Curve）

> 📚 隶属于 `neural-network-diagnosis` 技能。在步骤 5 中调用。
> 用途：用不同大小的训练子集训练模型，观察 J_train 和 J_val 随数据量变化的趋势。

## 何时使用

- 诊断模型效果时（必画）
- 判断模型是欠拟合还是过拟合
- 评估"加更多数据"是否对改善效果有效

## 完整代码模板

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_learning_curve(X_train, y_train, X_val, y_val,
                        model_builder, alpha, lam, iters,
                        steps=8, is_multiclass=True):
    """
    画学习曲线：用不同大小的训练子集训练，记录训练误差和验证误差。

    参数:
        X_train, y_train - 训练集（已标准化、含偏置列）
        X_val, y_val     - 验证集（已标准化、含偏置列）
        model_builder    - 无参函数，返回 (Theta1_init, Theta2_init) 新初始化参数
        alpha, lam, iters - 与原模型一致的训练参数
        steps            - 取几个不同的训练集大小（默认 8）
        is_multiclass    - 是否多分类（True: argmax, False: ≥0.5）

    返回:
        sizes          - 训练集大小列表 (list of int)
        train_errors   - 训练错误率列表 (list of float)
        val_errors     - 验证错误率列表 (list of float)
    """
    m = len(y_train)
    train_errors = []
    val_errors   = []

    # 确保最小训练子集至少有 3 条数据
    min_size = max(3, m // (steps * 2))
    sizes = np.linspace(min_size, m, min(steps, m // min_size)).astype(int)
    sizes = np.unique(sizes)  # 去重

    print(f"学习曲线 (m 从 {sizes[0]} 到 {m}):")
    print(f"{'训练集大小':>10}  {'J_train':>10}  {'J_val':>10}")
    print("-" * 34)

    for size in sizes:
        # 用前 size 条数据训练
        X_sub = X_train[:size]
        y_sub = y_train[:size]

        Theta1, Theta2 = model_builder()
        Theta1, Theta2, _ = gradient_descent(
            X_sub, y_sub, Theta1, Theta2, alpha, lam, iters
        )

        # 训练误差（在训练子集上评估）
        _, _, _, _, _, h_sub = forward_propagation(X_sub, Theta1, Theta2)
        if is_multiclass:
            y_pred_sub = np.argmax(h_sub, axis=1).reshape(-1, 1)
        else:
            y_pred_sub = (h_sub >= 0.5).astype(int)
        err_train = np.mean(y_pred_sub != y_sub)
        train_errors.append(err_train)

        # 验证误差（在完整验证集上评估）
        _, _, _, _, _, h_val = forward_propagation(X_val, Theta1, Theta2)
        if is_multiclass:
            y_pred_val = np.argmax(h_val, axis=1).reshape(-1, 1)
        else:
            y_pred_val = (h_val >= 0.5).astype(int)
        err_val = np.mean(y_pred_val != y_val)
        val_errors.append(err_val)

        print(f"  {size:6d}    {err_train:8.2%}    {err_val:8.2%}")

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(sizes, [e * 100 for e in train_errors], 'b-o', linewidth=2,
            markersize=6, label='Training Error (J_train)')
    ax.plot(sizes, [e * 100 for e in val_errors], 'r-s', linewidth=2,
            markersize=6, label='Validation Error (J_val)')

    ax.set_xlabel('Training Set Size (m)', fontsize=11)
    ax.set_ylabel('Error Rate (%)', fontsize=11)
    ax.set_title('Learning Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注最终值
    ax.annotate(f'J_train={train_errors[-1]:.1%}',
                xy=(sizes[-1], train_errors[-1] * 100),
                xytext=(sizes[-1] * 0.7, train_errors[-1] * 100 + 5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.0),
                fontsize=9, color='blue')
    ax.annotate(f'J_val={val_errors[-1]:.1%}',
                xy=(sizes[-1], val_errors[-1] * 100),
                xytext=(sizes[-1] * 0.7, val_errors[-1] * 100 - 10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
                fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('learning_curve_{TIMESTAMP}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n学习曲线图片已保存: learning_curve_{TIMESTAMP}.png")

    return sizes, train_errors, val_errors
```

## 学习曲线诊断速查

画完学习曲线后，对照以下四种形状诊断：

| 形状 | J_train | J_val | 两者关系 | 诊断 |
|------|:------:|:-----:|---------|------|
| 两高靠拢 | 高 | 高 ≈ J_train | 贴得很近 | **高偏差（欠拟合）** |
| 一低一高有 gap | 很低 | ≫ J_train | gap 随 m 增大缩小 | **高方差（过拟合）** |
| 两高且 J_val 更高 | 高 | 更高 | 有 gap | **两者都有** |
| 两低靠拢 | 低 | 低 ≈ J_train | 贴得很近 | ✅ **效果良好** |

## 注意事项

1. **每次用不同大小的子集都要重新初始化模型**（`model_builder()`），不能复用上次训练的参数
2. **验证集始终用完整的 X_val**（不缩小子集），保证评估基准一致
3. **m < steps×3 时自动减少 steps**（确保最小训练子集至少有 3 条数据）
4. **学习曲线需要多次完整训练**（steps 次 × 每次 iters 轮），小数据集较快，大数据集可能需要几分钟
5. **分类/回归的误差计算不同**：分类用错误率（0/1 loss），回归用 MSE。此模板仅覆盖分类场景
6. **占位符 `{TIMESTAMP}` 在实际生成时替换为当前时间戳**，格式 `YYYYMMDD_HHMMSS`
