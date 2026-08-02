# 早停训练（Early Stopping）

> 📚 隶属于 `neural-network-diagnosis` 技能。在步骤 10（复杂优化）中调用。
> 用途：训练过程中定期检查验证误差，在验证误差不再改善时自动停止，防止过拟合。

## 何时使用

- 高方差（过拟合）场景
- λ 网格搜索后效果仍有提升空间
- 想让模型充分学习的同时不过度训练

## 早停原理

```
Error ↑
      │
      │    ●验证误差 J_val ← 先降后升（过拟合开始）
      │   ╱ ╲
      │  ╱   ╲____
      │ ╱
      │╱
      │        ●训练误差 J_train ← 不断下降
      └──────────────────→ 迭代轮数
              ↑
           早停点 🎯 ← 取 J_val 最低时的参数
```

## 完整代码模板

```python
import numpy as np

def gradient_descent_with_early_stopping(X_train, y_train, X_val, y_val,
                                         Theta1, Theta2, alpha, lam,
                                         max_iters=10000,
                                         check_interval=100,
                                         patience=5):
    """
    带早停的梯度下降训练。

    参数:
        X_train, y_train - 训练集
        X_val, y_val     - 验证集（用于监控过拟合）
        Theta1, Theta2   - 初始参数（Xavier 初始化后）
        alpha, lam       - 学习率、正则化强度
        max_iters        - 最大迭代次数（设大一些，让早停来刹车）
        check_interval   - 每隔多少轮检查一次验证误差（默认 100）
        patience         - 验证误差连续多少次检查不降就停止（默认 5）

    返回:
        Theta1, Theta2  - 最佳参数（验证误差最小时的参数）
        J_history       - 训练代价历史 (max_iters,)
        best_iter       - 最佳参数对应的迭代轮数
        stopped_early   - 是否触发了早停（True/False）
    """
    m = len(y_train)
    J_history = np.zeros(max_iters)

    best_val_err = float('inf')
    best_Theta1 = Theta1.copy()
    best_Theta2 = Theta2.copy()
    best_iter = 0
    no_improve_count = 0
    stopped_early = False

    for i in range(max_iters):
        # 一次梯度下降
        Theta1_grad, Theta2_grad = backpropagation(
            X_train, y_train, Theta1, Theta2, lam
        )
        Theta1 -= alpha * Theta1_grad
        Theta2 -= alpha * Theta2_grad

        J_train = cost_function(X_train, y_train, Theta1, Theta2, lam)
        J_history[i] = J_train

        # 每隔 check_interval 轮检查验证误差
        if (i + 1) % check_interval == 0:
            J_val = cost_function(X_val, y_val, Theta1, Theta2, lam)

            if J_val < best_val_err:
                best_val_err = J_val
                best_Theta1 = Theta1.copy()
                best_Theta2 = Theta2.copy()
                best_iter = i + 1
                no_improve_count = 0
                print(f"  [{i+1:5d}] J_train={J_train:.6f}, J_val={J_val:.6f} ← 新最佳!")
            else:
                no_improve_count += 1
                print(f"  [{i+1:5d}] J_train={J_train:.6f}, J_val={J_val:.6f} (未改善 {no_improve_count}/{patience})")

            if no_improve_count >= patience:
                print(f"\n早停触发！第 {i+1} 轮（最佳在第 {best_iter} 轮）")
                stopped_early = True
                # 恢复到最佳参数
                Theta1 = best_Theta1
                Theta2 = best_Theta2
                break

    if not stopped_early:
        print(f"\n训练完成（达到 max_iters={max_iters}，未触发早停）")

    return Theta1, Theta2, J_history[:i+1], best_iter, stopped_early
```

## 参数调优指南

| 参数 | 默认值 | 何时调大 | 何时调小 |
|------|--------|---------|---------|
| `max_iters` | 10000 | 数据量 m 很小 | 数据量 m 很大 |
| `check_interval` | 100 | 训练特别慢 | 早停太频繁触发 |
| `patience` | 5 | 验证误差波动大（噪声大）| 想更快停止 |

## 注意事项

1. **早停配合减小 λ 效果更好**：如原模型 λ=1.0，用早停时可以降至 λ=0.1~0.3，让模型充分学习后由早停来防过拟合
2. **`best_Theta1`/`best_Theta2` 是深拷贝**（`.copy()`），不能是引用赋值
3. **早停依据的是代价 J 而非错误率**，因为代价更平滑、更早反映过拟合趋势
4. **patience 太小时可能过早停止**（验证误差的一次随机波动就触发停止），建议 patience ≥ 3
5. **早停后返回的 J_history 被截断**（`J_history[:i+1]`），只有实际训练轮数的历史
