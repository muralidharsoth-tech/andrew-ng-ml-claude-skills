# λ 网格搜索（Lambda Grid Search）

> 📚 隶属于 `neural-network-diagnosis` 技能。在步骤 10（复杂优化）中调用。
> 用途：在候选 λ 列表中系统搜索使验证误差最小的正则化强度。

## 何时使用

- 高方差（过拟合）场景
- 不知道当前的 λ 是否是最优值
- 简单调 λ 后效果改善但仍不满足标准

## 完整代码模板

```python
import numpy as np

def lambda_grid_search(X_train, y_train, X_val, y_val,
                       model_builder, forward_fn, alpha, iters,
                       is_multiclass=True):
    """
    λ 网格搜索：在验证集上找最佳正则化强度。

    参数:
        X_train, y_train - 训练集
        X_val, y_val     - 验证集
        model_builder    - 无参函数，返回 (Theta1_init, Theta2_init)
        forward_fn       - 前向传播函数
        alpha, iters     - 训练参数（λ 将在候选列表中被覆盖）
        is_multiclass    - 是否多分类

    返回:
        best_lambda - 使验证错误率最小的 λ 值
        results     - 列表，每项为 (lam, err_train, err_val)
    """
    # 候选 λ 列表（对数尺度）
    lambda_candidates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]

    results = []
    best_lambda = None
    best_val_err = float('inf')

    print("λ 网格搜索:")
    print(f"{'λ':>8}  {'训练错误率':>12}  {'验证错误率':>12}  {'判定':>12}")
    print("-" * 52)

    for lam in lambda_candidates:
        # 新鲜初始化 + 训练
        Theta1, Theta2 = model_builder()
        Theta1, Theta2, _ = gradient_descent(
            X_train, y_train, Theta1, Theta2, alpha, lam, iters
        )

        # 训练误差
        _, _, _, _, _, h_tr = forward_fn(X_train, Theta1, Theta2)
        if is_multiclass:
            yp_tr = np.argmax(h_tr, axis=1).reshape(-1, 1)
        else:
            yp_tr = (h_tr >= 0.5).astype(int)
        err_train = np.mean(yp_tr != y_train)

        # 验证误差
        _, _, _, _, _, h_v = forward_fn(X_val, Theta1, Theta2)
        if is_multiclass:
            yp_v = np.argmax(h_v, axis=1).reshape(-1, 1)
        else:
            yp_v = (h_v >= 0.5).astype(int)
        err_val = np.mean(yp_v != y_val)

        results.append((lam, err_train, err_val))

        mark = ""
        if err_val < best_val_err:
            best_val_err = err_val
            best_lambda = lam
            mark = "← 最佳!"

        print(f"  {lam:6.3f}  {err_train:12.2%}  {err_val:12.2%}  {mark:>12}")

    print(f"\n最佳 λ = {best_lambda}（验证错误率 = {best_val_err:.2%}）")
    return best_lambda, results
```

## λ 与偏差/方差的关系

```
Error ↑
      │  ●J_train
      │   ╲
      │    ╲  ●J_val
      │     ╲╱     ← λ 太小 → 高方差（过拟合）
      │     ╱╲          J_train 低，J_val 高，gap 大
      │    ╱  ╲    ← λ 太大 → 高偏差（欠拟合）
      │   ╱    ╲        J_train 和 J_val 都高，贴在一起
      └──────────────────────→ λ
       小 λ                  大 λ

🎯 目标：J_val 最低的那个 λ
```

## 注意事项

1. **候选 λ 用对数尺度**（0.001, 0.003, 0.01, 0.03, 0.1...），因为 λ 的最优值通常跨多个数量级
2. **λ=0 必须包含**（关闭正则化的基线）
3. **每次尝试都要重新初始化模型**（`model_builder()`），否则上一次的 λ 会污染本次结果
4. **如果最优 λ 在列表两端**（如 0 或 10），考虑在那一端扩展候选值再搜一轮
5. **λ 网格搜索需要 10×N 次训练**（10 个候选值 × 每次 iters 轮），预估耗时并提醒用户
