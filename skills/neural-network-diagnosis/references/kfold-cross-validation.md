# k-fold 交叉验证（K-Fold Cross-Validation）

> 📚 隶属于 `neural-network-diagnosis` 技能。在步骤 3（数据划分）中调用。
> 用途：当数据量小（m < 50）时，用 k-fold 替代单次 60/20/20 切分，获得更稳定的评估结果。

## 何时使用

- 样本数 m < 50（自动触发）
- m < 20 时 k 降为 3（折数太少不可靠）
- 用户坚持传统切分时也尊重用户选择

## 原理

```
全部数据 (m 条，打乱后)
    │
    ├── 第 1 轮: [折1 验证] [折2 折3 折4 折5 训练] → 得 J_val_1
    ├── 第 2 轮: [折1 折2 折3 折4 折5 训练] [折2 验证] → 得 J_val_2
    ├── ...
    └── 第 k 轮: [折1 ... 折k-1 训练] [折k 验证] → 得 J_val_k

最终 J_val = mean(J_val_1, ..., J_val_k)
```

**作用：** 每一条数据都当过验证集，15 栋小数据集也能得到稳定的评估结果。

## 完整代码模板

```python
import numpy as np

def k_fold_cross_validation(X, y, k, model_builder, alpha, lam, iters,
                            is_multiclass=True):
    """
    k 折交叉验证。

    参数:
        X, y          - 完整数据集（numpy 数组）
        k             - 折数（建议 5，m < 20 时用 3）
        model_builder - 无参函数，返回 (Theta1_init, Theta2_init)
        alpha, lam, iters - 训练参数
        is_multiclass - 是否多分类

    返回:
        avg_train_err - 平均训练错误率
        avg_val_err   - 平均验证错误率
        all_val_errs  - 每折的验证错误率列表（看波动）
    """
    m = len(y)
    fold_size = m // k

    # 随机打乱
    shuffle_idx = np.random.permutation(m)
    X_shuf = X[shuffle_idx]
    y_shuf = y[shuffle_idx]

    train_errors = []
    val_errors = []

    print(f"k-Fold 交叉验证 (k={k}, 每折约 {fold_size} 条):")
    print("-" * 50)

    for i in range(k):
        # 确定第 i 折的起止位置
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else m

        # 切分训练集和验证集
        X_val_fold = X_shuf[start:end]
        y_val_fold = y_shuf[start:end]
        X_train_fold = np.vstack([X_shuf[:start], X_shuf[end:]])
        y_train_fold = np.vstack([y_shuf[:start], y_shuf[end:]])

        # 训练模型（新鲜初始化）
        Theta1, Theta2 = model_builder()
        Theta1, Theta2, _ = gradient_descent(
            X_train_fold, y_train_fold, Theta1, Theta2, alpha, lam, iters
        )

        # 训练误差
        _, _, _, _, _, h_tr = forward_propagation(X_train_fold, Theta1, Theta2)
        if is_multiclass:
            yp_tr = np.argmax(h_tr, axis=1).reshape(-1, 1)
        else:
            yp_tr = (h_tr >= 0.5).astype(int)
        err_train = np.mean(yp_tr != y_train_fold)

        # 验证误差
        _, _, _, _, _, h_val = forward_propagation(X_val_fold, Theta1, Theta2)
        if is_multiclass:
            yp_val = np.argmax(h_val, axis=1).reshape(-1, 1)
        else:
            yp_val = (h_val >= 0.5).astype(int)
        err_val = np.mean(yp_val != y_val_fold)

        train_errors.append(err_train)
        val_errors.append(err_val)

        print(f"  折 {i+1}/{k}: J_train={err_train:.2%}, J_val={err_val:.2%}")

    avg_train_err = np.mean(train_errors)
    avg_val_err = np.mean(val_errors)
    std_val_err = np.std(val_errors)

    print("-" * 50)
    print(f"  平均: J_train={avg_train_err:.2%}, J_val={avg_val_err:.2%} ± {std_val_err:.2%}")

    return avg_train_err, avg_val_err, val_errors
```

## k 值选择

| 样本数 m | 建议 k | 原因 |
|----------|--------|------|
| m < 20 | 3 | 折数太多每折只剩几条数据 |
| 20 ≤ m < 50 | 5 | 标准选择，每折至少 4-10 条 |
| 50 ≤ m < 100 | 可使用 k-fold 但非必需 | 传统 60/20/20 也够用 |
| m ≥ 100 | 建议用 60/20/20 | k-fold 耗时且传统切分已够可靠 |

## 注意事项

1. **交叉验证只用于评估**，选好超参数后仍要用全部数据重新训练最终模型
2. **每折都要重新初始化**（`model_builder()`），参数不能跨折复用
3. **验证误差的波动（std）也很重要**——如果 std 很大，说明模型对数据划分敏感，数据可能太小
4. **k-fold 需要 k 次完整训练**，比单次训练慢 k 倍，估算并告知用户耗时
5. **随机打乱的种子固定**（`np.random.seed(42)`），保证每次运行结果可复现
