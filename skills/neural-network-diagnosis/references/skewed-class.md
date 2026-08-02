# 偏斜类问题（Skewed Classes）

> 📚 隶属于 `neural-network-diagnosis` 技能。在步骤 2（审定标准）和步骤 4（计算误差）中调用。
> 用途：当类别分布严重不均时，用 F1 分数替代准确率作为评估指标。

## 何时触发

- 任一类别的样本占比 < **15%**（自动检测）
- 或用户在步骤 2 中明确提到"类别不均衡"

## 为什么准确率会骗人

如果 100 栋楼中 99 栋安全、1 栋有隐患，"全部判安全"就有 99% 准确率—但模型完全没用。

> ⚠️ 准确率 = (TP + TN) / (TP + TN + FP + FN)，当正负样本极度不均时，TN 主导了准确率。

## 混淆矩阵

| | 预测为正类（1）| 预测为负类（0）|
|---|:--:|:--:|
| **实际为正类（1）** | TP（真正例）✅ | FN（假负例）❌ 漏报 |
| **实际为负类（0）** | FP（假正例）❌ 误报 | TN（真负例）✅ |

## 三个关键指标

| 指标 | 公式 | 含义 | 建筑场景类比 |
|------|------|------|-------------|
| **Precision（查准率）** | TP / (TP + FP) | 你说有问题的楼里，多少真的有问题？| 不想虚报警情 |
| **Recall（查全率）** | TP / (TP + FN) | 真有问题的楼里，你找出了多少？| 不想漏掉隐患 |
| **F₁ Score** | 2·P·R / (P + R) | 平衡两者（P 和 R 的调和平均）| ✅ **偏斜类首选指标** |

## 完整代码模板

```python
import numpy as np

def compute_metrics(y_true, y_pred, positive_class=1):
    """
    计算 Precision, Recall, F1 Score。

    参数:
        y_true         - 真实标签 (m, 1) 或 (m,)
        y_pred         - 预测标签 (m, 1) 或 (m,)
        positive_class - 正类（少数类）的标签值（默认 1）

    返回:
        precision, recall, f1, confusion_matrix_dict
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    tn = np.sum((y_true != positive_class) & (y_pred != positive_class))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    confusion = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    return precision, recall, f1, confusion


def detect_skewed(y, threshold=0.15):
    """
    检测是否为偏斜类问题。

    返回:
        is_skewed (bool), minority_class (int), minority_ratio (float)
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for cls, cnt in zip(unique, counts):
        ratio = cnt / total
        if ratio < threshold:
            return True, int(cls), ratio
    return False, None, None
```

## 阈值调节

在二分类中，不一定要用 h=0.5 作为决策阈值：

```python
# h(x) >= 0.9 → 判正类: Precision 高（宁缺毋滥），Recall 低（漏了很多）
# h(x) >= 0.5 → 判正类: 中等（默认）
# h(x) >= 0.1 → 判正类: Precision 低（很多误报），Recall 高（几乎不漏）
```

**选择逻辑：**
- 漏报代价高（如安全隐患）→ 降低阈值，提高 Recall
- 误报代价高（如虚警浪费人力）→ 提高阈值，提高 Precision
- 两者都重要 → 用默认 0.5 + F1 评估

## 多分类 + 偏斜类

多分类时，对每个类别分别计算 P/R/F1，然后取**宏平均**（macro-average）：

```python
def compute_multiclass_metrics(y_true, y_pred, K):
    """对 K 个类别分别计算 P/R/F1，取宏平均"""
    precisions, recalls, f1s = [], [], []
    for c in range(K):
        # 把类别 c 视为"正类"，其他为"负类"
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        p, r, f1, _ = compute_metrics(y_true_bin, y_pred_bin, positive_class=1)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)
```

## 注意事项

1. **偏斜类判定优先于量化标准选择**：如果检测到偏斜类，自动将"好"的标准从错误率切换为 F1
2. **F1 > 0.85 默认是"好"**（可在步骤 2 中由用户调整）
3. **多分类偏斜类用宏平均**，而非加权平均（加权平均会放大多数类的权重）
4. **不要只看一个指标**：P 高 R 低 = 太保守，R 高 P 低 = 太多误报，始终三个指标一起看
5. **阈值调节使用验证集**：画 P-R 曲线，根据业务需求在验证集上确定最佳阈值
