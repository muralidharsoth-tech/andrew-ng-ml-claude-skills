# 课程章节 ↔ 技能对照表

> 按吴恩达《机器学习》（Andrew Ng Machine Learning）课程章节排列，帮助你按上课进度找到对应技能。

---

## 完整对照

| 周 | 课程主题 | 核心知识点 | 对应技能 | 技能功能 |
|----|---------|-----------|---------|---------|
| 1-2 | 线性回归 | 梯度下降、代价函数、特征缩放、正规方程 | `linear-regression` | 单变量/多变量线性回归、正规方程、岭回归（L2 正则化）、多项式回归、学习曲线 |
| 3 | 逻辑回归 | Sigmoid、决策边界、正则化、多分类 | `logistic-regression` | 二分类逻辑回归、多项式特征扩展、One-vs-All 多分类、混淆矩阵 |
| 4-5 | 神经网络 | 前向传播、反向传播、梯度检验、随机初始化 | `neural-network` | 任意层数前馈网络、多分类输出、向量化反向传播、梯度检验 |
| 6 | 模型评估与选择 | 偏差/方差、学习曲线、正则化调参 | `neural-network-diagnosis` | 偏差方差诊断、学习曲线绘制、λ 网格搜索、早停法、K 折交叉验证 |
| 7 | SVM | 最大间隔分类器、核函数、SVM vs 逻辑回归 | `svm` | 线性 SVM、高斯核 SVM、参数 C 和 σ 调优 |
| 7 | 模型推荐 | 何时用哪种算法 | `model-recommendation` | 根据数据规模、特征类型、问题类型自动推荐最佳模型 |
| 9 | 异常检测 | 高斯分布、异常检测 vs 监督学习、特征选择 | `anomaly-detection` | 高斯分布建模、多元高斯分布、特征变换（对数/幂次）、阈值 ε 选择 |
| 9 | 推荐系统 | 协同过滤、内容推荐、均值归一化 | `recommender-system` | 协同过滤（同时学习 X 和 θ）、基于内容推荐（给定 X 学 θ）、均值归一化、相似物品检索 |
| 10 | 机器学习系统设计 | 误差分析、偏斜类、精确率/召回率 | `neural-network-diagnosis` | 误差分析（手动审查错分样本）、偏斜类指标（Precision/Recall/F₁）、上限分析思维 |
| 17 | 大规模机器学习 | 随机梯度下降、小批量梯度下降、MapReduce | 🚧 规划中 | — |
| 18 | OCR 流水线 | 滑动窗口、人工数据合成、上限分析 | 🚧 规划中 | — |

---

## 按学习路径推荐使用顺序

```
第 1-2 周  → linear-regression        入门：第一个 ML 模型
第 3 周    → logistic-regression      分类问题登场
第 4-5 周  → neural-network           深度学习初体验
第 6 周    → neural-network-diagnosis 学会"诊断"模型的好坏
第 7 周    → svm                      另一种强大的分类器
第 7 周    → model-recommendation     综合：面对新问题该选哪个算法？
第 9 周    → anomaly-detection        无监督学习：找异常
第 9 周    → recommender-system       无监督学习：做推荐
第 10 周   → neural-network-diagnosis 进阶诊断：偏斜类、误差分析
```

---

## 技能间的依赖关系

```
linear-regression ──→ logistic-regression ──→ neural-network
         │                    │                      │
         │                    │                      │
         └────────────────────┴──────────────────────┘
                             │
                             ▼
                   neural-network-diagnosis
                             │
                             ▼
                        svm ◄─── model-recommendation
                             │
                             ▼
                   anomaly-detection
                             │
                             ▼
                   recommender-system
```

- `linear-regression` 是基础——所有后续技能都依赖回归思维
- `neural-network-diagnosis` 是可复用诊断层——它的技巧（学习曲线、交叉验证）适用于几乎所有技能
- `model-recommendation` 是综合应用——需要先理解各个算法的适用场景

---

## 想快速找到适合你的技能？

| 你的问题 | 用什么技能 |
|---------|-----------|
| "我有一些数据，想预测一个数值" | → `linear-regression` |
| "我想判断是/否（二选一）" | → `logistic-regression` |
| "我想分类，但不止两类" | → `logistic-regression`（多分类）或 `neural-network` |
| "我的模型效果不好，不知道为什么" | → `neural-network-diagnosis` |
| "数据很复杂，线性分不开" | → `svm` 或 `neural-network` |
| "不知道该选什么算法" | → `model-recommendation` |
| "想找出不正常的数据点" | → `anomaly-detection` |
| "想给用户推荐东西" | → `recommender-system` |
