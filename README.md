# 吴恩达《机器学习》Claude Code 技能集

[![技能数量](https://img.shields.io/badge/技能-8个-blue)](./skills/)
[![许可](https://img.shields.io/badge/许可-MIT-green)](./LICENSE)

**8 个 Claude Code 交互式技能**，覆盖吴恩达《机器学习》（Andrew Ng Machine Learning）课程的核心章节。

每个技能不是冰冷的代码模板——你用**自然语言描述你的数据和问题**，它引导你完成：智能选择算法 → 推荐参数 → 训练模型 → 可视化诊断 → 生成可独立运行的 Python 代码。

---

## 适用人群

- 🎓 正在学习吴恩达《机器学习》课程的同学
- 🔧 需要快速上手 ML 项目但不想从零写代码的工程师
- 📊 有数据、想尝试不同算法对比效果的初学者

**前提条件**：安装了 [Claude Code](https://claude.ai/code)，本地有 Python + NumPy + Matplotlib。

---

## 技能列表

| # | 技能 | 课程章节 | 一句话描述 |
|---|------|---------|-----------|
| 1 | `linear-regression` | 第 1-2 周 | 线性回归、正规方程、岭回归、多项式回归 |
| 2 | `logistic-regression` | 第 3 周 | 逻辑回归、决策边界可视化、多分类（One-vs-All） |
| 3 | `svm` | 第 7 周 | 支持向量机、高斯核 SVM、非线性分类 |
| 4 | `neural-network` | 第 4-5 周 | 前馈神经网络、反向传播、多分类输出 |
| 5 | `neural-network-diagnosis` | 第 6、10 周 | 偏差/方差诊断、学习曲线、交叉验证、误差分析、偏斜类处理 |
| 6 | `model-recommendation` | 第 7 周 | 根据数据特征自动推荐最合适的 ML 模型 |
| 7 | `anomaly-detection` | 第 9 周 | 高斯分布异常检测、多元高斯、特征变换 |
| 8 | `recommender-system` | 第 9 周 | 协同过滤、基于内容推荐、相似物品检索 |

> 📖 完整的课程章节对照表见 [COURSE_MAP.md](./COURSE_MAP.md)

---

## 安装

1. 找到你的 Claude Code 项目目录（或任意目录，Claude Code 会识别）

2. 将 `skills/` 下的文件夹复制到 `.claude/skills/`：

   ```bash
   # 复制全部技能
   cp -r skills/* 你的项目路径/.claude/skills/

   # 或者只复制你需要的
   cp -r skills/linear-regression 你的项目路径/.claude/skills/
   ```

3. 在 Claude Code 中输入对应关键词即可触发技能：

   | 输入关键词示例 | 触发技能 |
   |--------------|---------|
   | "帮我做线性回归" | `linear-regression` |
   | "用逻辑回归分类" | `logistic-regression` |
   | "诊断神经网络" | `neural-network-diagnosis` |
   | "检测异常数据" | `anomaly-detection` |
   | "搭建推荐系统" | `recommender-system` |

---

## 多平台支持

技能采用开放 Agent Skills 格式（SKILL.md），除 Claude Code 外，**DeepSeek Harness 等支持 Skills 规范的 agent 工具可直接使用，无需任何改动**。

### DeepSeek Harness 安装

```bash
# 用户级（所有项目可用）
cp -r skills/* ~/.dsh/skills/

# 或项目级（仅当前项目）
cp -r skills/* 你的项目路径/.dsh/skills/

```

安装后在对话中直接描述需求或输入 `/linear-regression` 等命令即可触发。

---

## 工作流程

每个技能遵循统一的交互流程：

```
你的数据/问题
    │
    ▼
① 智能识别 —— 自动判断数据格式（CSV？粘贴数组？口头描述？）
    │
    ▼
② 算法决策 —— 有物品特征？→ 内容推荐。没有？→ 协同过滤。
    │
    ▼
③ 参数推荐 —— 根据数据规模自动推荐 λ、α、迭代次数
    │
    ▼
④ 训练诊断 —— 实时绘制代价曲线，自动检测收敛，α 过大自动减半
    │
    ▼
⑤ 结果输出 —— 预测值 + Top-N 推荐 + 图表 + 可运行 .py 文件
```

---

## 技能目录结构

```
skills/
├── linear-regression/
│   ├── SKILL.md                # 主技能：交互流程 + 代码模板
│   └── references/             # 算法详细实现
│       ├── normal-equation.md
│       ├── ridge-regression.md
│       └── polynomial-regression.md
├── logistic-regression/
│   ├── SKILL.md
│   └── references/
│       ├── polynomial-logistic.md
│       └── multi-class-logistic.md
├── svm/
│   ├── SKILL.md
│   └── references/
│       └── gaussian-kernel-svm.md
├── neural-network/
│   ├── SKILL.md
│   └── references/
│       └── multi-class-nn.md
├── neural-network-diagnosis/
│   ├── SKILL.md
│   └── references/
│       ├── bias-variance-diagnosis.md
│       ├── learning-curve.md
│       ├── kfold-cross-validation.md
│       ├── error-analysis.md
│       ├── skewed-class.md
│       ├── early-stopping.md
│       └── lambda-grid-search.md
├── model-recommendation/
│   ├── SKILL.md
│   └── references/
│       └── model-comparison-table.md
├── anomaly-detection/
│   ├── SKILL.md
│   └── references/
│       ├── multivariate-gaussian.md
│       ├── feature-engineering.md
│       └── gaussian-transform.md
└── recommender-system/
    ├── SKILL.md
    └── references/
        ├── collaborative-filtering.md
        ├── content-based-filtering.md
        ├── mean-normalization.md
        └── similar-items.md
```

---

## 设计理念

- **教学优先**：技能交互用语面向初学者，每个步骤解释"为什么这样做"
- **安全兜底**：数据太少时警告、α 发散时自动回退、参数在边界时提示扩展搜索
- **与课程对齐**：术语、符号、算法流程严格遵循吴恩达课程原文
- **中文交流，代码英文**：所有用户交互用简体中文，代码标识符保持英文字符

---

## 待开发

| 课程章节 | 内容 | 状态 |
|---------|------|------|
| 第 17 周 | 大规模机器学习（SGD、MapReduce） | 🚧 规划中 |
| 第 18 周 | OCR 流水线（滑动窗口、上限分析） | 🚧 规划中 |

---

## 许可

MIT License © 2026 — 详见 [LICENSE](./LICENSE)
