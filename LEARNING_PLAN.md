# LogicRAG 学习与复现计划

> 论文：[You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures](https://arxiv.org/abs/2508.06105)
> 
> 会议：AAAI 2026

---

## 📅 学习进度跟踪

| 阶段 | 预计时间 | 开始日期 | 完成日期 | 状态 |
|------|----------|----------|----------|------|
| 阶段 1: 基础知识 | 1-2 周 | | | ⬜ 未开始 |
| 阶段 2: 深入代码 | 1-2 周 | | | ⬜ 未开始 |
| 阶段 3: 实验复现 | 1-2 周 | | | ⬜ 未开始 |
| 阶段 4: 论文精读 | 1 周 | | | ⬜ 未开始 |
| 阶段 5: 进阶扩展 | 2-4 周 | | | ⬜ 未开始 |

**状态图例**: ⬜ 未开始 | 🟡 进行中 | ✅ 已完成

---

## 📚 阶段 1：基础知识（1-2 周）

### 1.1 RAG 基础

- [x] 阅读 RAG Survey Paper: https://arxiv.org/abs/2312.10997
- [x] 学习 Sentence-Transformers 文档
- [x] 理解向量检索与余弦相似度
- [x] 阅读 `BaseRAG.retrieve()` 实现

**学习资源**:
| 主题 | 资源 | 笔记 |
|------|------|------|
| RAG 概念 | [RAG Survey](https://arxiv.org/abs/2312.10997) | |
| 向量检索 | [Sentence-Transformers](https://www.sbert.net/) | |
| 余弦相似度 | `src/models/base_rag.py` L149-152 | |

### 1.2 图算法基础

- [x] 理解 DAG（有向无环图）概念
- [x] 学习拓扑排序算法
- [x] 理解 DFS 实现拓扑排序
- [x] 阅读 `_topological_sort()` 实现

**代码对应**:
```python
# src/models/logic_rag.py
_topological_sort()  # L314-346 - DFS 拓扑排序实现
_sort_dependencies() # L249-311 - 依赖排序入口
```

### 1.3 LLM Prompting

- [x] 学习结构化输出（JSON 格式）
- [x] 理解多轮对话设计
- [x] 分析 `warm_up_analysis()` 的 prompt 设计
- [x] 分析 `dependency_aware_rag()` 的 prompt 设计

---

## 🔬 阶段 2：深入代码（1-2 周）

### 2.1 核心流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    LogicRAG 核心流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入问题                                                   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │ Stage 1: 预热阶段 (Warm-up)             │                │
│  │  1. 初始检索 retrieve(question)         │                │
│  │  2. 生成摘要 refine_summary_with_context│                │
│  │  3. 分析依赖 warm_up_analysis()         │                │
│  └─────────────────────────────────────────┘                │
│      ↓                                                      │
│  可以直接回答？ ──Yes──> generate_answer() → 返回           │
│      ↓ No                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │依赖图构建                               │                │
│  │  1. 提取依赖 dependencies               │                │
│  │  2. LLM生成依赖对 dependency_pairs      │                │
│  │  3. 拓扑排序 _topological_sort()        │                │
│  └─────────────────────────────────────────┘                │
│      ↓                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │Stage 2: 迭代检索 (Agentic Retrieval)    │                │
│  │   for each dependency in sorted_order:  │                │
│  │     1. retrieve(dependency)             │                │
│  │     2. refine_summary_with_context()    │                │
│  │     3. dependency_aware_rag() 判断      │                │
│  │     4. 可回答则 break                   │                │
│  └─────────────────────────────────────────┘                │
│      ↓                                                      │
│  generate_answer() → 返回                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 代码阅读顺序

按以下顺序阅读代码：

| 顺序 | 文件 | 重点 | 完成 |
|------|------|------|------|
| 1 | `config/config.py` | 配置参数、API 设置 | ⬜ |
| 2 | `src/utils/utils.py` | `get_response_with_retry()`, `fix_json_response()` | ⬜ |
| 3 | `src/models/base_rag.py` | 嵌入计算、余弦相似度检索 | ⬜ |
| 4 | `src/models/logic_rag.py` | **核心算法实现** | ⬜ |
| 5 | `src/evaluation/evaluation.py` | 评估指标（EM, F1） | ⬜ |

### 2.3 关键方法清单

需要深入理解的 5 个核心方法：

| 方法 | 文件位置 | 功能 | 理解程度 |
|------|----------|------|----------|
| `warm_up_analysis()` | logic_rag.py L93-176 | 初始分析：能否直接回答 + 依赖提取 | ⬜ |
| `_sort_dependencies()` | logic_rag.py L249-311 | LLM 生成依赖对 + 拓扑排序 | ⬜ |
| `_topological_sort()` | logic_rag.py L314-346 | DFS 实现拓扑排序 | ⬜ |
| `dependency_aware_rag()` | logic_rag.py L178-224 | 迭代时判断是否可回答 | ⬜ |
| `refine_summary_with_context()` | logic_rag.py L30-91 | 滚动摘要（Rolling Memory） | ⬜ |

### 2.4 调试任务

- [ ] 在 `answer_question()` 入口打断点，跟踪完整流程
- [ ] 打印 `sorted_dependencies` 观察依赖排序结果
- [ ] 打印 `dependency_analysis_history` 观察推理过程
- [ ] 测试简单问题 vs 复杂问题的处理差异

---

## 🧪 阶段 3：实验复现（1-2 周）

### 3.1 环境搭建

```bash
# 安装依赖
cd /Users/similarityyoung/Documents/RAG/LogicRAG
pip install -r requirements.txt

# 配置 API Key
echo "OPENAI_API_KEY=your_key_here" > .env
```

- [ ] 安装依赖
- [ ] 配置 OpenAI API Key
- [ ] 验证环境可用

### 3.2 小规模测试

```bash
# 单问题测试
python run.py --model logic-rag \
  --question "Who is the mayor of the capital of France?" \
  --corpus dataset/hotpotqa_corpus.json \
  --max-rounds 5 --top-k 3

# 小规模评估（10 个问题）
python run.py --model logic-rag \
  --dataset dataset/hotpotqa.json \
  --corpus dataset/hotpotqa_corpus.json \
  --limit 10
```

- [ ] 运行单问题测试
- [ ] 运行小规模评估
- [ ] 记录运行结果

### 3.3 对比实验

| 实验 | 配置 | EM | F1 | Rounds |
|------|------|----|----|--------|
| Baseline (Naive RAG) | max_rounds=1 | | | |
| LogicRAG (default) | max_rounds=3, top_k=5 | | | |
| LogicRAG (more rounds) | max_rounds=5, top_k=3 | | | |
| LogicRAG (more context) | max_rounds=3, top_k=10 | | | |

- [ ] 实现 Naive RAG 基线
- [ ] 运行对比实验
- [ ] 分析实验结果

### 3.4 消融实验

- [ ] 去掉拓扑排序，观察效果变化
- [ ] 去掉 Rolling Memory，观察 token 消耗变化
- [ ] 调整 early stop 策略

---

## 📖 阶段 4：论文精读（1 周）

### 4.1 论文获取

- 论文 PDF: https://arxiv.org/pdf/2508.06105
- OpenReview: https://openreview.net/forum?id=ov1bwU35Mf

- [ ] 下载论文 PDF
- [ ] 打印/标注关键内容

### 4.2 重点章节对应

| 章节 | 主题 | 对应代码 | 阅读状态 |
|------|------|----------|----------|
| 3.1 | Logic Dependency Analysis | `warm_up_analysis()` | ⬜ |
| 3.2 | Graph Reasoning Linearization | `_sort_dependencies()`, `_topological_sort()` | ⬜ |
| 3.3 | Graph Pruning | 依赖剪枝逻辑 | ⬜ |
| 3.4 | Context Pruning | `refine_summary_with_context()` | ⬜ |
| 4 | Experiments | `evaluation.py` | ⬜ |

### 4.3 关键创新点笔记

1. **Query-time 构建**
   - 不预构建知识图谱，查询时动态生成依赖图
   - 优势：避免高昂的图构建成本，适应动态知识库
   - 笔记：

2. **DAG 线性化**
   - 拓扑排序保证逻辑一致性
   - 优势：按正确顺序解决子问题
   - 笔记：

3. **Rolling Memory**
   - 摘要精炼减少 context 长度
   - 优势：控制 token 消耗
   - 笔记：

4. **Early Stop**
   - 依赖可满足时提前终止
   - 优势：减少不必要的检索
   - 笔记：

### 4.4 与其他方法对比

| 方法 | 核心思路 | 优势 | 劣势 |
|------|----------|------|------|
| Naive RAG | 单次检索 | 简单快速 | 无法处理复杂问题 |
| GraphRAG | 预构建知识图谱 | 关系建模强 | 构建成本高 |
| Self-RAG | 自我反思检索 | 自适应 | 复杂度高 |
| LogicRAG | 查询时动态依赖图 | 灵活高效 | 依赖 LLM 质量 |

---

## 🚀 阶段 5：进阶扩展（2-4 周）

### 5.1 改进方向

| 方向 | 思路 | 难度 | 状态 |
|------|------|------|------|
| 更好的依赖提取 | 使用 CoT 或多步 prompting | ⭐⭐ | ⬜ |
| 并行检索 | 同级依赖可并行处理 | ⭐⭐⭐ | ⬜ |
| 更强的剪枝 | 相似度阈值过滤无关依赖 | ⭐⭐ | ⬜ |
| 多模态支持 | 图片/表格检索 | ⭐⭐⭐⭐ | ⬜ |
| 本地模型支持 | 替换 OpenAI 为本地 LLM | ⭐⭐ | ⬜ |

### 5.2 其他数据集验证

- [ ] MuSiQue 数据集
- [ ] 2WikiMultiHopQA 数据集
- [ ] HotpotQA 完整测试

### 5.3 发表目标

- [ ] 确定改进点
- [ ] 设计实验方案
- [ ] 撰写论文初稿
- [ ] 选择目标会议/期刊

---

## 📝 学习笔记区

### 问题与解答

| 问题 | 解答 | 日期 |
|------|------|------|
| | | |

### 灵感记录

| 想法 | 可行性 | 日期 |
|------|--------|------|
| | | |

### 遇到的坑

| 问题 | 解决方案 | 日期 |
|------|----------|------|
| | | |

---

## 📌 快速命令参考

```bash
# 单问题测试
python run.py --model logic-rag --question "YOUR_QUESTION" \
  --corpus dataset/hotpotqa_corpus.json

# 数据集评估
python run.py --model logic-rag \
  --dataset dataset/hotpotqa.json \
  --corpus dataset/hotpotqa_corpus.json \
  --limit 20

# 完整评估
python run.py --model logic-rag \
  --dataset dataset/hotpotqa.json \
  --corpus dataset/hotpotqa_corpus.json \
  --limit 0
```

---

*最后更新: 2026-01-14*
