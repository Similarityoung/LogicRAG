# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LogicRAG 是一个基于查询逻辑依赖图引导的结构化检索增强生成(RAG)系统。该项目实现了在不预先构建知识图谱的情况下,通过构建查询逻辑依赖图来指导多步检索的自适应方法。

## 核心架构

### 三层架构
1. **基础层** ([`src/models/base_rag.py`](src/models/base_rag.py))
   - `BaseRAG` 类提供核心检索功能
   - 文档加载和预处理
   - 句子嵌入向量计算和缓存(使用 sentence-transformers)
   - 基于余弦相似度的 top-k 检索

2. **逻辑层** ([`src/models/logic_rag.py`](src/models/logic_rag.py))
   - `LogicRAG` 类继承自 `BaseRAG`
   - 实现多轮迭代检索和推理
   - 逻辑依赖分析和拓扑排序
   - 信息摘要生成和精炼

3. **应用层** ([`src/main.py`](src/main.py))
   - 命令行接口和参数解析
   - 数据集加载和评估协调
   - 单问题处理和批量评估模式

### 核心流程

LogicRAG 的检索过程分为两个阶段:

**阶段1: 热身检索** (Warm-up Retrieval)
- 对原始问题进行初步检索
- 生成初始信息摘要
- 判断是否可以进行简单事实检索(无需依赖分析)
- 如果需要深度推理,则分析依赖关系并进行拓扑排序

**阶段2: 代理迭代检索** (Agentic Iterative Retrieval)
- 按照拓扑排序的依赖顺序依次处理每个子问题
- 每轮检索后更新信息摘要
- 检查当前信息是否足够回答问题
- 达到最大轮次或所有依赖处理完毕后生成最终答案

### 关键组件

1. **依赖图构建** ([`logic_rag.py:249-311`](src/models/logic_rag.py#L249-L311))
   - `_sort_dependencies()`: 使用 LLM 识别依赖对关系
   - `_topological_sort()`: 基于 DFS 的拓扑排序算法

2. **信息摘要管理** ([`logic_rag.py:30-91`](src/models/logic_rag.py#L30-L91))
   - `refine_summary_with_context()`: 递进式更新信息摘要
   - 保持摘要简洁性同时保留关键信息

3. **检索过滤** ([`logic_rag.py:348-370`](src/models/logic_rag.py#L348-L370))
   - `_retrieve_with_filter()`: 过滤已检索的重复块
   - 支持动态扩展检索窗口

## 常用命令

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 设置 OpenAI API 密钥(在 .env 文件中)
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 运行评估
```bash
# 在数据集上运行评估(默认处理20个问题)
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json

# 处理所有问题(设置 limit=0)
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --limit 0

# 自定义参数
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --max-rounds 5 --top-k 3
```

### 单问题测试
```bash
# 测试单个问题
python run.py --model logic-rag --question "What is the capital of France?" --corpus dataset/hotpotqa_corpus.json
```

### 代码使用示例
```python
from src.models.logic_rag import LogicRAG

# 初始化 RAG 系统
rag = LogicRAG('dataset/hotpotqa_corpus.json')
rag.set_max_rounds(5)
rag.set_top_k(3)

# 提问
answer, contexts, rounds = rag.answer_question("Your question here")
print(f"Answer: {answer}")
print(f"Retrieved in {rounds} rounds")
```

## 配置说明

主要配置在 [`config/config.py`](config/config.py):

- **OPENAI_API_KEY**: 从 `.env` 文件加载
- **DEFAULT_MODEL**: 默认使用 `gpt-4o-mini`
- **EMBEDDING_MODEL**: 使用 `sentence-transformers/all-MiniLM-L6-v2`
- **CALLS_PER_MINUTE**: API 速率限制(默认20次/分钟)
- **CACHE_DIR**: 嵌入向量缓存目录(默认 `cache/`)
- **RESULT_DIR**: 结果保存目录(默认 `result/`)

## 数据集格式

项目支持的数据集格式(HotpotQA、2WikiMultihopQA、Musique):
```json
[
  {
    "question": "问题文本",
    "answer": "答案",
    "id": "问题ID"
  }
]
```

语料库格式:
```json
[
  {
    "title": "文档标题",
    "text": "文档内容"
  }
]
```

## 工具函数

主要工具在 [`src/utils/utils.py`](src/utils/utils.py):

- **`get_response_with_retry()`**: 带重试和速率限制的 OpenAI API 调用
- **`fix_json_response()`**: 修复不完整的 JSON 响应
- **`normalize_answer()`**: 答案标准化(用于评估)
- **`string_based_evaluation()`**: 基于字符串匹配的评估
- **`evaluate_with_llm()`**: 使用 LLM 进行答案评估
- **`save_results()`**: 保存评估结果

## 评估模块

[`src/evaluation/evaluation.py`](src/evaluation/evaluation.py) 包含 `RAGEvaluator` 类,提供:
- 批量问题评估
- 检查点保存(防止数据丢失)
- Top-k 准确率评估
- 支持 string-based 和 LLM-based 评估方法

## 重要提示

1. **缓存管理**: 嵌入向量会缓存在 `cache/` 目录,文件名格式为 `embeddings_{corpus_size}.pt`
2. **API 限制**: 默认每分钟20次调用,可在 `config.py` 中调整
3. **内存优化**: 批量编码时自动清理 GPU 内存
4. **依赖历史**: `LogicRAG.last_dependency_analysis` 存储最后一次依赖分析历史,用于调试和评估

## 扩展新的 RAG 模型

要添加新的 RAG 模型:

1. 在 [`src/models/`](src/models/) 创建新文件
2. 继承 `BaseRAG` 类
3. 实现 `answer_question()` 方法
4. 在 [`src/main.py`](src/main.py) 的 `RAG_MODELS` 字典中注册模型
