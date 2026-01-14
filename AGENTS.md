# AGENTS.md

LogicRAG 仓库的 AI 编程代理指南。

## 快速参考

### 构建与运行命令

```bash
# 安装依赖
pip install -r requirements.txt

# 在数据集上运行评估（默认：20个问题）
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json

# 运行所有问题（limit=0）
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --limit 0

# 运行单个问题
python run.py --model logic-rag --question "你的问题" --corpus dataset/hotpotqa_corpus.json

# 自定义参数
python run.py --model logic-rag --max-rounds 5 --top-k 3 --checkpoint-interval 10
```

### 无测试套件
本项目没有测试套件。验证更改的方式：
1. 运行 `python run.py --question "..." --corpus dataset/hotpotqa_corpus.json`
2. 使用 `lsp_diagnostics` 检查类型错误
3. 手动检查输出结果

### 无 Linter/Formatter 配置
项目中没有 `.flake8`、`pyproject.toml` 或格式化工具配置。请遵循现有代码风格。

---

## 项目架构

### 三层结构
```
src/
  models/
    base_rag.py      # BaseRAG：核心检索、嵌入向量、缓存
    logic_rag.py     # LogicRAG：依赖图、迭代检索
  evaluation/
    evaluation.py    # RAGEvaluator：批量评估、检查点
  utils/
    utils.py         # API调用、JSON修复、评估指标
  main.py            # CLI入口、参数解析
config/
  config.py          # API密钥、模型设置、速率限制
```

### 核心类
- `BaseRAG`：嵌入向量计算、余弦相似度检索、缓存管理
- `LogicRAG(BaseRAG)`：两阶段检索（预热 + 智能迭代）
- `RAGEvaluator`：批量评估，支持检查点保存/恢复

### 核心流程（LogicRAG）
1. **预热阶段**：初始检索 + 摘要生成
2. **依赖分析**：LLM 提取逻辑依赖关系
3. **拓扑排序**：基于 DFS 的依赖排序
4. **迭代检索**：按顺序处理每个依赖

---

## 代码风格指南

### Import 顺序
顺序：标准库 -> 第三方库 -> 本地模块。用空行分隔。

```python
import os
import json
import logging
from typing import List, Dict, Tuple, Any

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config.config import CACHE_DIR, RESULT_DIR
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry
```

### 类型提示
函数签名必须使用类型提示：

```python
def refine_summary_with_context(self, question: str, new_contexts: List[str], 
                                current_summary: str = "") -> str:
```

### 文档字符串
使用 Google 风格的文档字符串：

```python
def evaluate_question(self, question: str, gold_answer: str) -> Dict:
    """评估模型在单个问题上的表现。
    
    Args:
        question: 要评估的问题
        gold_answer: 标准答案
        
    Returns:
        包含评估结果的字典
    """
```

### 命名约定
- **类名**：PascalCase（`LogicRAG`、`RAGEvaluator`）
- **函数/方法**：snake_case（`answer_question`、`refine_summary_with_context`）
- **私有方法**：下划线前缀（`_sort_dependencies`、`_retrieve_with_filter`）
- **常量**：全大写下划线分隔（`CALLS_PER_MINUTE`、`DEFAULT_MODEL`）
- **变量**：snake_case（`info_summary`、`round_count`）

### 错误处理
LLM 调用和 I/O 操作需要 try-except 并记录日志：

```python
try:
    response = get_response_with_retry(prompt)
    result = fix_json_response(response)
    return result
except Exception as e:
    logger.error(f"{Fore.RED}dependency_aware_rag 错误: {e}{Style.RESET_ALL}")
    return {"can_answer": True, "current_understanding": f"错误: {str(e)}"}
```

### 日志记录
使用模块级 logger，配合 colorama 输出彩色日志：

```python
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

logger.info(f"{Fore.CYAN}正在处理: {question}{Style.RESET_ALL}")
logger.error(f"{Fore.RED}错误: {e}{Style.RESET_ALL}")
```

### LLM Prompt 格式
使用多行 f-string，结构清晰：

```python
prompt = f"""问题: {question}

可用信息:
{info_summary}

请将回复格式化为 JSON 对象，包含以下键:
- "can_answer": 布尔值
- "current_understanding": 字符串
"""
```

---

## 关键模式

### API 速率限制
所有 LLM 调用都通过 `get_response_with_retry()` 处理：
- 速率限制（config 中的 CALLS_PER_MINUTE）
- 失败时指数退避重试
- Token 消耗追踪

### 嵌入向量缓存
嵌入向量缓存在 `cache/embeddings_{corpus_size}.pt`，请勿删除。

### JSON 响应处理
LLM 返回的 JSON 可能格式错误，务必使用 `fix_json_response()`：

```python
response = get_response_with_retry(prompt)
result = fix_json_response(response)  # 返回 dict 或 None
if result is None:
    # 处理解析失败
```

### 检查点系统
评估过程会将检查点保存到 `result/checkpoints/`，恢复是自动的。

---

## 添加新的 RAG 模型

1. 创建 `src/models/your_model.py`
2. 继承 `BaseRAG`
3. 实现 `answer_question(question: str) -> Tuple[str, List[str], int]`
4. 在 `src/main.py` 中注册：
   ```python
   RAG_MODELS = {
       "logic-rag": LogicRAG,
       "your-model": YourModel,  # 在此添加
   }
   ```
5. 同时在 `src/evaluation/evaluation.py` 的 RAG_MODELS 字典中注册

---

## 环境配置

### 必需的环境变量
在项目根目录创建 `.env` 文件：
```
OPENAI_API_KEY=你的密钥
```

### 配置文件（config/config.py）
- `DEFAULT_MODEL`：LLM 模型（默认：gpt-4o-mini）
- `EMBEDDING_MODEL`：句子嵌入模型
- `CALLS_PER_MINUTE`：API 速率限制（默认：20）
- `MAX_RETRIES`：重试次数（默认：3）

---

## 数据格式

### 数据集 JSON
```json
[{"question": "...", "answer": "...", "id": "..."}]
```

### 语料库 JSON
```json
[{"title": "...", "text": "..."}]
```

---

## 常见陷阱

1. **空 API 响应**：检查 `get_response_with_retry()` 是否返回空字符串
2. **JSON 解析失败**：始终处理 `fix_json_response()` 返回 `None` 的情况
3. **GPU 内存**：`encode_sentences_batch()` 会在批次间清理 CUDA 缓存
4. **pdb 导入**：提交前删除 `import pdb` 和 `pdb.set_trace()`
