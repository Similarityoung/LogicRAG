# LogicRAG 优化指南

本文档记录了 LogicRAG 系统在算法实现和 Token 开销方面的优化方向与建议。

---

## 一、Token 开销优化

### 1.1 合并冗余的 LLM 调用

**问题：** `warm_up_analysis` 和 `dependency_aware_rag` 功能高度重叠，都在判断"能否回答"和"当前理解"。

**优化方式：** 将两个函数合并为 `unified_analysis`，通过参数控制行为：

```python
def unified_analysis(self, question: str, info_summary: str, 
                     dependencies: List[str] = None, current_idx: int = 0) -> Dict:
    """统一分析函数
    
    Args:
        dependencies: 若为 None，执行预热分析并提取依赖；否则执行依赖感知分析
    """
    if dependencies:
        dep_context = f"\nDependencies: {dependencies}\nCurrent: {dependencies[current_idx]}"
    else:
        dep_context = "\nExtract key dependencies needed to answer this question."
    
    prompt = f"""Question: {question}
Available Information: {info_summary}
{dep_context}

Analyze and respond in JSON:
- "can_answer": boolean
- "current_understanding": string
- "dependencies": list (only if extracting new dependencies)
"""
    # ... 调用 LLM
```

**预期收益：** 每个问题减少 0~N 次 LLM 调用

---

### 1.2 摘要长度控制与压缩

**问题：** `refine_summary_with_context` 没有长度限制，随着迭代轮次增加，摘要可能无限膨胀，导致后续每次 LLM 调用的输入 token 量持续增加。

**优化方式：** 在 prompt 中添加长度约束：

```python
def refine_summary_with_context(self, question: str, new_contexts: List[str], 
                                 current_summary: str = "",
                                 max_words: int = 300) -> str:
    prompt = f"""Question: {question}

Current summary: {current_summary}
New information: {chr(10).join(new_contexts)}

Create a refined summary that:
1. Integrates new relevant facts
2. Removes redundancies  
3. **MUST be under {max_words} words** - compress older/less relevant information
4. Prioritizes information that directly helps answer the question

Refined summary:"""
```

**预期收益：** 控制每次 LLM 调用的输入 token 在可预测范围内

---

### 1.3 依赖提取与排序合并

**问题：** 当前需要两次 LLM 调用：
1. `warm_up_analysis` 提取依赖列表
2. `_sort_dependencies` 生成依赖对进行拓扑排序

**优化方式：** 在预热分析时直接要求 LLM 输出已排序的依赖：

```python
# 修改 warm_up_analysis 的 prompt
prompt = f"""Question: {question}
Available Information: {info_summary}

Analyze and respond in JSON:
- "can_answer": boolean
- "dependencies": list of strings, **ordered by resolution priority** 
  (prerequisite dependencies first, e.g., if B is needed to answer A, put B before A)
- "current_understanding": string
"""
```

**预期收益：** 减少 1 次 LLM 调用，移除 `_sort_dependencies` 函数

---

### 1.4 Prompt 精简

**问题：** 多个函数的 prompt 包含冗余指令，增加不必要的 token 消耗。

**优化方式：** 精简各处 prompt，以 `generate_answer` 为例：

```python
# 优化前（约 150 tokens）
prompt = f"""You must give ONLY the direct answer in the most concise way possible. 
DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
...
Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """

# 优化后（约 40 tokens）
prompt = f"""Answer concisely with ONLY the direct answer (name/date/number/yes/no). No explanation.

Q: {question}
Info: {info_summary}
A:"""
```

**预期收益：** 每次调用减少 50-100 输入 tokens

---

### 1.5 启用结构化 JSON 输出

**问题：** LLM 可能输出格式错误的 JSON，需要 `fix_json_response` 修复。

**优化方式：** 使用 OpenAI API 的 `response_format` 参数：

```python
# 在 utils.py 的 get_response_with_retry 中添加可选参数
def get_response_with_retry(prompt: str, temperature: float = 0.0, 
                             json_mode: bool = False) -> str:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
        response_format={"type": "json_object"} if json_mode else None
    )
```

**预期收益：** 减少 JSON 解析错误和重试

---

## 二、算法实现优化

### 2.1 智能早停机制

**问题：** 即使新检索的信息与已有信息高度重复，系统仍会继续迭代直到 `can_answer=True` 或达到最大轮数。

**优化方式：** 添加信息增益检测，增益过低时提前终止：

```python
def _calculate_info_gain(self, old_summary: str, new_summary: str) -> float:
    """计算信息增益（基于词汇差异）"""
    old_tokens = set(old_summary.lower().split())
    new_tokens = set(new_summary.lower().split())
    new_info = new_tokens - old_tokens
    return len(new_info) / max(len(new_tokens), 1)

# 在 answer_question 的迭代循环中使用
old_summary = info_summary
info_summary = self.refine_summary_with_context(question, new_contexts, info_summary)

if self._calculate_info_gain(old_summary, info_summary) < 0.05:
    logger.info("信息增益过低，提前终止迭代")
    break
```

**预期收益：** 避免无效迭代，减少 2-3 次 LLM 调用

---

### 2.2 检索缓存优化

**问题：** `_retrieve_with_filter` 在去重结果不足时会多次调用检索，造成重复计算。

**优化方式：** 一次性检索更多候选：

```python
def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set,
                           buffer_multiplier: int = 3) -> list:
    """一次性检索足够多的候选以避免重复计算"""
    # 直接检索 top_k * buffer_multiplier 个结果
    candidate_count = min(self.top_k * buffer_multiplier, len(self.corpus))
    all_results = self._retrieve_top_n(query, candidate_count)
    
    # 过滤已检索过的
    unique_results = [c for c in all_results if c not in retrieved_chunks_set]
    return unique_results[:self.top_k]
```

**预期收益：** 减少重复的向量计算

---

### 2.3 大规模语料库向量检索加速

**问题：** 每次检索都是 O(n) 的全量余弦相似度计算，对于大规模语料库效率较低。

**优化方式：** 使用 FAISS 构建近似最近邻索引：

```python
import faiss
import numpy as np

class BaseRAG:
    def _build_faiss_index(self):
        """构建 FAISS 索引（语料库 > 10000 时推荐）"""
        embeddings_np = self.corpus_embeddings.numpy().astype('float32')
        faiss.normalize_L2(embeddings_np)  # 归一化以支持余弦相似度
        
        self.faiss_index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.faiss_index.add(embeddings_np)
    
    def _retrieve_with_faiss(self, query: str) -> List[str]:
        """使用 FAISS 进行快速检索"""
        query_emb = self.model.encode([query], convert_to_tensor=True).numpy().astype('float32')
        faiss.normalize_L2(query_emb)
        
        _, indices = self.faiss_index.search(query_emb, self.top_k)
        return [self.corpus[idx] for idx in indices[0]]
```

**预期收益：** 检索时间从 O(n) 降低到 O(log n)

---

### 2.4 依赖图剪枝

**问题：** 当前处理所有提取的依赖，即使某些依赖已被当前摘要覆盖。

**优化方式：** 使用语义相似度剪枝已覆盖的依赖：

```python
def _prune_covered_dependencies(self, dependencies: List[str], 
                                  info_summary: str,
                                  threshold: float = 0.85) -> List[str]:
    """移除已被摘要覆盖的依赖"""
    summary_emb = self.model.encode(info_summary, convert_to_tensor=True)
    pruned = []
    
    for dep in dependencies:
        dep_emb = self.model.encode(dep, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            summary_emb.unsqueeze(0), dep_emb.unsqueeze(0)
        ).item()
        
        if similarity < threshold:
            pruned.append(dep)
        else:
            logger.info(f"剪枝依赖 (相似度={similarity:.2f}): {dep}")
    
    return pruned
```

**预期收益：** 减少不必要的迭代轮次

---

## 三、代码质量优化

### 3.1 移除调试代码

**问题：** 代码中残留 `import pdb` 语句。

**优化方式：** 移除以下文件中的 pdb 导入：
- `src/models/logic_rag.py` (Line 17)
- `src/models/base_rag.py` (Line 14)  
- `src/evaluation/evaluation.py` (Line 2)

---

### 3.2 System Prompt 优化

**问题：** 当前使用通用的 "You are a helpful assistant"，未充分利用角色设定。

**优化方式：** 使用更具体的 System Prompt：

```python
SYSTEM_PROMPTS = {
    "analysis": "You are a precise information analyst. Always respond in valid JSON.",
    "summarization": "You are a concise summarizer. Extract only relevant facts.",
    "answering": "You are a direct answerer. Give only the essential answer."
}

def get_response_with_retry(prompt: str, role: str = "default", ...) -> str:
    system_content = SYSTEM_PROMPTS.get(role, "You are a helpful assistant.")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
```

---

## 四、优化优先级建议

| 优先级 | 优化项 | 预期收益 | 实现难度 |
|-------|-------|---------|---------|
| 🔴 高 | 1.2 摘要长度控制 | Token 消耗可控 | 低 |
| 🔴 高 | 1.3 依赖提取与排序合并 | 减少 1 次 LLM 调用 | 低 |
| 🔴 高 | 2.1 智能早停机制 | 减少 2-3 次 LLM 调用 | 中 |
| 🟡 中 | 1.1 合并冗余调用 | 减少 N 次 LLM 调用 | 中 |
| 🟡 中 | 1.4 Prompt 精简 | 每次减少 50-100 tokens | 低 |
| 🟢 低 | 2.2 检索缓存优化 | 减少向量计算 | 低 |
| 🟢 低 | 2.3 FAISS 加速 | 大语料库检索加速 | 中 |
| 🟢 低 | 2.4 依赖图剪枝 | 减少迭代轮次 | 中 |

---

## 五、实施建议

1. **第一阶段**：实现高优先级的三项优化（1.2、1.3、2.1）
2. **第二阶段**：进行 A/B 测试，确保准确率不受影响
3. **第三阶段**：根据测试结果，逐步实施中低优先级优化
