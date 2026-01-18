"""
LogicRAG 模块

实现基于逻辑依赖图的检索增强生成系统。
通过分析问题的逻辑依赖关系，将复杂问题分解为可顺序解决的子问题，
实现结构化的多轮迭代检索。

主要特性：
- 逻辑依赖分析：将复杂问题分解为依赖子图
- 拓扑排序：确保按正确顺序解决依赖关系
- 滚动摘要：高效管理上下文窗口
- 可解释性：提供清晰的推理路径
"""

import json
import logging
import pdb
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# 初始化 colorama，支持终端彩色输出
init()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicRAG(BaseRAG):
    """
    基于逻辑依赖图的 RAG 系统

    继承自 BaseRAG，添加了逻辑依赖分析和多轮迭代检索能力。

    核心流程：
    1. 预热阶段：初始检索 + 判断是否需要深度推理
    2. 依赖分析：提取问题的逻辑依赖关系
    3. 拓扑排序：基于 DFS 的依赖排序
    4. 迭代检索：按顺序解决每个依赖子问题

    属性:
        max_rounds: 最大迭代检索轮数
        filter_repeats: 是否过滤已检索过的文档块
        last_dependency_analysis: 最后一次依赖分析的历史记录
    """

    def __init__(
        self,
        corpus_path: str = None,
        cache_dir: str = "./cache",
        filter_repeats: bool = False,
    ):
        """
        初始化 LogicRAG 系统

        Args:
            corpus_path: 语料库文件路径
            cache_dir: 缓存目录路径
            filter_repeats: 是否过滤重复的检索结果（跨轮次去重）
        """
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # 默认最大迭代轮数
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # 是否过滤跨轮次的重复文档块

    def set_max_rounds(self, max_rounds: int):
        """
        设置最大检索轮数

        Args:
            max_rounds: 最大迭代检索轮数
        """
        self.max_rounds = max_rounds

    def refine_summary_with_context(
        self, question: str, new_contexts: List[str], current_summary: str = ""
    ) -> str:
        """
        基于新检索的上下文生成或优化信息摘要

        该方法实现了滚动摘要机制，在每轮检索后整合新信息，
        保持摘要简洁的同时保留所有关键信息。

        Args:
            question: 原始问题
            new_contexts: 新检索到的上下文文档块列表
            current_summary: 当前的信息摘要（如果有的话）

        Returns:
            更新后的信息摘要字符串
        """
        try:
            context_text = "\n".join(new_contexts)

            if not current_summary:
                # 生成初始摘要
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                # 用新信息优化现有摘要
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""

            summary = get_response_with_retry(prompt)
            return summary

        except Exception as e:
            logger.error(f"{Fore.RED}生成/优化摘要时出错: {e}{Style.RESET_ALL}")
            # 发生错误时，将当前摘要与新上下文拼接作为后备方案
            if current_summary:
                return f"{current_summary}\n\n新信息:\n{context_text}"
            return context_text

    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        """
        预热分析：判断问题是否可以通过简单事实检索回答

        这是一个初步分析步骤，用于判断问题的复杂度：
        - 如果问题可以直接回答，则跳过依赖分析
        - 如果需要多步推理，则提取逻辑依赖关系

        Args:
            question: 原始问题
            info_summary: 当前信息摘要

        Returns:
            包含分析结果的字典:
            - can_answer: 是否可以直接回答
            - missing_info: 缺失的信息
            - subquery: 用于查找缺失信息的子查询
            - current_understanding: 当前理解
            - dependencies: 关键信息依赖列表
            - missing_reason: 信息缺失原因
        """
        try:
            prompt = f"""Question: {question}

Available Information:
{info_summary}

Based on the information provided, please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.
5. What are the key dependencies needed to answer this question?
6. Why is information missing? (max 20 words)

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string
- "dependencies": list of strings (key information dependencies)
- "missing_reason": string (brief explanation why info is missing, max 20 words)"""

            response = get_response_with_retry(prompt)

            # 清理响应以确保是有效的 JSON
            response = response.strip()

            # 移除 markdown 代码块标记
            response = response.replace("```json", "").replace("```", "")

            # 使用 fix_json_response 解析清理后的响应
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "解析反思响应失败。",
                    "dependencies": ["与问题相关的信息"],
                    "missing_reason": "发生解析错误",
                }

            # 验证必填字段
            required_fields = [
                "can_answer",
                "missing_info",
                "subquery",
                "current_understanding",
            ]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}响应缺少必填字段: {response}{Style.RESET_ALL}")
                raise ValueError("缺少必填字段")

            # 为新的可解释性字段添加默认值（如果缺失）
            if "dependencies" not in result:
                result["dependencies"] = ["与问题相关的信息"]
            if "missing_reason" not in result:
                result["missing_reason"] = (
                    "需要额外上下文" if not result["can_answer"] else "无缺失信息"
                )

            # 确保 can_answer 是布尔类型
            result["can_answer"] = bool(result["can_answer"])

            # 确保子查询非空
            if not result["subquery"]:
                result["subquery"] = question

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}依赖图分析出错: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"分析过程出错: {str(e)}",
                "dependencies": ["与问题相关的信息"],
                "missing_reason": "分析错误",
            }

    def dependency_aware_rag(
        self, question: str, info_summary: str, dependencies: List[str], idx: int
    ) -> Dict[str, Any]:
        """
        依赖感知的 RAG 分析

        与 warm_up_analysis 类似，分析当前信息是否足以回答问题，
        但这里会参考分解后的依赖列表进行分析。

        该函数判断问题是否可以回答，如果不能，则基于依赖关系更新当前查询。

        Args:
            question: 原始问题
            info_summary: 当前信息摘要
            dependencies: 分解后的依赖列表（已排序）
            idx: 当前正在处理的依赖索引

        Returns:
            包含分析结果的字典:
            - can_answer: 是否可以回答
            - current_understanding: 当前理解
        """
        try:
            prompt = f"""
            We pre-parsed the question into a list of dependencies, and the dependencies are sorted in a topological order, below is the question, the information summary, and the decomposed dependencies:

            Question: {question}

            Available Information:
            {info_summary}

            Decomposed dependencies:
            {dependencies}

            Current dependency to be answered:
            {dependencies[idx]}

            Please analyze the question and the information summary, and the decomposed dependencies, and answer the following questions:
            Please analyze:
            1. Can the question be answered completely with this information? (Yes/No)
            2. Summarize our current understanding based on available information.

            Please format your response as a JSON object with these keys:
            - "can_answer": boolean
            - "current_understanding": string
            """
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": False,
                    "current_understanding": "解析 dependency_aware_rag 响应失败。",
                }
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}dependency_aware_rag 出错: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"分析过程出错: {str(e)}",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        """
        基于信息摘要生成最终答案

        提示模型生成简洁直接的答案，避免不必要的解释。

        Args:
            question: 原始问题
            info_summary: 信息摘要

        Returns:
            生成的答案字符串
        """
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """

            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}生成答案时出错: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query) -> List[str]:
        """
        对依赖列表进行拓扑排序

        给定依赖列表和原始查询，按拓扑顺序排列依赖关系。
        即：如果解决依赖 A 需要先解决依赖 B，则 B 应该排在 A 之前。

        Args:
            dependencies: 依赖列表
            query: 原始查询

        Returns:
            拓扑排序后的依赖列表

        示例:
            问题: "法国首都的市长是谁？"
            输入依赖:
            - 法国的首都
            - 这个首都的市长

            输出（排序后）:
            - 法国的首都
            - 这个首都的市长

        算法步骤:
        1. 通过 LLM 生成依赖对（A 依赖于 B）
        2. 使用基于图的拓扑排序算法排序依赖
        """
        # 步骤 1: 通过提示 LLM 生成依赖对
        # 提示词的中文翻译： 给出问题及其分解的依赖关系，输出依赖对，其中依赖 A 依赖于依赖 B。
        # 如果没有找到依赖对，则输出空列表。将响应格式化为包含键 "dependency_pairs" 的 JSON 对象，其值为整数元组列表。
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        if not result or "dependency_pairs" not in result:
            logger.error(f"{Fore.RED}解析依赖对失败；默认使用空列表。{Style.RESET_ALL}")
            dependency_pairs = []
        else:
            dependency_pairs = result["dependency_pairs"]

        # 步骤 2: 使用基于图的拓扑排序算法
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(
        dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]
    ) -> List[str]:
        """
        使用 DFS 算法对依赖进行拓扑排序

        构建依赖图并执行深度优先搜索，得到满足所有依赖约束的线性顺序。

        Args:
            dependencies: 依赖列表
            dependencies_pairs: 依赖对列表，每个元组 (A, B) 表示 A 依赖于 B

        Returns:
            拓扑排序后的依赖列表
        """
        # 构建邻接表形式的依赖图
        graph = {dep: [] for dep in dependencies}

        # 添加边：dependency -> dependent
        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent

        visited = set()
        stack = []

        def dfs(node):
            """深度优先搜索辅助函数"""
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        # 对所有未访问的节点执行 DFS
        for node in graph:
            if node not in visited:
                dfs(node)

        # 返回逆序（拓扑顺序）
        return stack[::-1]

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """
        带去重过滤的检索

        检索 top_k 个不在已检索集合中的唯一文档块。
        如果唯一文档块不足，会扩大检索窗口。

        Args:
            query: 查询文本
            retrieved_chunks_set: 已检索过的文档块集合

        Returns:
            不重复的文档块列表（最多 top_k 个）
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k

        # 如果 top_k 中的唯一结果不足，继续扩大检索范围
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # 扩大检索窗口
            all_results = (
                self.retrieve(query)
                if idx == self.top_k
                else self._retrieve_top_n(query, idx)
            )
            unique_results = [
                chunk for chunk in all_results if chunk not in retrieved_chunks_set
            ]
            idx += self.top_k

        return unique_results[: self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """
        检索前 N 个结果（过滤辅助函数）

        Args:
            query: 查询文本
            n: 要检索的结果数量

        Returns:
            前 n 个检索结果列表
        """
        # 临时覆盖 top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """
        回答问题的主入口函数

        实现完整的 LogicRAG 流程：
        1. 预热检索和分析
        2. 依赖提取和排序
        3. 迭代检索和摘要更新
        4. 最终答案生成

        Args:
            question: 要回答的问题

        Returns:
            三元组:
            - answer: 生成的答案
            - contexts: 最后一轮检索的上下文
            - round_count: 总检索轮数
        """
        # 初始化状态变量
        info_summary = ""  # 滚动信息摘要
        round_count = 0  # 检索轮数计数
        current_query = question  # 当前查询
        retrieval_history = []  # 检索历史记录
        last_contexts = []  # 最后一轮检索的上下文
        dependency_analysis_history = []  # 依赖分析历史

        # 如果启用去重，初始化已检索文档块集合
        retrieved_chunks_set = set() if self.filter_repeats else None

        print(
            f"\n\n{Fore.CYAN}{self.MODEL_NAME} 正在处理问题: {question}{Style.RESET_ALL}\n\n"
        )

        # ===============================================
        # == 阶段 1: 预热检索 ==
        # ===============================================
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)

        last_contexts = new_contexts

        # 生成初始信息摘要
        info_summary = self.refine_summary_with_context(
            question, new_contexts, info_summary
        )

        # 执行预热分析
        analysis = self.warm_up_analysis(question, info_summary)

        if analysis["can_answer"]:
            # 问题可以通过简单事实检索回答，无需依赖分析
            print("预热阶段: 通过简单事实检索找到充分证据；跳过依赖分析。")
            answer = self.generate_answer(question, info_summary)
            # 重置依赖分析历史
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count
        else:
            # 需要进行深度推理增强的 RAG
            logger.info(
                f"预热分析表明需要更深层次的推理增强 RAG。现在使用逻辑依赖图进行分析。"
            )
            logger.info(f"依赖项: {', '.join(analysis.get('dependencies', []))}")

            # 对依赖项进行拓扑排序
            sorted_dependencies = self._sort_dependencies(
                analysis["dependencies"], question
            )
            dependency_analysis_history.append(
                {"sorted_dependencies": sorted_dependencies}
            )
            logger.info(f"排序后的依赖项: {sorted_dependencies}\n\n")

        # ===============================================
        # == 阶段 2: 智能迭代检索 ==
        # ===============================================
        idx = 0  # 当前依赖索引

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1

            # 使用当前依赖作为查询
            current_query = sorted_dependencies[idx]

            # 检索相关文档
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(
                    current_query, retrieved_chunks_set
                )
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)

            last_contexts = new_contexts  # 保存当前上下文

            # 用新上下文更新信息摘要
            info_summary = self.refine_summary_with_context(
                question, new_contexts, info_summary
            )

            logger.info(f"第 {round_count} 轮智能检索")
            logger.info(f"当前查询: {current_query}")

            # 执行依赖感知的 RAG 分析
            analysis = self.dependency_aware_rag(
                question, info_summary, sorted_dependencies, idx
            )

            # 记录检索历史
            retrieval_history.append(
                {
                    "round": round_count,
                    "query": current_query,
                    "contexts": new_contexts,
                }
            )

            # 记录依赖分析历史
            dependency_analysis_history.append(
                {"round": round_count, "query": current_query, "analysis": analysis}
            )

            if analysis["can_answer"]:
                # 可以生成最终答案
                answer = self.generate_answer(question, info_summary)
                # 保存依赖分析历史供评估使用
                self.last_dependency_analysis = dependency_analysis_history
                # 返回最后检索的上下文供评估使用
                return answer, last_contexts, round_count
            else:
                # 继续处理下一个依赖
                idx += 1

        # 达到最大轮数，生成尽可能好的答案
        logger.info(f"已达到最大轮数 ({self.max_rounds})。正在生成最终答案...")
        answer = self.generate_answer(question, info_summary)
        # 保存依赖分析历史供评估使用
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count
