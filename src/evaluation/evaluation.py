import logging
import pdb
import torch
import time
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm
from datetime import datetime

from src.utils.utils import (
    normalize_answer, 
    evaluate_with_llm, 
    string_based_evaluation,
    save_results,
    TOKEN_COST
)
from src.models.logic_rag import LogicRAG
from config.config import RESULT_DIR

# 配置日志系统
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 可用的 RAG 模型字典
RAG_MODELS = {
    "logic-rag": LogicRAG,
}

# 检查点保存目录
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = 5  # 默认:每处理 5 个问题保存一次检查点

class RAGEvaluator:
    """RAG 模型评估器

    用于评估 RAG 模型在数据集上的性能，支持：
    - 批量问题评估
    - 检查点保存和恢复（防止数据丢失）
    - 多种评估指标（准确率、召回率、Top-k 检索等）
    """

    def __init__(self, model_name: str, corpus_path: str, max_rounds: int = 3, top_k: int = 5,
                eval_top_ks: List[int] = [5, 10], checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        """初始化评估器

        Args:
            model_name: 要评估的 RAG 模型名称
            corpus_path: 语料库文件路径
            max_rounds: 代理式 RAG 的最大迭代轮数
            top_k: 每次检索的上下文数量
            eval_top_ks: Top-k 准确率评估的 k 值列表
            checkpoint_interval: 保存检查点前处理的问题数量
        """
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)  # 排序以确保一致的处理顺序
        self.checkpoint_interval = checkpoint_interval

        # 创建结果目录（如果不存在）
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 创建检查点目录（如果不存在）
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # 初始化 RAG 模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化指定的 RAG 模型"""
        if self.model_name not in RAG_MODELS:
            raise ValueError(f"未知的 RAG 模型: {self.model_name}")

        # 创建模型实例
        model_class = RAG_MODELS[self.model_name]
        self.model = model_class(self.corpus_path)

        # 配置模型参数
        self.model.set_top_k(self.top_k)

        # 设置代理式模型的最大迭代轮数
        if hasattr(self.model, 'set_max_rounds'):
            self.model.set_max_rounds(self.max_rounds)

        logger.info(f"已初始化 {self.model_name} 模型")
    
    def evaluate_question(self, question: str, gold_answer: str) -> Dict:
        """评估单个问题的回答质量

        Args:
            question: 待评估的问题
            gold_answer: 标准答案

        Returns:
            包含评估结果的字典，包括：
            - question: 问题
            - gold_answer: 标准答案
            - answer: 模型生成的答案
            - contexts: 检索到的上下文列表
            - time: 处理时间（秒）
            - rounds: 迭代轮数
            - is_correct: LLM 评估的正确性
            - dependency_analysis: 依赖分析（如果模型支持）
        """
        # 记录开始时间
        start_time = time.time()

        # 运行模型生成答案
        answer, contexts, rounds = self.model.answer_question(question)
        elapsed_time = time.time() - start_time

        # 使用 LLM 评估答案正确性
        is_correct = evaluate_with_llm(answer, gold_answer)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "answer": answer,
            "contexts": contexts,
            "time": elapsed_time,
            "rounds": rounds,
            "is_correct": is_correct
        }

        # 添加依赖分析（用于可解释性分析）
        if hasattr(self.model, 'last_dependency_analysis'):
            result["dependency_analysis"] = self.model.last_dependency_analysis

        return result
        
    def calculate_retrieval_metrics(self, retrieved_contexts: List[List[str]], answers: List[str]) -> Dict[str, float]:
        """计算基于检索的评估指标

        Args:
            retrieved_contexts: 检索到的上下文列表（每个问题对应一个上下文列表）
            answers: 标准答案列表

        Returns:
            包含以下指标的字典：
            - answer_found_in_context: 答案在上下文中被找到的比例
            - total_questions: 总问题数
            - answer_in_top{k}: 答案在 Top-k 检索结果中的比例
        """
        total = len(answers)
        found_in_context = 0

        if total == 0:
            result = {
                "answer_found_in_context": 0.0,
                "total_questions": 0
            }
            # Add top-k keys with 0 values
            for k in self.eval_top_ks:
                result[f"answer_in_top{k}"] = 0.0
            return result

        # 为每个 top-k 值初始化计数器
        answer_in_top_k = {k: 0 for k in self.eval_top_ks}

        for contexts, answer in zip(retrieved_contexts, answers):
            normalized_answer = normalize_answer(answer)

            # 检查答案是否出现在任何上下文中
            for i, context in enumerate(contexts):
                if normalized_answer in normalize_answer(context):
                    found_in_context += 1
                    # 更新每个 k 值的计数器
                    for k in self.eval_top_ks:
                        if i < k:
                            answer_in_top_k[k] += 1
                    break

        # 准备结果字典
        result = {
            "answer_found_in_context": found_in_context / total,
            "total_questions": total
        }

        # 添加 top-k 指标到结果中
        for k in self.eval_top_ks:
            result[f"answer_in_top{k}"] = answer_in_top_k[k] / total

        return result
    
    def _generate_unique_filename(self, output_file: str, use_timestamp: bool = True) -> str:
        """生成唯一的输出文件名，避免覆盖
        
        Args:
            output_file: 用户指定的输出文件名 (例如 "hotpotqa_results.json")
            use_timestamp: 是否添加时间戳（默认 True）
            
        Returns:
            带时间戳的文件名 (例如 "hotpotqa_results_20250118_120000.json")
        """
        output_path = os.path.join(RESULT_DIR, output_file)
        
        # 如果不使用时间戳且文件不存在，直接返回原名
        if not use_timestamp and not os.path.exists(output_path):
            return output_file
            
        base_name = os.path.basename(output_file)
        name, ext = os.path.splitext(base_name)
        
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{name}_{timestamp}{ext}"
        else:
            # 简单的数字递增备选方案
            counter = 1
            while True:
                new_name = f"{name}_{counter}{ext}"
                new_path = os.path.join(RESULT_DIR, new_name)
                if not os.path.exists(new_path):
                    break
                counter += 1
                
        return new_name

    def _get_checkpoint_path(self, output_file: str) -> str:
        """根据输出文件基础名称生成检查点文件路径
        
        命名规则: {基础名}_k{top_k}_r{max_rounds}_checkpoint.json
        例如: hotpotqa_results_k5_r3_checkpoint.json
        
        注意：这种命名方式支持断点续传。只要参数相同，就会映射到同一个检查点文件。
        """
        base_name = os.path.basename(output_file)
        # 去掉可能已有的扩展名
        name = os.path.splitext(base_name)[0]
        
        # 移除可能存在的时间戳后缀 (假设格式为 _YYYYMMDD_HHMMSS)
        # 这样确保 "hotpotqa_results_2025..." 和 "hotpotqa_results_2026..." 
        # 都能映射到同一个 "hotpotqa_results_checkpoint"
        import re
        timestamp_pattern = r'_\d{8}(_\d{6})?$'
        name_clean = re.sub(timestamp_pattern, '', name)

        checkpoint_name = f"{name_clean}_k{self.top_k}_r{self.max_rounds}_checkpoint.json"
        return os.path.join(CHECKPOINT_DIR, checkpoint_name)

    def _save_checkpoint(self, results: List[Dict], metrics: Dict, processed_count: int, output_file: str):
        """保存当前评估进度的检查点

        Args:
            results: 已处理问题的结果列表
            metrics: 当前的评估指标
            processed_count: 已处理的问题数量
            output_file: 输出文件名

        注意:
            - 如果最后一个结果的答案为空，说明 LLM API 连接断开
            - 这种情况下不保存检查点，并终止整个流程
        """
        # 如果最后一个结果的答案为空，说明 LLM API 连接断开
        # 此时不应保存检查点，并终止整个流程
        if results[-1]["answer"] == "":
            print("\n\n\033[91mLLM API 连接断开，跳过检查点保存\033[0m\n\n")
            exit(1)  # 使用错误代码 1 表示异常终止

        checkpoint = {
            "model": self.model_name,
            "metrics": metrics,
            "results": results,
            "processed_count": processed_count,
            "token_cost": {
                "prompt": TOKEN_COST["prompt"],
                "completion": TOKEN_COST["completion"]
            }
        }
        
        # 确保使用原始输出文件名（不含时间戳）来生成检查点路径
        # 这样无论这次运行的时间戳是什么，都会更新同一个检查点文件
        checkpoint_path = self._get_checkpoint_path(output_file)

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"检查点已保存: 已处理 {processed_count} 个问题")
    
    def _load_checkpoint(self, output_file: str) -> Tuple[List[Dict], Dict, int]:
        """加载评估检查点（如果存在）

        Args:
            output_file: 输出文件名

        Returns:
            包含三个元素的元组：
            - results: 已处理问题的结果列表
            - metrics: 当前的评估指标
            - processed_count: 已处理的问题数量
            如果检查点不存在，返回空值
        """
        checkpoint_path = self._get_checkpoint_path(output_file)

        if not os.path.exists(checkpoint_path):
            return [], {}, 0

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # 恢复 token 成本
            if "token_cost" in checkpoint:
                TOKEN_COST["prompt"] = checkpoint["token_cost"]["prompt"]
                TOKEN_COST["completion"] = checkpoint["token_cost"]["completion"]
                logger.info(f"已恢复 token 成本 - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")

            logger.info(f"已加载检查点: 已处理 {checkpoint['processed_count']} 个问题")
            return checkpoint["results"], checkpoint["metrics"], checkpoint["processed_count"]
        except Exception as e:
            logger.error(f"加载检查点时出错: {e}")
            return [], {}, 0
    
    def run_single_model_evaluation(self, eval_data: List[Dict], output_file: str = "evaluation_results.json",
                                    auto_rename: bool = True) -> Dict:
        """在给定的评估数据上运行单个模型的评估

        Args:
            eval_data: 评估数据列表，每个元素包含 'question' 和 'answer' 字段
            output_file: 结果输出文件名
            auto_rename: 是否自动重命名重复的输出文件（默认 True）

        Returns:
            评估摘要字典，包含：
            - model: 模型名称
            - metrics: 组织好的评估指标
            - results: 详细结果列表

        注意:
            - 支持从检查点恢复评估（断点续传）
            - 定期保存检查点以防数据丢失
            - 自动计算多种评估指标
            - Checkpoint 命名格式: {dataset}_k{top_k}_r{max_rounds}_checkpoint.json
            - Result 命名格式: {dataset}_{timestamp}.json
        """
        original_output_file = output_file
        
        # 1. 尝试从检查点恢复 (使用原始文件名来定位检查点)
        # 即使这次我们最终会生成一个新的带时间戳的结果文件
        # 我们依然应该先尝试加载可能存在的、对应此配置的检查点
        results, metrics, processed_count = self._load_checkpoint(original_output_file)

        # 2. 生成本次运行的实际输出文件名（防止覆盖旧结果）
        if auto_rename:
            output_file = self._generate_unique_filename(output_file, use_timestamp=True)
            logger.info(f"本次运行结果将保存至: {output_file}")

        # 跳过已处理的问题
        if processed_count > 0:
            eval_data = eval_data[processed_count:]
            logger.info(f"从检查点恢复: 已处理 {processed_count} 个问题，剩余 {len(eval_data)} 个")

        # 如果所有问题都已处理
        if not eval_data:
            logger.info("所有问题在之前的运行中已处理完毕。")
            # 从最终输出加载完整结果
            output_path = os.path.join(RESULT_DIR, output_file)
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    evaluation_summary = json.load(f)
                return evaluation_summary

        # 如果是全新开始，初始化指标并重置 token 成本
        if not metrics:
            # 为当前模型评估重置 token 成本
            TOKEN_COST["prompt"] = 0
            TOKEN_COST["completion"] = 0

            # 初始化指标字典（包含动态的 top-k 键）
            metrics = {
                "total_time": 0,
                "answer_coverage": 0,
                "answer_accuracy": 0,
                "string_accuracy": 0,
                "string_precision": 0,
                "string_recall": 0
            }

            # 为每个 top-k 值添加命中计数
            for k in self.eval_top_ks:
                metrics[f"top{k}_hits"] = 0

            # 添加轮次追踪
            metrics["total_rounds"] = 0
        else:
            logger.info(f"已恢复 token 成本 - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")

        # 评估指标
        total_questions = len(eval_data) + processed_count
        
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {self.model_name}")):
            question = item['question']
            gold_answer = item['answer']

            # 评估模型在这个问题上的表现
            result = self.evaluate_question(
                question=question,
                gold_answer=gold_answer
            )
            results.append(result)

            # 更新指标
            metrics["total_time"] += result["time"]
            normalized_gold = normalize_answer(gold_answer)

            # 基于字符串的评估
            string_metrics = string_based_evaluation(
                result["answer"],
                gold_answer
            )
            metrics["string_accuracy"] += string_metrics["accuracy"]
            metrics["string_precision"] += string_metrics["precision"]
            metrics["string_recall"] += string_metrics["recall"]

            # 检查检索覆盖率（答案是否在检索到的上下文中）
            for j, ctx in enumerate(result["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    metrics["answer_coverage"] += 1
                    # 更新每个 k 值的计数器
                    for k in self.eval_top_ks:
                        if j < k:
                            metrics[f"top{k}_hits"] += 1
                    break

            # 更新迭代轮次
            if "rounds" in result:
                metrics["total_rounds"] += result["rounds"]

            # 使用 LLM 评估答案
            if result["is_correct"]:
                metrics["answer_accuracy"] += 1

            # 定期保存检查点
            current_count = processed_count + i + 1
            if (current_count % self.checkpoint_interval == 0) or (i == len(eval_data) - 1):
                # 即使本次输出文件变了，我们依然更新那个基于"原始文件名"的检查点
                # 这样下次运行（无论叫什么时间戳）都能找到这个最新的检查点
                self._save_checkpoint(results, metrics, current_count, original_output_file)
        
        # 计算平均指标
        avg_metrics = {
            "avg_time": metrics["total_time"] / total_questions,
            "answer_coverage": metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": metrics["string_accuracy"] / total_questions * 100,
            "string_precision": metrics["string_precision"] / total_questions * 100,
            "string_recall": metrics["string_recall"] / total_questions * 100
        }

        # 为每个 top-k 值添加覆盖率指标
        for k in self.eval_top_ks:
            avg_metrics[f"top{k}_coverage"] = metrics[f"top{k}_hits"] / total_questions * 100

        # 添加平均轮次
        avg_metrics["avg_rounds"] = metrics["total_rounds"] / total_questions

        # 按类别组织指标
        organized_metrics = {
            "performance": {
                "avg_time": avg_metrics["avg_time"]
            },
            "string_based": {
                "accuracy": avg_metrics["string_accuracy"],
                "precision": avg_metrics["string_precision"],
                "recall": avg_metrics["string_recall"]
            },
            "llm_evaluated": {
                "answer_accuracy": avg_metrics["answer_accuracy"]
            },
            "retrieval": {
                "answer_coverage": avg_metrics["answer_coverage"]
            }
        }

        # 添加平均轮次
        organized_metrics["performance"]["avg_rounds"] = avg_metrics["avg_rounds"]

        # 添加 token 成本指标
        if total_questions > 0:
            organized_metrics["performance"]["avg_prompt_tokens"] = TOKEN_COST["prompt"] / total_questions
            organized_metrics["performance"]["avg_completion_tokens"] = TOKEN_COST["completion"] / total_questions
            organized_metrics["performance"]["avg_total_tokens"] = (TOKEN_COST["prompt"] + TOKEN_COST["completion"]) / total_questions
        else:
            organized_metrics["performance"]["avg_prompt_tokens"] = 0
            organized_metrics["performance"]["avg_completion_tokens"] = 0
            organized_metrics["performance"]["avg_total_tokens"] = 0

        # 添加 top-k 覆盖率指标
        for k in self.eval_top_ks:
            organized_metrics["retrieval"][f"top{k}_coverage"] = avg_metrics[f"top{k}_coverage"]

        # 添加原始指标以保持向后兼容性
        organized_metrics["raw"] = metrics

        # 准备最终评估摘要
        evaluation_summary = {
            "model": self.model_name,
            "metrics": organized_metrics,
            "results": results
        }

        # 保存结果
        save_results(
            results=evaluation_summary,
            output_file=output_file,
            results_dir=RESULT_DIR
        )

        # 分三部分记录结果日志
        logger.info(f"\n{self.model_name} 评估摘要:")

        # 性能指标
        logger.info(f"平均每题用时: {avg_metrics['avg_time']:.2f} 秒")
        logger.info(f"平均每题轮次: {avg_metrics['avg_rounds']:.2f}")

        # 记录 token 成本
        logger.info(f"平均每题 prompt tokens: {organized_metrics['performance']['avg_prompt_tokens']:.2f}")
        logger.info(f"平均每题 completion tokens: {organized_metrics['performance']['avg_completion_tokens']:.2f}")
        logger.info(f"平均每题总 tokens: {organized_metrics['performance']['avg_total_tokens']:.2f}")

        # 1. 基于字符串的指标
        logger.info("\n1. 基于字符串的指标:")
        logger.info(f"  • 准确率: {avg_metrics['string_accuracy']:.2f}%")
        logger.info(f"  • 精确率: {avg_metrics['string_precision']:.2f}%")
        logger.info(f"  • 召回率: {avg_metrics['string_recall']:.2f}%")

        # 2. LLM 评估的指标
        logger.info("\n2. LLM 评估的指标:")
        logger.info(f"  • 答案准确率: {avg_metrics['answer_accuracy']:.2f}%")

        # 3. 检索性能
        logger.info("\n3. 检索性能:")
        logger.info(f"  • 答案覆盖率: {avg_metrics['answer_coverage']:.2f}%")

        # 记录 top-k 覆盖率指标
        for k in self.eval_top_ks:
            logger.info(f"  • Top-{k} 覆盖率: {avg_metrics[f'top{k}_coverage']:.2f}%")

        return evaluation_summary