"""
工具函数模块
提供 API 调用、JSON 修复、答案评估等通用功能
"""
import os
import logging
import re
import json
import time
import backoff
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from collections import Counter
from typing import List, Dict, Any
from colorama import Fore, Style, init
from config.config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    CALLS_PER_MINUTE,
    PERIOD,
    MAX_RETRIES,
    RETRY_DELAY
)

# 初始化 colorama（用于终端彩色输出）
init()

# 配置日志系统
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 初始化 token 使用量跟踪
TOKEN_COST = {"prompt": 0, "completion": 0}

# 配置 OpenAI 客户端（支持第三方兼容的 API）
from config.config import OPENAI_API_BASE

# 构建客户端初始化参数
client_kwargs = {"api_key": OPENAI_API_KEY}

# 如果配置了第三方 API 的 base_url，则使用它
if OPENAI_API_BASE:
    client_kwargs["base_url"] = OPENAI_API_BASE

# 创建 OpenAI 客户端实例
client = OpenAI(**client_kwargs)
# 设置环境变量，禁用 tokenizers 并行化（避免警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 反思提示词模板（用于分析是否需要更多信息）
REFLECTION_PROMPT = """Based on the question and the retrieved context, analyze:
1. Can you confidently answer the question with the given context and your knowledge?
2. If not, what specific information is missing?
3. Generate a focused search query to find the missing information.

Format your response as:
{
    "can_answer": true/false,
    "missing_info": "description of what information is missing",
    "subquery": "specific search query for missing information",
    "current_understanding": "brief summary of current understanding"
}
"""


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=MAX_RETRIES,
    max_time=300
)
def get_response_with_retry(prompt: str, temperature: float = 0.0, print_cost: bool = False) -> str:
    """
    调用 OpenAI API 获取响应（带重试机制）

    功能：
    - 使用装饰器实现速率限制（每分钟最多 CALLS_PER_MINUTE 次调用）
    - 使用指数退避策略进行异常重试（最多重试 MAX_RETRIES 次）
    - 自动跟踪 token 使用量

    Args:
        prompt: 发送给 LLM 的提示词
        temperature: 温度参数，控制输出的随机性（0.0 表示确定性输出）
        print_cost: 是否打印 token 使用量信息

    Returns:
        str: LLM 的响应内容，失败时返回空字符串
    """
    global TOKEN_COST
    try:
        # 构建消息列表
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # 调用 OpenAI API
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS
        )

        # 更新 token 使用量统计
        if response.usage:
            TOKEN_COST["prompt"] += response.usage.prompt_tokens
            TOKEN_COST["completion"] += response.usage.completion_tokens

        # 如果需要，打印 token 使用信息
        if print_cost:
            logger.info(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.info(f"Completion tokens: {response.usage.completion_tokens}")
            logger.info(f"Total tokens: {response.usage.total_tokens}")

        # 返回响应内容
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error in get_response_with_retry: {str(e)}")
        return ""


def fix_json_response(response: str) -> str:
    """
    修复 LLM 返回的不完整 JSON 响应

    功能：
    - 移除 markdown 代码块标记（```json 和 ```）
    - 处理两种常见的不完整情况：
      1. 缺少结尾的右花括号 }
      2. current_understanding 字段被截断

    Args:
        response: LLM 返回的原始响应字符串

    Returns:
        dict: 修复后的 JSON 对象，如果修复失败则返回 None
    """
    # 移除 markdown 代码块标记和空白字符
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')
    original_response = response  # 保存原始响应用于比较

    try:
        # 尝试直接解析
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # 情况 1: 检查是否缺少结尾的花括号
        if response.count('{') > response.count('}'):
            try:
                fixed_response = response + '}'
                result = json.loads(fixed_response)
                logger.info(f"{Fore.RED}Fixed JSON by adding closing brace:{Style.RESET_ALL}")
                logger.info(f"{Fore.RED}Original: {original_response}{Style.RESET_ALL}")
                logger.info(f"{Fore.GREEN}Fixed: {fixed_response}{Style.RESET_ALL}")
                return result
            except json.JSONDecodeError:
                pass

        # 情况 2: 检查 current_understanding 字段是否被截断
        try:
            # 查找 current_understanding 字段之前的最后一个完整字段
            pattern = r'(.*"current_understanding"\s*:\s*"[^"]*)("|$)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                # 获取到截断点之前的所有内容
                fixed_response = match.group(1)
                # 如果没有以引号结尾，添加省略号并闭合引号
                if not fixed_response.endswith('"'):
                    fixed_response += '..."'
                # 如果缺少闭合花括号，则添加
                if response.count('{') > response.count('}'):
                    fixed_response += '}'
                try:
                    result = json.loads(fixed_response)
                    logger.info(f"{Fore.RED}Fixed truncated current_understanding:{Style.RESET_ALL}")
                    logger.info(f"{Fore.RED}Original: {original_response}{Style.RESET_ALL}")
                    logger.info(f"{Fore.GREEN}Fixed: {fixed_response}{Style.RESET_ALL}")
                    return result
                except json.JSONDecodeError:
                    pass
        except:
            pass

    # 修复失败
    logger.error(f"{Fore.RED}Failed to fix JSON response: {original_response}{Style.RESET_ALL}")
    return None


def normalize_answer(text: str) -> str:
    """
    标准化答案文本（用于答案比较和评估）

    处理步骤：
    1. 转换为小写
    2. 将连字符替换为空格
    3. 移除所有标点符号
    4. 移除多余的空白字符

    Args:
        text: 待标准化的文本

    Returns:
        str: 标准化后的文本
    """
    if not isinstance(text, str):
        return ""

    # 转换为小写
    text = text.lower()

    # 将连字符替换为空格
    text = text.replace('-', ' ')

    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 移除多余的空白字符
    text = ' '.join(text.split())

    return text


def save_results(results: Dict, output_file: str, results_dir: str = 'result'):
    """
    保存评估结果到 JSON 文件

    Args:
        results: 包含评估结果的字典
        output_file: 输出文件名
        results_dir: 结果保存目录（默认为 'result'）
    """
    # 创建结果目录（如果不存在）
    os.makedirs(results_dir, exist_ok=True)

    # 构建输出路径
    output_path = os.path.join(results_dir, output_file)

    # 保存结果（使用 UTF-8 编码，不转义中文字符）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")


def evaluate_with_llm(generated: str, gold: str) -> bool:
    """
    使用 LLM 评估生成的答案是否正确

    评估标准：
    1. 生成的答案是否包含金标准答案的关键信息
    2. 答案是否事实准确且与金标准一致
    3. 是否没有矛盾信息

    Args:
        generated: 模型生成的答案
        gold: 金标准答案（ground truth）

    Returns:
        bool: 答案是否正确（True/False）
    """
    # 类型检查
    if not isinstance(generated, str) or not isinstance(gold, str):
        return False

    # 构建评估提示词
    prompt = f"""You are an expert evaluator. Please evaluate if the generated answer is correct by comparing it with the gold answer.

Generated answer: {generated}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""

    try:
        # 调用 LLM 进行评估
        response = get_response_with_retry(prompt, temperature=0.0, print_cost=True)
        return response.strip().lower() == "correct"
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        return False


def string_based_evaluation(generated: str, gold: str) -> dict:
    """
    基于字符串相似度评估答案质量

    评估指标：
    - Accuracy: 金标准答案是否是生成答案的子串
    - Precision: 预测结果中相关 token 的比例
    - Recall: 金标准答案中被正确预测的 token 比例

    Args:
        generated: 生成的答案字符串
        gold: 金标准答案字符串

    Returns:
        dict: 包含 accuracy、precision、recall 的评估指标字典
    """
    # 标准化答案
    normalized_prediction = normalize_answer(generated)
    normalized_ground_truth = normalize_answer(gold)

    # 计算 accuracy（金标准是否包含在预测中）
    accuracy = 1 if normalized_ground_truth in normalized_prediction else 0

    # 将答案分词
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    # 特殊处理 yes/no/noanswer 类型的答案
    # 如果预测和金标准都是 yes/no 类型但不一致，则 precision 和 recall 都为 0
    if (normalized_prediction in ["yes", "no", "noanswer"] and
        normalized_prediction != normalized_ground_truth) or \
       (normalized_ground_truth in ["yes", "no", "noanswer"] and
        normalized_prediction != normalized_ground_truth):
        return {
            "accuracy": accuracy,
            "precision": 0,
            "recall": 0
        }

    # 计算 token 重叠（使用 Counter 的交集操作）
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    # 计算 precision 和 recall
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
