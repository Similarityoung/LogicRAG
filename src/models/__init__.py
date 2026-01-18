"""
LogicRAG 模型模块

该模块提供了 RAG（检索增强生成）系统的核心实现：
- BaseRAG: 基础 RAG 类，提供嵌入向量计算、文档检索等核心功能
- LogicRAG: 逻辑依赖感知的 RAG 实现，支持多轮迭代检索和依赖图分析
"""

from src.models.base_rag import BaseRAG
from src.models.logic_rag import LogicRAG

__all__ = ["BaseRAG", "LogicRAG"]
