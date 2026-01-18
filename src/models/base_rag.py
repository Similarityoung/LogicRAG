"""
基础 RAG 模块

提供 RAG 系统的核心功能，包括：
- 文档语料库加载与管理
- 句子嵌入向量计算（支持批量处理）
- 基于余弦相似度的文档检索
- 嵌入向量缓存机制
"""

import os
import gc
import json
import pdb
import pickle
import torch
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from config.config import CACHE_DIR, RESULT_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BaseRAG:
    """
    基础 RAG（检索增强生成）类

    提供文档检索的核心功能：
    - 加载和处理文档语料库
    - 计算文档的嵌入向量
    - 基于语义相似度检索相关文档

    属性:
        MODEL_NAME: 模型名称标识
        cache_dir: 缓存目录路径
        model: SentenceTransformer 嵌入模型
        corpus: 文档语料库字典 {索引: 文档内容}
        corpus_embeddings: 语料库的嵌入向量张量
        retrieval_cache: 检索结果缓存
        top_k: 检索返回的文档数量
    """

    def __init__(self, corpus_path: str = None, cache_dir: str = CACHE_DIR):
        """
        初始化 BaseRAG 系统

        Args:
            corpus_path: 语料库文件路径（JSON 格式）
            cache_dir: 缓存目录路径，用于存储嵌入向量
        """
        self.MODEL_NAME = "BaseRAG"
        self.cache_dir = cache_dir

        # 创建必要的目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 初始化嵌入模型
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # 初始化数据结构
        self.corpus = {}  # 文档语料库
        self.corpus_embeddings = None  # 语料库嵌入向量
        self.embeddings = None  # 兼容 vanilla_retrieve 的嵌入向量引用
        self.sentences = None  # 兼容 vanilla_retrieve 的句子列表
        self.retrieval_cache = {}  # 检索结果缓存，避免重复计算
        self.top_k = 5  # 默认检索数量

        # 如果提供了语料库路径，则加载语料库
        if corpus_path:
            self.load_corpus(corpus_path)

    def load_corpus(self, corpus_path: str):
        """
        加载并处理文档语料库

        从 JSON 文件加载文档，将每个文档转换为 "Title: ... Content: ..." 格式，
        然后计算或加载缓存的嵌入向量。

        Args:
            corpus_path: 语料库 JSON 文件路径，格式为 [{"title": "...", "text": "..."}, ...]
        """
        logger.info("正在加载语料库...")
        with open(corpus_path, "r") as f:
            documents = json.load(f)

        # 将文档处理为统一格式
        self.corpus = {
            i: f"Title: {doc['title']}. Content: {doc['text']}"
            for i, doc in enumerate(documents)
        }

        # 存储句子列表以支持 vanilla 检索
        self.sentences = list(self.corpus.values())

        # 尝试加载缓存的嵌入向量
        cache_file = os.path.join(self.cache_dir, f"embeddings_{len(self.corpus)}.pt")

        if os.path.exists(cache_file):
            logger.info("正在加载缓存的嵌入向量...")
            self.corpus_embeddings = torch.load(cache_file)
            self.embeddings = self.corpus_embeddings  # 保持兼容性
        else:
            logger.info("正在计算嵌入向量...")
            texts = list(self.corpus.values())
            self.corpus_embeddings = self.encode_sentences_batch(texts)
            self.embeddings = self.corpus_embeddings  # 保持兼容性
            # 保存嵌入向量到缓存
            torch.save(self.corpus_embeddings, cache_file)

    # def encode_batch(
    #     self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
    # ) -> np.ndarray:
    #     """
    #     批量编码文本为嵌入向量

    #     Args:
    #         texts: 待编码的文本列表
    #         batch_size: 每批处理的文本数量

    #     Returns:
    #         合并后的嵌入向量张量
    #     """
    #     all_embeddings = []
    #     for i in range(0, len(texts), batch_size):
    #         batch = texts[i : i + batch_size]
    #         embeddings = self.model.encode(batch, convert_to_tensor=True)
    #         all_embeddings.append(embeddings)
    #     return torch.cat(all_embeddings)

    def encode_sentences_batch(
        self, sentences: List[str], batch_size: int = 32
    ) -> torch.Tensor:
        """
        批量编码句子，带有内存管理优化

        该方法在每个批次处理后会清理 GPU 缓存，适用于大规模语料库的嵌入计算。

        Args:
            sentences: 待编码的句子列表
            batch_size: 每批处理的句子数量

        Returns:
            所有句子的嵌入向量张量 (num_sentences, embedding_dim)
        """
        all_embeddings = []


        for i in tqdm(range(0, len(sentences), batch_size), desc="编码句子中"):
            batch = sentences[i : i + batch_size]

            # 内存清理：释放未使用的内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 计算嵌入向量（禁用梯度计算以节省内存）
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch, convert_to_tensor=True, show_progress_bar=False
                )
                # 将嵌入向量移到 CPU 以释放 GPU 内存
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        # 合并所有批次的嵌入向量
        final_embeddings = torch.cat(all_embeddings, dim=0)

        # 清理临时变量
        del all_embeddings
        gc.collect()

        return final_embeddings

    # def build_index(self, sentences: List[str], batch_size: int = 32):
    #     """
    #     为句子列表构建嵌入索引

    #     优先尝试从缓存加载已有的嵌入向量，否则重新计算并保存。

    #     Args:
    #         sentences: 待索引的句子列表
    #         batch_size: 编码时的批量大小
    #     """
    #     self.sentences = sentences

    #     # 尝试加载已有的嵌入向量
    #     embedding_file = f"cache/embeddings_{len(sentences)}.pkl"
    #     if os.path.exists(embedding_file):
    #         try:
    #             with open(embedding_file, "rb") as f:
    #                 self.embeddings = pickle.load(f)
    #             logger.info(f"已从 {embedding_file} 加载嵌入向量")
    #             return
    #         except Exception as e:
    #             logger.error(f"加载嵌入向量时出错: {e}")

    #     # 构建新的嵌入向量
    #     self.embeddings = self.encode_sentences_batch(sentences, batch_size)

    #     # 保存嵌入向量到缓存
    #     try:
    #         os.makedirs("cache", exist_ok=True)
    #         with open(embedding_file, "wb") as f:
    #             pickle.dump(self.embeddings, f)
    #     except Exception as e:
    #         logger.error(f"保存嵌入向量时出错: {e}")

    def retrieve(self, query: str) -> List[str]:
        """
        检索与查询最相似的文档

        使用余弦相似度计算查询与所有文档的相似度，返回 top_k 个最相似的文档。
        结果会被缓存以避免重复计算。

        Args:
            query: 查询文本

        Returns:
            最相似的 top_k 个文档内容列表
        """
        # 检查缓存
        if query in self.retrieval_cache:
            return self.retrieval_cache[query]

        # 检查语料库是否已加载
        if self.corpus_embeddings is None or not self.corpus:
            return []

        try:
            # 编码查询
            with torch.no_grad():
                query_embedding = self.model.encode([query], convert_to_tensor=True)[0]
                query_embedding = query_embedding.cpu()

            # 计算余弦相似度
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), self.corpus_embeddings
            )

            # 获取 top_k 个最相似的文档索引
            _, top_k_indices = similarities.topk(self.top_k)
            indices = top_k_indices.tolist()

            # 根据索引获取文档内容
            results = [self.corpus[idx] for idx in indices]

            # 缓存结果
            self.retrieval_cache[query] = results
            return results

        except Exception as e:
            logger.error(f"检索时出错: {e}")
            return []

    def set_top_k(self, top_k: int):
        """
        设置检索返回的文档数量

        Args:
            top_k: 返回的最相似文档数量
        """
        self.top_k = top_k
