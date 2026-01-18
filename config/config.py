"""
Configuration file for API keys and other settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Third-party API Configuration (可选)
# 如果使用第三方兼容 OpenAI 的服务，取消下面的注释并配置
OPENAI_API_BASE = os.environ.get(
    "OPENAI_API_BASE", None
)  # 例如: "https://api.deepseek.com/v1"
OPENAI_API_TYPE = os.environ.get(
    "OPENAI_API_TYPE", "openai"
)  # 可选值: "openai", "azure", "deepseek", "siliconflow" 等

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")  # 默认模型，可在 .env 中配置
DEFAULT_MAX_TOKENS = 250

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # please specify your preferred embedding model
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "result"
