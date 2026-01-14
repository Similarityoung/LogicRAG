# API 配置指南

本文档说明如何配置 LogicRAG 使用不同的 API 服务。

## 🚀 快速开始

### 步骤 1: 复制配置模板

```bash
cp .env.example .env
```

### 步骤 2: 编辑 `.env` 文件

根据你使用的服务，取消相应配置的注释并填入你的 API Key。

### 步骤 3: 验证配置

```bash
python run.py --model logic-rag \
  --question "测试问题" \
  --corpus dataset/hotpotqa_corpus.json
```

## 🔧 支持的 API 服务

### 1. OpenAI 官方

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
# OPENAI_API_BASE 不需要设置
```

- **费用**: 较高
- **稳定性**: ⭐⭐⭐⭐⭐
- **推荐模型**: gpt-4o-mini
- **获取 API Key**: https://platform.openai.com/api-keys

### 2. DeepSeek (推荐)

```bash
OPENAI_API_KEY=sk-your-deepseek-api-key-here
OPENAI_API_BASE=https://api.deepseek.com/v1
```

- **费用**: 低 (约 ¥1/百万 tokens)
- **稳定性**: ⭐⭐⭐⭐⭐
- **推荐模型**: deepseek-chat
- **获取 API Key**: https://platform.deepseek.com/
- **配置修改**: 在 `config/config.py` 中修改 `DEFAULT_MODEL = "deepseek-chat"`

**优点**:
- 性价比极高
- 完全兼容 OpenAI API
- 中文支持好
- 适合学习和实验

### 3. SiliconFlow (硅基流动)

```bash
OPENAI_API_KEY=sk-your-siliconflow-api-key-here
OPENAI_API_BASE=https://api.siliconflow.cn/v1
```

- **费用**: 低
- **稳定性**: ⭐⭐⭐⭐
- **推荐模型**: Qwen/Qwen2.5-7B-Instruct
- **获取 API Key**: https://siliconflow.cn/
- **配置修改**: 在 `config/config.py` 中修改 `DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"`

**优点**:
- 提供多个开源模型
- 价格便宜
- 国内访问稳定

### 4. 其他服务

#### Moonshot (月之暗面 - Kimi)
```bash
OPENAI_API_KEY=sk-your-moonshot-api-key
OPENAI_API_BASE=https://api.moonshot.cn/v1
```
获取 API Key: https://platform.moonshot.cn/

#### 智谱 AI (BigModel)
```bash
OPENAI_API_KEY=your-zhipu-api-key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
```
获取 API Key: https://open.bigmodel.cn/

#### 阿里云百炼
```bash
OPENAI_API_KEY=sk-your-dashscope-api-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```
获取 API Key: https://dashscope.aliyuncs.com/

## 📝 配置说明

### 必需配置

1. **`.env` 文件**
   - 必须在项目根目录创建
   - 不要将 `.env` 文件提交到 Git
   - 参考 `.env.example` 进行配置

2. **API Key**
   - 必须设置 `OPENAI_API_KEY`
   - 格式通常是 `sk-` 开头

### 可选配置

1. **API Base URL**
   - 如果使用第三方服务，需要设置 `OPENAI_API_BASE`
   - OpenAI 官方不需要设置

2. **模型选择**
   - 在 `config/config.py` 中修改 `DEFAULT_MODEL`
   - 确保模型名称与你的 API 服务兼容

3. **速率限制**
   - 在 `config/config.py` 中调整 `CALLS_PER_MINUTE`
   - DeepSeek: 可设置为 100+
   - OpenAI: 建议 20-50
   - 其他服务根据实际情况调整

## 🧪 测试配置

### 测试 1: 简单问题测试

```bash
python run.py --model logic-rag \
  --question "什么是人工智能？" \
  --corpus dataset/hotpotqa_corpus.json \
  --max-rounds 1 --top-k 3
```

### 测试 2: 检查 API 连接

创建测试脚本 `test_api.py`:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_API_BASE

load_dotenv()

# 测试 API 连接
client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_API_BASE:
    client_kwargs["base_url"] = OPENAI_API_BASE

client = OpenAI(**client_kwargs)

try:
    response = client.chat.completions.create(
        model=os.environ.get("DEFAULT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("✅ API 连接成功!")
    print(f"响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API 连接失败: {e}")
```

## ⚠️ 常见问题

### 1. API Key 无效

**错误**: `Error: Incorrect API key provided`

**解决**:
- 检查 `.env` 文件中的 API Key 是否正确
- 确保 API Key 没有多余的空格
- 确认 API Key 没有过期

### 2. 网络连接失败

**错误**: `Error: Connection timeout` 或 `Error: Unable to connect`

**解决**:
- 如果使用 OpenAI 官方，可能需要科学上网
- 推荐使用国内 API 服务（DeepSeek、SiliconFlow 等）
- 检查防火墙设置

### 3. 速率限制

**错误**: `Rate limit exceeded`

**解决**:
- 在 `config/config.py` 中降低 `CALLS_PER_MINUTE` 的值
- 使用没有严格速率限制的 API 服务
- 增加重试次数 `MAX_RETRIES`

### 4. 模型不支持

**错误**: `Error: Model not found`

**解决**:
- 检查 `DEFAULT_MODEL` 配置
- 确认模型名称与你使用的 API 服务匹配
- 查看各服务的模型列表文档

## 💰 费用估算

### 学习阶段（推荐配置）

使用 **DeepSeek**:
- 预计处理 100 个问题
- 每个问题平均 1000 tokens
- 总计约 100K tokens
- 费用约 ¥0.1-0.5

使用 **OpenAI gpt-4o-mini**:
- 相同用量约 $0.1-0.5 (¥1-3)

## 🎯 推荐配置

### 学习和实验（推荐）

**使用 DeepSeek**:
```bash
# .env
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com/v1

# config/config.py
DEFAULT_MODEL = "deepseek-chat"
CALLS_PER_MINUTE = 100
```

### 生产环境

**使用 OpenAI**:
```bash
# .env
OPENAI_API_KEY=sk-your-openai-key

# config/config.py
DEFAULT_MODEL = "gpt-4o-mini"
CALLS_PER_MINUTE = 50
```

## 📚 参考资源

- [OpenAI API 文档](https://platform.openai.com/docs)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [SiliconFlow API 文档](https://docs.siliconflow.cn/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

如有问题，请检查:
1. `.env` 文件是否在项目根目录
2. API Key 和 Base URL 是否正确
3. 网络连接是否正常
4. `config/config.py` 中的模型配置是否匹配
