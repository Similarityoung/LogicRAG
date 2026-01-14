# 🚀 快速开始指南

本指南帮助你在 5 分钟内配置好环境并运行 LogicRAG。

## 📋 前置要求

- Python 3.7+
- 一个 API 服务账号

## 🔧 配置步骤

### 步骤 1: 安装依赖 (2分钟)

```bash
pip install -r requirements.txt
```

### 步骤 2: 配置 API (2分钟)

#### 方法 A: 使用配置脚本 (推荐)

```bash
bash setup_api.sh
```

按提示选择你的 API 服务并输入 API Key。

#### 方法 B: 手动配置

1. 复制配置模板:
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，选择一个服务配置:

**推荐: DeepSeek (性价比高)**
```bash
OPENAI_API_KEY=sk-your-deepseek-api-key
OPENAI_API_BASE=https://api.deepseek.com/v1
```

**或者: OpenAI 官方**
```bash
OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_API_BASE 不需要设置
```

3. 如果使用 DeepSeek，还需要修改 `config/config.py`:
```python
DEFAULT_MODEL = "deepseek-chat"
CALLS_PER_MINUTE = 100  # DeepSeek 速率限制较宽松
```

### 步骤 3: 测试配置 (1分钟)

```bash
python test_api.py
```

如果看到 "✅ API 连接成功!"，说明配置正确！

## 🎯 运行你的第一个问题

### 简单测试

```bash
python run.py --model logic-rag \
  --question "What is artificial intelligence?" \
  --corpus dataset/hotpotqa_corpus.json \
  --max-rounds 1 --top-k 3
```

### 复杂问题测试 (多跳推理)

```bash
python run.py --model logic-rag \
  --question "What is the mayor of the capital of France?" \
  --corpus dataset/hotpotqa_corpus.json \
  --max-rounds 5 --top-k 3
```

## 📊 批量评估

```bash
# 在 10 个问题上评估
python run.py --model logic-rag \
  --dataset dataset/hotpotqa.json \
  --corpus dataset/hotpotqa_corpus.json \
  --limit 10

# 查看结果
cat result/evaluation_results.json
```

## 🔍 观察输出

当你运行一个问题时，你会看到:

1. **LogicRAG answering:** - 显示问题
2. **Agentic retrieval at round X** - 当前的检索轮次
3. **current query:** - 当前正在处理的子问题
4. **Answer:** - 最终答案

## ⚙️ 参数说明

- `--max-rounds`: 最大检索轮次 (默认: 3)
  - 简单问题: 1-2 轮
  - 复杂多跳问题: 3-5 轮

- `--top-k`: 每轮检索的上下文数 (默认: 5)
  - 值越大，检索的信息越多
  - 值越小，速度越快

- `--limit`: 评估的问题数量 (默认: 20)
  - 设置为 0 表示处理所有问题

## 💡 常用命令速查

```bash
# 测试 API 连接
python test_api.py

# 单个问题
python run.py --model logic-rag --question "问题" --corpus dataset/hotpotqa_corpus.json

# 小规模评估
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --limit 5

# 完整评估
python run.py --model logic-rag --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --limit 0
```

## ❓ 遇到问题?

### API 连接失败
```bash
# 检查 .env 文件
cat .env

# 重新测试
python test_api.py
```

### 找不到语料库
```bash
# 检查数据集文件
ls -lh dataset/
```

### 详细帮助
查看 [API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md) 了解更多配置选项。

## 🎉 下一步

配置完成后，你可以:
1. 阅读 [API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md) 了解更多 API 服务选项
2. 查看 [LEARNING_PLAN.md](LEARNING_PLAN.md) 开始系统学习
3. 运行实验并分析结果

## 📚 推荐学习路径

1. ✅ 运行几个简单问题，熟悉系统
2. 📖 阅读 `src/models/logic_rag.py` 理解核心算法
3. 🔍 使用调试模式观察中间结果
4. 📊 运行评估并分析结果
5. 📝 记录学习笔记

祝你学习愉快！🎓
