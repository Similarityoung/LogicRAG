# 配置更新总结

## 📝 本次更新内容

为了支持第三方 API 服务，我对代码进行了以下修改：

## 🔧 代码修改

### 1. [config/config.py](config/config.py)
**修改内容**:
- 添加了 `OPENAI_API_BASE` 配置项，支持自定义 API 端点
- 添加了 `OPENAI_API_TYPE` 配置项，用于标识不同的 API 服务

**作用**: 允许用户使用兼容 OpenAI 的第三方 API 服务

### 2. [src/utils/utils.py](src/utils/utils.py)
**修改内容**:
- 修改了 OpenAI 客户端初始化逻辑
- 当 `OPENAI_API_BASE` 有值时，自动使用自定义的 base_url

**作用**: 使代码能够连接到第三方 API 端点

## 📄 新增文件

### 1. [.env.example](.env.example)
**用途**: API 配置模板文件

**包含内容**:
- OpenAI 官方配置
- DeepSeek 配置
- SiliconFlow 配置
- Moonshot、智谱 AI、阿里云百炼等其他服务配置

**使用方法**:
```bash
cp .env.example .env
# 然后编辑 .env 文件，填入你的 API Key
```

### 2. [API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md)
**用途**: 详细的 API 配置指南

**包含内容**:
- 支持的所有 API 服务介绍
- 每个服务的配置方法
- 费用对比
- 常见问题解决
- 推荐配置

### 3. [setup_api.sh](setup_api.sh)
**用途**: 自动化配置脚本

**功能**:
- 交互式选择 API 服务
- 自动生成 .env 文件
- 自动更新 config.py 中的模型配置

**使用方法**:
```bash
bash setup_api.sh
```

### 4. [test_api.py](test_api.py)
**用途**: API 连接测试脚本

**功能**:
- 验证 API 配置是否正确
- 测试 API 连接
- 可选的简单 RAG 功能测试

**使用方法**:
```bash
python test_api.py
```

### 5. [QUICKSTART.md](QUICKSTART.md)
**用途**: 5分钟快速开始指南

**包含内容**:
- 简化的配置步骤
- 常用命令
- 快速测试示例

## 🎯 推荐使用流程

### 方式 1: 使用自动化脚本（最简单）

```bash
# 1. 运行配置脚本
bash setup_api.sh

# 2. 测试连接
python test_api.py

# 3. 运行示例
python run.py --model logic-rag \
  --question "测试问题" \
  --corpus dataset/hotpotqa_corpus.json
```

### 方式 2: 手动配置（更灵活）

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑 .env 文件，填入 API Key

# 3. 如果使用第三方服务，修改 config/config.py

# 4. 测试连接
python test_api.py
```

## 🌟 推荐配置

### 学习和实验

**使用 DeepSeek**:
- ✅ 性价比极高（约 ¥1/百万 tokens）
- ✅ 完全兼容 OpenAI API
- ✅ 中文支持好
- ✅ 速率限制宽松

配置:
```bash
# .env
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com/v1

# config/config.py
DEFAULT_MODEL = "deepseek-chat"
CALLS_PER_MINUTE = 100
```

## 📊 API 服务对比

| 服务 | 费用 | 稳定性 | 推荐度 | 适用场景 |
|------|------|--------|--------|----------|
| DeepSeek | 低 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 学习、实验 |
| OpenAI | 高 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 生产环境 |
| SiliconFlow | 低 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 开源模型测试 |
| Moonshot | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中文场景 |
| 智谱 AI | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中文场景 |

## ⚠️ 注意事项

1. **.env 文件不会被提交到 Git** (已在 .gitignore 中)
2. **首次使用建议先运行 test_api.py 测试**
3. **不同服务的模型名称可能不同**，注意修改 config/config.py
4. **速率限制因服务而异**，建议根据实际情况调整 CALLS_PER_MINUTE

## 🔗 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md) - 详细配置指南
- [LEARNING_PLAN.md](LEARNING_PLAN.md) - 学习计划
- [CLAUDE.md](CLAUDE.md) - 项目架构说明

## 🎓 下一步

配置完成后，你可以:
1. 运行 `python test_api.py` 确认配置正确
2. 运行 `python run.py --help` 查看所有选项
3. 按照 [LEARNING_PLAN.md](LEARNING_PLAN.md) 开始学习

---

如有问题，请查看 [API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md) 的常见问题部分。
