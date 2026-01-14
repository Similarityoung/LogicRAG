#!/bin/bash

# API 配置助手脚本
# 帮助快速设置 API 配置

echo "=========================================="
echo "  LogicRAG API 配置助手"
echo "=========================================="
echo ""

# 检查 .env 文件是否存在
if [ -f .env ]; then
    echo "⚠️  .env 文件已存在"
    read -p "是否要重新配置? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "配置已取消"
        exit 0
    fi
    echo "备份旧配置到 .env.backup"
    cp .env .env.backup
fi

# 选择 API 服务
echo "请选择你使用的 API 服务:"
echo "1) OpenAI 官方"
echo "2) DeepSeek (推荐，性价比高)"
echo "3) SiliconFlow (硅基流动)"
echo "4) Moonshot (月之暗面)"
echo "5) 智谱 AI (BigModel)"
echo "6) 其他兼容 OpenAI 的服务"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        api_base=""
        model_name="gpt-4o-mini"
        echo ""
        echo "配置 OpenAI 官方 API"
        echo "获取 API Key: https://platform.openai.com/api-keys"
        ;;
    2)
        api_base="https://api.deepseek.com/v1"
        model_name="deepseek-chat"
        echo ""
        echo "配置 DeepSeek API"
        echo "获取 API Key: https://platform.deepseek.com/"
        ;;
    3)
        api_base="https://api.siliconflow.cn/v1"
        model_name="Qwen/Qwen2.5-7B-Instruct"
        echo ""
        echo "配置 SiliconFlow API"
        echo "获取 API Key: https://siliconflow.cn/"
        ;;
    4)
        api_base="https://api.moonshot.cn/v1"
        model_name="moonshot-v1-8k"
        echo ""
        echo "配置 Moonshot API"
        echo "获取 API Key: https://platform.moonshot.cn/"
        ;;
    5)
        api_base="https://open.bigmodel.cn/api/paas/v4"
        model_name="glm-4"
        echo ""
        echo "配置智谱 AI API"
        echo "获取 API Key: https://open.bigmodel.cn/"
        ;;
    6)
        echo ""
        read -p "请输入 API Base URL: " api_base
        model_name="gpt-4o-mini"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

# 获取 API Key
echo ""
read -p "请输入你的 API Key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ API Key 不能为空"
    exit 1
fi

# 创建 .env 文件
echo "创建 .env 文件..."
cat > .env << EOF
# API Configuration
OPENAI_API_KEY=$api_key
EOF

if [ ! -z "$api_base" ]; then
    echo "OPENAI_API_BASE=$api_base" >> .env
fi

echo ""
echo "✅ .env 文件创建成功!"
echo ""

# 更新 config.py
if [ ! -z "$model_name" ]; then
    echo "正在更新 config/config.py 中的模型配置..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/DEFAULT_MODEL = \".*\"/DEFAULT_MODEL = \"$model_name\"/" config/config.py
    else
        # Linux
        sed -i "s/DEFAULT_MODEL = \".*\"/DEFAULT_MODEL = \"$model_name\"/" config/config.py
    fi
    echo "✅ 模型已设置为: $model_name"
fi

echo ""
echo "=========================================="
echo "  配置完成!"
echo "=========================================="
echo ""
echo "配置摘要:"
echo "  API Key: ${api_key:0:8}..."
echo "  Base URL: ${api_base:-"(OpenAI 官方)"}"
echo "  模型: $model_name"
echo ""
echo "下一步:"
echo "  1. 安装依赖: pip install -r requirements.txt"
echo "  2. 运行测试: python run.py --model logic-rag --question \"测试问题\" --corpus dataset/hotpotqa_corpus.json"
echo ""
echo "详细文档请查看: API_CONFIG_GUIDE.md"
echo ""
