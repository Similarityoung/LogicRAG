#!/usr/bin/env python
"""
测试 API 连接
用于验证 API 配置是否正确
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_MODEL

# 加载环境变量
load_dotenv()

def test_api_connection():
    """测试 API 连接"""
    print("=" * 50)
    print("  API 连接测试")
    print("=" * 50)
    print()

    # 检查 API Key
    if not OPENAI_API_KEY:
        print("❌ 错误: 未找到 OPENAI_API_KEY")
        print("   请在 .env 文件中设置 OPENAI_API_KEY")
        return False

    # 显示配置信息
    print("📋 当前配置:")
    print(f"   API Key: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
    print(f"   Base URL: {OPENAI_API_BASE or '(OpenAI 官方)'}")
    print(f"   模型: {DEFAULT_MODEL}")
    print()

    # 创建客户端
    try:
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_API_BASE:
            client_kwargs["base_url"] = OPENAI_API_BASE

        client = OpenAI(**client_kwargs)
        print("✅ 客户端创建成功")
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return False

    # 测试 API 调用
    print()
    print("🔄 测试 API 调用...")

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Hello! 请用中文回复: API 连接成功!"}
            ],
            max_tokens=50
        )

        print()
        print("✅ API 连接成功!")
        print()
        print("📝 响应信息:")
        print(f"   模型: {response.model}")
        print(f"   Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"   Completion Tokens: {response.usage.completion_tokens}")
        print(f"   Total Tokens: {response.usage.total_tokens}")
        print()
        print(f"💬 回复内容:")
        print(f"   {response.choices[0].message.content}")
        print()
        print("=" * 50)
        print("  配置正确，可以开始使用 LogicRAG!")
        print("=" * 50)
        return True

    except Exception as e:
        print()
        print("❌ API 调用失败!")
        print(f"   错误信息: {e}")
        print()
        print("💡 可能的原因:")
        print("   1. API Key 错误或已过期")
        print("   2. Base URL 配置错误")
        print("   3. 网络连接问题")
        print("   4. 模型名称不正确")
        print("   5. API 余额不足")
        print()
        print("📚 查看详细配置指南: API_CONFIG_GUIDE.md")
        return False

def test_simple_rag():
    """测试简单的 RAG 功能"""
    print()
    print("=" * 50)
    print("  测试简单 RAG 功能")
    print("=" * 50)
    print()

    # 检查语料库
    corpus_path = "dataset/hotpotqa_corpus.json"
    if not os.path.exists(corpus_path):
        print(f"⚠️  语料库文件不存在: {corpus_path}")
        print("   跳过 RAG 测试")
        return True

    try:
        from src.models.logic_rag import LogicRAG

        print("🔄 加载语料库...")
        rag = LogicRAG(corpus_path)
        rag.set_max_rounds(1)
        rag.set_top_k(3)
        print("✅ 语料库加载成功")
        print()

        # 测试简单问题
        test_question = "What is artificial intelligence?"
        print(f"❓ 测试问题: {test_question}")
        print()

        answer, contexts, rounds = rag.answer_question(test_question)

        print()
        print("✅ RAG 测试成功!")
        print()
        print("📝 结果:")
        print(f"   答案: {answer}")
        print(f"   轮次: {rounds}")
        print(f"   检索到的上下文数: {len(contexts)}")
        print()
        return True

    except Exception as e:
        print(f"❌ RAG 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试 API 连接
    api_success = test_api_connection()

    if api_success:
        # 询问是否测试 RAG
        print()
        response = input("是否测试 RAG 功能? (y/n): ")
        if response.lower() == 'y':
            test_simple_rag()
    else:
        print()
        print("⚠️  请先修复 API 配置后再试")
        sys.exit(1)
