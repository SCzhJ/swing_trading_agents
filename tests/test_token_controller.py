# test_real_api_parallel.py
import asyncio
import os
import pytest
from openai import AsyncOpenAI
from b_provider_adapter.token_controller import TokenController
from a_utils.config_setup import get_project_root
import logging
from dotenv import load_dotenv
import sys

# === 测试配置 ===
TEST_PROVIDER = "kimi"
TEST_MODEL = "kimi-k2-0905-preview"  # 低成本模型，适合测试
MAX_OUTPUT_TOKENS = 50  # 限制输出长度，节省token
CONCURRENT_REQUESTS = 5  # 并发请求数

# 配置日志，方便调试
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Project root: {get_project_root()}")
sys.path.append(str(get_project_root()))
load_dotenv()
logger.info(f"MOONSHOT_API_KEY: {os.getenv('MOONSHOT_API_KEY')}")

# === Fixtures ===

@pytest.fixture(scope="session")
def api_key():
    """从环境变量加载API Key"""
    key = os.getenv("MOONSHOT_API_KEY")
    if not key:
        pytest.skip("MOONSHOT_API_KEY not set")
    return key


@pytest.fixture
def client(api_key):
    """创建MOONSHOT客户端"""
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )


@pytest.fixture
def strict_controller():
    """
    创建严格限流的控制器（低阈值，便于观察限流效果）

    设置：
    - TPM: 500 (故意设低，容易触发等待)
    - RPM: 5 (每分钟最多3个真实请求)
    - Max Concurrent: 2 (最多同时2个请求)
    """
    return TokenController(
        tpm=1000,  # 故意设低，单次请求约100 tokens
        rpm=20,    # 每分钟最多3个真实请求
        max_concurrent=20,  # 最多2个并发
        provider=TEST_PROVIDER
    )


@pytest.fixture
def loose_controller():
    """
    创建宽松限流的控制器（高阈值，确保不触发限流）

    用于对比测试
    """
    return TokenController(
        tpm=1000000,
        rpm=25,
        max_concurrent=30,
        provider=TEST_PROVIDER
    )

# === 测试用例 ===


# test_token_controller.py (简化版)

# 保持旧测试兼容（直接调用底层API）
# @pytest.mark.asyncio
# async def test_manual_api_still_works(client, strict_controller):
#     """验证旧的手动API仍然可用"""
#     call_id = await strict_controller.wait_before_call_if_needed("Test", 100)
    
#     try:
#         response = await client.chat.completions.create(
#             model=TEST_MODEL,
#             messages=[{"role": "user", "content": "Test"}],
#             max_tokens=100
#         )
        
#         await strict_controller.wait_after_call_if_needed(
#             call_id,
#             response.usage.prompt_tokens,
#             response.usage.completion_tokens
#         )
        
#         assert True  # 成功
#     except Exception:
#         await strict_controller.cleanup_call(call_id)
#         raise

# @pytest.mark.asyncio
# async def test_context_manager_api(client, strict_controller):
#     """
#     测试上下文管理器API
    
#     更灵活，适合复杂场景
#     """
    
#     async with strict_controller.acquire_slot("What is AI?", 100) as ctx:
#         # 在上下文内调用API
#         response = await client.chat.completions.create(
#             model=TEST_MODEL,
#             messages=[{"role": "user", "content": ctx.prompt}],
#             max_tokens=ctx.max_output_token
#         )
        
#         # 设置结果
#         ctx.set_result(
#             input_tokens=response.usage.prompt_tokens,
#             output_tokens=response.usage.completion_tokens,
#             result=response.choices[0].message.content
#         )
        
#         # 获取结果
#         result = ctx.result
    
#     assert len(result) > 0
#     logger.info(f"✅ Context manager result: {result}")

@pytest.mark.asyncio
async def test_llm_caller(client, loose_controller):
    """测试LLM调用器"""
    async def call_llm(prompt: str, max_tokens=10000):
        async with loose_controller.acquire_slot(prompt, max_tokens) as ctx:
            # 在上下文内调用API
            response = await client.chat.completions.create(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": ctx.prompt}],
                max_tokens=ctx.max_output_token
            )

            # 设置结果
            ctx.set_result(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                result=response.choices[0].message.content
            )

            # 获取结果
            result = ctx.result
        return result
    prompts = [
        "what is the capital of Afghanistan?",
        "what is the capital of Algeria?",
        "what is the capital of Argentina?",
        "what is the capital of Australia?",
        "what is the capital of Brazil?",
        "what is the capital of Canada?",
        "what is the capital of China?",
        "what is the capital of Colombia?",
        "what is the capital of Egypt?",
        "what is the capital of Ethiopia?",
        "what is the capital of France?",
        "what is the capital of Germany?",
        "what is the capital of Ghana?",
        "what is the capital of India?",
        "what is the capital of Indonesia?",
        "what is the capital of Iran?",
        "what is the capital of Iraq?",
        "what is the capital of Italy?",
        "what is the capital of Japan?",
        "what is the capital of Kenya?",
        "what is the capital of Malaysia?",
        "what is the capital of Mexico?",
        "what is the capital of Morocco?",
        "what is the capital of Nepal?",
        "what is the capital of Netherlands?",
        "what is the capital of New Zealand?",
        "what is the capital of Nigeria?",
        "what is the capital of North Korea?",
        "what is the capital of Norway?",
        "what is the capital of Pakistan?",
        "what is the capital of Peru?",
        "what is the capital of Philippines?",
        "what is the capital of Poland?",
        "what is the capital of Russia?",
        "what is the capital of Saudi Arabia?",
        "what is the capital of South Africa?",
        "what is the capital of South Korea?",
        "what is the capital of Spain?",
        "what is the capital of Sweden?",
        "what is the capital of Switzerland?",
        "what is the capital of Thailand?",
        "what is the capital of Uganda?",
        "what is the capital of Ukraine?",
        "what is the capital of United Kingdom?",
        "what is the capital of United States?",
        "what is the capital of Venezuela?",
        "what is the capital of Vietnam?",
        "what is the capital of Yemen?",
        "what is the capital of Zambia?",
        "what is the capital of Zimbabwe?",
    ]
    results = await asyncio.gather(*[call_llm(prompt) for prompt in prompts])
    assert all(len(result) > 0 for result in results)
    for prompt, result in zip(prompts, results):
        logger.info(f"✅ LLM caller result for prompt '{prompt}': {result}")


# === 辅助调试命令 ===

if __name__ == "__main__":
    """直接运行文件进行手动调试"""
    # 设置环境变量

    # 运行测试
    asyncio.run(test_single_call_success(
        client=AsyncOpenAI(),
        strict_controller=TokenController(500, 3, 2, "debug")
    ))
