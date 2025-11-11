import pytest
from c_data_acquisition.web_search_llm import dashscope_websearch
import logging
from dotenv import load_dotenv
import sys
import os
from b_provider_adapter.token_controller import TokenController

# === 测试配置 ===
TEST_PROVIDER = "aliyun-cn"
TEST_MODEL = "qwen3-max"  # 低成本模型，适合测试
MAX_OUTPUT_TOKENS = 2048  # 限制输出长度，节省token
CONCURRENT_REQUESTS = 5  # 并发请求数

# 配置日志，方便调试
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()
logger.info(f"DASHSCOPE_API_KEY: {os.getenv('DASHSCOPE_API_KEY')}")

@pytest.fixture(scope="session")
def api_key():
    """从环境变量加载API Key"""
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        pytest.skip("DASHSCOPE_API_KEY not set")
    return key


@pytest.fixture
def client(api_key):
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )


@pytest.fixture
def controller():
    return TokenController(
        tpm=5000,
        rpm=5,
        max_concurrent=5,
        provider=TEST_PROVIDER
    )

@pytest.mark.asyncio
async def test_dashscope_websearch(controller):
    query = "summarize the content in this link: https://globalnews.ca/news/11519953/donald-trump-tariffs-buy-canadian-mark-carney/"
    model = "qwen3-max"
    max_tokens = 50000
    extra_body = {
        "enable_search": True,
        "enable_source": True,
        "enable_citation": True,
        "search_options": {
            "forced_search": True,  # Force a web search
            "enable_search_extension": True,
        }
    }
    result = await dashscope_websearch(controller, query, model, max_tokens, extra_body)
    logging.info(f"✅ Test result: {result}")