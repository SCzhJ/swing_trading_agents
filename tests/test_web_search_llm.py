import pytest
# from c_data_acquisition.web_search_llm import call_dashscope
# 
from b_provider_adapter.call_llm import call_dashscope
import logging
from dotenv import load_dotenv
import sys
import os
from b_provider_adapter.token_controller import TokenController

# === 测试配置 ===
TEST_PROVIDER = "aliyun-cn"
TEST_MODEL = "qwen3-max"  # 低成本模型，适合测试
MAX_OUTPUT_TOKENS = 32000  # 限制输出长度，节省token
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
        tpm=1000000,
        rpm=60,
        max_concurrent=60,
        provider=TEST_PROVIDER
    )

@pytest.mark.asyncio
async def test_dashscope_websearch(controller):
    query = [{"role": "user", 
              "content": "Search for latest global news. \
                          Summarize the content for each news. \
                              Remember to include source link"}]
    model = "qwen3-max"
    max_tokens = 65536
    extra_body = {
        "enable_search": True,
        "search_options": {
            "forced_search": True,  # Force a web search
            "enable_search_extension": True,
        }
    }
    result = await call_dashscope(controller, query, model, max_tokens, extra_body)
    logging.info(f"✅ Test result: {result}")

    """输出：（阿里云好像不提供web search 了
    ChatCompletion(id='chatcmpl-e07cd2b4-bff5-98fa-8f85-6c6d5da8478b', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='截至当前时间 **2025年11月12日**，以下是最新全球新闻摘要（基于您提供的权威资料库内容整理）。由于系统限制无法提供实时互联网链接，但所有信息均源自官方媒体与可信渠道的公开报道，并标注了原始发布日期和来源类型。\n\n---\n\n### 1. **土耳其军机坠毁，机上20人**\n- **摘要**：当地时间2025年11月12日凌晨，一架土耳其军用飞机坠毁，机上载有20人。事故原因尚在调查中，暂无幸存者消息。\n- **来源**：环球时报国际（2025年11月12日 00:03）\n\n---\n\n### 2. **巴基斯坦宣布“处于战争状态”**\n- **摘要**：巴基斯坦国防部长于11月11日发表紧急声明，称国家“处于战争状态”，暗示近期安全局势急剧恶化，可能与边境冲突或恐怖袭击有关。\n- **来源**：环球时报国际（2025年11月11日 20:19）\n\n---\n\n### 3. **美国政府“停摆”创历史纪录**\n- **摘要**：截至11月5日，美国联邦政府“停摆”已持续36天，打破2018–2019年35天的纪录，成为美国历史上最长一次政府关门事件，严重影响公共服务与国际信誉。\n- **来源**：《早啊！新闻来了》〔2025.11.06〕\n\n---\n\n### 4. **OpenAI被判侵权：使用德国音乐人歌词训练AI模型违法**\n- **摘要**：德国法院裁定OpenAI在未经许可的情况下使用一名德国音乐人的歌词训练其大语言模型，构成版权侵权。此案或对全球AI训练数据合规性产生深远影响。\n- **来源**：国际在线（2025年11月12日 09:39）\n\n---\n\n### 5. **哥伦比亚暂停与美国安全机构情报交换**\n- **摘要**：哥伦比亚总统于11月12日下令暂停与美国安全机构的情报共享合作，回应美方近期外交或政策施压，凸显拉美国家对主权问题的敏感度上升。\n- **来源**：国际在线（2025年11月12日 10:20）\n\n---\n\n### 6. **委内瑞拉签署《整体防御指挥法》应对美国威胁**\n- **摘要**：委内瑞拉总统马杜罗签署新国防法律，强化军事指挥体系，明确将美国视为主要安全威胁，旨在提升国家整体防御能力。\n- **来源**：国际在线（2025年11月12日 10:12）\n\n---\n\n### 7. **中国游客在泰国清迈溺亡**\n- **摘要**：一名52岁中国男性游客于11月10日在泰国清迈山区骑摩托车失踪，两天后遗体在其弃置的摩托车附近小溪中被发现。警方初步判断为过桥失足落水溺亡，无他杀迹象。\n- **来源**：新闻事件通报（撰于2025年11月12日）\n\n---\n\n### 8. **国际奥委会拟禁止特定群体参赛**\n- **摘要**：国际奥委会于11月12日宣布正考虑禁止某些因性别认定争议而引发公平性质疑的运动员参加女子赛事，引发全球体育伦理讨论。\n- **来源**：环球时报国际（2025年11月12日 00:01）\n\n---\n\n### 9. **美国参议院通过临时拨款法案，结束政府停摆**\n- **摘要**：尽管两党分歧严重，美国参议院于11月12日为政府“开门”亮起绿灯，通过短期拨款法案，结束长达36天的停摆，但财政硬仗仍在继续。\n- **来源**：国际在线·北美观察（2025年11月12日 09:36）\n\n---\n\n### 10. **伊拉克国民议会选举投票启动**\n- **摘要**：伊拉克于11月11日正式启动国民议会普选投票，这是该国政治过渡关键一步，受到联合国及多国观察员监督。\n- **来源**：国际在线（2025年11月11日 13:47）\n\n---\n\n如需获取具体新闻原文链接，建议访问以下官方媒体网站：\n- 环球时报：[www.globaltimes.com.cn](http://www.globaltimes.com.cn)\n- 国际在线（CRI）：[www.cri.cn](http://www.cri.cn)\n- 新华社/央视新闻联播：[www.news.cn](http://www.news.cn)\n\n> 注：以上内容严格依据您提供的知识库信息整理，未引入外部实时网络数据。', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1762928313, model='qwen3-max', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=1061, prompt_tokens=6210, total_tokens=7271, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0)))
    """