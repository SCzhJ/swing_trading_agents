from pathlib import Path
project_root = None
for parent in [Path.cwd(), *Path.cwd().parents]:
    if (parent / "pyproject.toml").exists():
        project_root = parent
        break
if project_root is None:
    raise FileNotFoundError("pyproject.toml 未找到，无法确定项目根目录")

import sys
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import os
from openai import AsyncOpenAI
from b_provider_adapter.token_controller import TokenController
from typing import List, Dict, Any


async def call_dashscope(token_controller: TokenController, 
                         query: List[Dict[str, str]] = [{"role": "user", 
                                                         "content": "Search for latest global news. \
                                                             Summarize the content for each news. \
                                                                 Remember to include source link"}],
                         model: str = "qwen-flash", 
                         max_tokens: int = 32768, 
                         extra_body: dict = {
                             "enable_search": True,
                             "search_options": {
                                 "forced_search": True,  # Force a web search
                                 "enable_search_extension": True,
                                 }
                             },
                         base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
                         ) -> str:
    """
    使用Dashscope API进行Web搜索
    """
    client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=base_url
    )
    async with token_controller.acquire_slot(query, max_tokens) as ctx:
        # 在上下文内调用API
        response = await client.chat.completions.create(
            model=model,
            messages=query,
            max_tokens=ctx.max_output_token,
            extra_body=extra_body
        )
        
        # 设置结果
        ctx.set_result(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            result=response
        )
        
        # 获取结果
        result = ctx.result
    return result

