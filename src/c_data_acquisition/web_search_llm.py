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
from dashscope.aigc.generation import AioGeneration
import dashscope


async def dashscope_websearch(token_controller: TokenController, 
                              query: str, 
                              model: str = "qwen3-max", 
                              max_tokens: int = 2048, 
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
            messages=[{"role": "user", "content": ctx.prompt}],
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
    
async def dashscope_websearch_with_source(token_controller: TokenController, 
                                          query: str, 
                                          model: str = "qwen3-max", 
                                          max_tokens: int = 4096, 
                                          search_options={
                                              "enable_source": True,       # Must be enabled to use superscript annotations 
                                              "enable_citation": True,     # Enable superscript annotations 
                                              "citation_format": "[ref_<number>]", # Set the superscript style 
                                            },
                                          base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
                                          ) -> str:
    """
    使用Dashscope API进行Web搜索
    """
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    async with token_controller.acquire_slot(query, max_tokens) as ctx:
        # 在上下文内调用API
        response = await AioGeneration.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus",  # For a list of models, see https://www.alibabacloud.com/help/en/model-studio/models
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": query}],
            search_options=search_options,
            result_format="message",
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

