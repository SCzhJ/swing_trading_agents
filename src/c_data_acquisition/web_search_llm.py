from openai import OpenAI
# from .config import get_config # no longer needed
import os
import dotenv
import json
from datetime import datetime, timedelta
import time
import random

dotenv.load_dotenv()
# 使用 Moonshot API Key
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

def chat_with_websearch(messages, model="kimi-k2-0905-preview"):
    """处理带有联网搜索的对话流程"""
    finish_reason = None
    final_response = None

    client = OpenAI(
        api_key=MOONSHOT_API_KEY,
        base_url="https://api.moonshot.cn/v1"  # Moonshot API 端点
    )
    
    while finish_reason is None or finish_reason == "tool_calls":
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            tools=[
                {
                    "type": "builtin_function",
                    "function": {
                        "name": "$web_search",
                    },
                }
            ]
        )
        
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        
        if finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(tool_call.function.arguments)
                
                if tool_call_name == "$web_search":
                    # Moonshot 内置的 $web_search 只需要返回参数即可
                    tool_result = tool_call_arguments
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),
                })
        else:
            final_response = choice.message.content
    
    return final_response

def chat_with_websearch_and_retry(messages, model="kimi-k2-0905-preview", max_attempt=5):
    for attempt in range(1, max_attempt + 1):
        try:
            return chat_with_websearch(messages, model)
        except openai.APITimeoutError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"[{attempt}/{max_attempt}] 请求超时，{wait:.1f}s 后重试…")
            time.sleep(wait)
    raise RuntimeError("多次重试后仍无法连接 Moonshot API")

def get_stock_news_kimi(symbol, start_date, end_date):
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_date - timedelta(days=look_back_days))
    # get the month name in english 
    month_name1 = start_date.strftime("%B")
    month_name2 = curr_date.strftime("%B")
    date_range = f"{start_date.strftime('%Y')} {month_name1} {start_date.strftime('%d')} to {curr_date.strftime('%Y')} {month_name2} {curr_date.strftime('%d')}"
    messages = [
        {
            "role": "system", 
            "content": "You are a professional financial analyst skilled in searching and analyzing stock-related information."
        },
        {
            "role": "user",
            "content": f"Can you search Social Media News for stock \"{symbol}\" between {date_range}? Make sure you only get the data posted during that period."
        }
    ]

    response = chat_with_websearch_and_retry(messages, "kimi-k2-0905-preview")
    return response

def get_global_news_kimi(curr_date, look_back_days=7, limit=5):
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_date - timedelta(days=look_back_days))
    # get the month name in english 
    month_name1 = start_date.strftime("%B")
    month_name2 = curr_date.strftime("%B")
    date_range = f"{start_date.strftime('%Y')} {month_name1} {start_date.strftime('%d')} to {curr_date.strftime('%Y')} {month_name2} {curr_date.strftime('%d')}"
    
    messages = [
        {
            "role": "system",
            "content": "You are a professional macroeconomic analyst skilled in searching and analyzing global news."
        },
        {
            "role": "user",
            "content": f"Can you search global or macroeconomics news between {date_range} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles."
        }
    ]

    response = chat_with_websearch_and_retry(messages, "kimi-k2-0905-preview")
    return response


if __name__ == "__main__":
    # print(get_stock_news_kimi("TSLA", "2025-10-01", "2025-10-07"))
    print(get_global_news_kimi("2025-10-07"))
