from pprint import pprint
import os
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()
# 初始化客户端
api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=api_key)

# 搜索内容
response = tavily_client.search("天津天气怎么样")

# 读取搜索结果
results = response.get("results", [])

print(f"关键词：{response.get('query', 'N/A')}")
print(f"共找到 {len(results)} 条结果\n")

for index, item in enumerate(results[:5], start=1):
    title = item.get("title", "无标题")
    url = item.get("url", "无链接")
    score = item.get("score", "N/A")
    content = item.get("content", "")

    # 提取前 200 个字符作为摘要，去掉多余空白
    summary = " ".join(content.split())[:200]
    if len(content) > 200:
        summary += "..."

    print(f"===== 结果 {index} =====")
    print(f"标题: {title}")
    print(f"链接: {url}")
    print(f"相关度: {score}")
    print(f"摘要: {summary}\n")

print("===== 原始响应（仅保留结构） =====")
pprint({
    "query": response.get("query"),
    "count": len(results),
    "results": [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "score": item.get("score"),
            "content": " ".join(item.get("content", "").split())[:400]
        }
        for item in results[:3]
    ]
}, width=120, sort_dicts=False)