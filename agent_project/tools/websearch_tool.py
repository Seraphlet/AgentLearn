from .base import BaseTool
import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
client = None 
# 初始化客户端
class websearchTool(BaseTool):
    name = "search_web"
    description = (
        "提供联网搜索能力，搜索网页信息(天气/新闻/最新版本/不确定的事实)，返回JSON："
        "query、count、results(每条的title/url/score/content摘要)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回条数，默认5"},
            "search_depth": {"type": "string",
                             "enum": ["basic", "advanced"],
                             "description": "搜索深度，advanced更准但更慢，默认basic"},
        },
        "required": ["query"],
    }

    def __int__(self,**kwargs):
        super().__init__(**kwargs)
        self._client = kwargs.get("client")  

    def _get_client(self):
        if self._client is None:              # 仍然是惰性，但状态归实例管
            key = os.getenv("TAVILY_API_KEY")
            if not key:
                raise RuntimeError("缺少 TAVILY_API_KEY，请在 .env 配置")
            self._client = TavilyClient(api_key=key)
        return self._client
    def execute(self,query: str, max_results: int = 5,search_depth: str = "basic") -> str:
        try:
           resp=self._get_client.search(query, max_results=max_results, search_depth=search_depth)
        except Exception as e:
            return f"搜索失败: {type(e).__name__}: {e}"
        results = resp.get("results", [])
        return json.dumps({
            "query": resp.get("query"),
            "count": len(results),
            "results": [
                {"title": item.get("title", ""), "url": item.get("url", ""),
                 "score": item.get("score"), 
                 "content": " ".join(item.get("content", "").split())[:300]}
                for item in results[:max_results]
            ],
        }, ensure_ascii=False, indent=2)
