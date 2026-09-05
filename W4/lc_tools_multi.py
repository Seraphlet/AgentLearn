import os, json,sys
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env",override=True)

#----把自研的WebSearch的"芯"装进@tool里，闭包注入client

def make_web_search_tool(client=None):
    """工厂函数：把有状态的 client 闭包捕获，返回 @tool 工具"""
    if client is None:
        from tavily import TavilyClient
        client =TavilyClient(
            api_key=os.environ.get("TAVILY_API_KEY"),)
    @tool
    def search_web(query:str, max_results:int=3)->str:
        """搜索网页信息。当用户询问需要联网查询的问题、最新动态时，调用此工具。参数query: 搜索关键词; max_results:返回条数(默认三条)"""
        result=client.search(query,max_results=max_results)
        return json.dumps(result,ensure_ascii=False)
    return search_web

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。参数 city: 城市名。"""
    return f"{city} 今天晴，26℃"

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#回到根目录，硬编码
from memory.memory_manager import MemoryManager
from W4.retriever import MemoryRetriever
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    from langchain_core.tools.retriever import create_retriever_tool
mm=MemoryManager(window=8,max_tokens=1500,keep=4)
mm.add({"role":"user","content":"用户喜欢Deekseek，预算2000元"},persist=True)
mm.add({"role": "user", "content": "用户叫小明，喜欢蓝色"}, persist=True)
memory_retriever=MemoryRetriever(mm=mm,k=3)
memory_tool=create_retriever_tool(memory_retriever,name="memory_retriever",description="从长期记忆检索用户偏好、历史事实。当用户问起以前说过/聊过的事情时调用")




search_web = make_web_search_tool()          # 闭包捕获 TavilyClient
tools = [get_weather, search_web, memory_tool]
tools_by_name = {t.name: t for t in tools}

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
).bind_tools(tools)

def _normalize_args(args: dict) -> dict:
    for k, v in list(args.items()):
        while isinstance(v, dict) and v:
            v = next(iter(v.values()))  
        args[k] = v
    return args

MAX_ROUNDS = 3

def run(query:str, verbose:bool=True):
    messages=[HumanMessage(query)]
    ai=model.invoke(messages)
    messages.append(ai)

    round_=0
    while ai.tool_calls and round_<MAX_ROUNDS:
        round_ += 1

        if verbose:
            print(f"\n[第{round_}轮] 模型发起 {len(ai.tool_calls)} 个工具调用:")
            for call in ai.tool_calls:
                print(f"  工具: {call['name']}  参数: {call['args']}  解析后: {_normalize_args(call['args'])}")
        for call in ai.tool_calls:
            args=_normalize_args(call["args"])
            result=tools_by_name[call["name"]].invoke(args)
            messages.append(ToolMessage(content=str(result),tool_call_id=call["id"]))
            if verbose:
                print(f"  工具返回 {str(result)[:80]}...")
        ai = model.invoke(messages)             # 带着结果再问模型
        messages.append(ai)
    if ai.tool_calls:
        print(f"\n[第{round_}轮] 模型发起 {len(ai.tool_calls)} 个工具调用，但已达到最大轮数 {MAX_ROUNDS}，停止调用。")
    return ai.content
if __name__ == "__main__":
    print("\n=== 测试1：天气 ===")
    print(run("北京天气怎么样?"))
    print("\n=== 测试2：联网搜索 ===")
    print(run("今天的最新AI新闻是什么?"))
    print("\n========== 测试3：长期记忆 ==========")
    print(run("用户喜欢什么模型，预算多少?"))