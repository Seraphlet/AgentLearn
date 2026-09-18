"""test_worker_tool.py v2 —— 检测工具是否真被调用
v1 的 bug: 工具里 print() 走 stdout, 不进 messages; v1 却在 content 里找 print 的字符串 → 必然假阴性
v2: 用两个独立通道 —— 事件记录(CALLS) + 状态里的 ToolMessage 实例
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

CALLS = []          # ★ 单一事实来源: 工具每次被调用就记一笔(不依赖 stdout, 不依赖 content)

@tool
def web_search(query: str) -> str:
    """搜索网页资料, 返回要点。"""
    CALLS.append(query)
    print(f"  >>> [web_search 真身被调用] query={query!r}")
    return "LangGraph 是 LangChain 团队开发的、用于构建有状态多步 Agent 的框架。"

llm = ChatOpenAI(model="deepseek-v4-flash", base_url="https://api.deepseek.com",
                 api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
agent = create_react_agent(llm, [web_search], prompt="你是研究员, 用 web_search 查资料。")

out = agent.invoke({"messages": [("user", "帮我查一下 LangGraph 是什么")]})

# 通道A: 事件记录(最可靠)
print(f"\n通道A · 事件记录 CALLS = {len(CALLS)} 次")
for i, q in enumerate(CALLS, 1):
    print(f"  #{i} {q!r}")

# 通道B: 状态里的 ToolMessage
tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
print(f"通道B · 状态中 ToolMessage = {len(tool_msgs)} 条")

same = len(CALLS) == len(tool_msgs)
print(f"\n工具真的被调用了吗 → {'✅ 是' if CALLS else '❌ 否'}")
print(f"两通道一致吗 → {'✅ 一致' if same else f'⚠️ 不一致: 有 {len(CALLS)} 次调用但状态里只有 {len(tool_msgs)} 条'}")
print(f"消息类型序列: {[type(m).__name__ for m in out['messages']]}")