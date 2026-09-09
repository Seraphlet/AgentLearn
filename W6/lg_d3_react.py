"""W6-D1: 给 W5 的 ReAct 图接上 MemorySaver —— 让图记住过去"""
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1.原样复用w5-d3的 react图

class AgentState(TypedDict):
    messages: Annotated[list,add_messages]


@tool
def add(a:int,b:int)->int:
    """计算两个整数之和"""
    return a+b

@tool
def multiply(a:int,b:int)->int:
    """计算两个整数之积"""
    return a*b

llm= ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
).bind_tools([add,multiply])

def agent_node(state: AgentState) ->dict:
    return {"messages": [llm.invoke(state["messages"])]}

def build():
    b = StateGraph(AgentState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode([add, multiply]))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)
    b.add_edge("tools", "agent")
    return b

# ── 2. 关键差异：同一张图，编译成两个版本 ──
graph_cp =build().compile(checkpointer=MemorySaver())#有存档
graph_nocp = build().compile() #无存档（对照组）

# ── 3. 验证 1：有 checkpoint → 跨 invoke 记忆 ──

config = {"configurable": {"thread_id": "demo-1"}}  # ⚠️ 每次都带同一 thread_id

r1 = graph_cp.invoke({"messages": [("user","我叫小明，记住我")]},config)
print("第一次:",r1["messages"][-1].content)

r2 = graph_cp.invoke({"messages": [("user","我叫什么名字")]},config)
print("第二次",r2["messages"][-1].content)

# ── 4. 验证 2：checkpoint 真的在累加消息（读档确认，不是猜）──

state=graph_cp.get_state(config)
print("消息总数",len(state.values["messages"])) # 期望 ≥ 4（2次提问+2次回答）

# ── 5. 验证 3：对照组 —— 忘了带 config 会怎样 ──
config2 = {"configurable": {"thread_id": "demo-2"}}   # ← 换槽位
r3 = graph_cp.invoke({"messages": [("user", "我叫什么名字？")]},config2)  # 没带 config
print("没带config:", r3["messages"][-1].content)   # 失忆（新档）

# ── 6. 验证 4：无 checkpoint 的图 ──

g1 = graph_nocp.invoke({"messages": [("user", "我叫小明")]})
g2 = graph_nocp.invoke({"messages": [("user", "我叫什么？")]})
print("无cp版消息数:", len(g2["messages"]))        # 期望 = 2（每次都从零开始，只有提问+回答）


