"""W6-D3: HITL —— interrupt_before 哨卡 vs interrupt() 对话"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True) 

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ── 危险工具（示例版，不真删）──

def delete_file(path: str) ->str:
    """【危险】删除本地文件"""
    print(f"【工具真的被执行了】路径: {path}") 
    return f"文件已删除：{path}"

class S(TypedDict):
    messages: Annotated[list,add_messages]

llm=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
).bind_tools([delete_file])

def agent_node(state: S)->dict:
    return {"messages": [llm.invoke(state["messages"])]}

# ═══════ 方式一：interrupt_before（编译期哨卡）═══════
b1=StateGraph(S)
b1.add_node("agent",agent_node)
b1.add_node("tools",ToolNode([delete_file]))
b1.add_edge(START,"agent")
b1.add_conditional_edges("agent",tools_condition)
b1.add_edge("tools","agent")
g1=b1.compile(checkpointer=MemorySaver(),interrupt_before=["tools"])

cfg1={"configurable": {"thread_id":"hitl-1"}}
r1=g1.invoke({"messages":[("user","帮我删掉 /tem/a.txt")]},cfg1)
st = g1.get_state(cfg1)
print("① 暂停！下一步将执行:", st.next)
print("   消息数:", len(st.values["messages"])) 
print("   r1 也是 dict，消息数:", len(r1["messages"]))
# 人看了 state，决定"批准" → 不带新输入，用同一 config 继续
r1b=g1.invoke(None,cfg1)
print("   最终回复:", r1b["messages"][-1].content)

# ═══════ 方式二：interrupt()（节点内对话）═══════

def agent_ask(state:S)->dict:
    last=state["messages"][-1].content
    decision=interrupt({
        "question":f"模型想执行危险操作：{last}",
        "options":["approve","reject"],
    })
    if decision == "approve":
        return {"messages": [("assistant", "【已获批准】继续执行")]}
    return {"messages": [("assistant", "【已拒绝】取消删除操作")]}

b2 = StateGraph(S)
b2.add_node("agent", agent_ask)
b2.add_node("tools", ToolNode([delete_file]))
b2.add_edge(START, "agent")
b2.add_conditional_edges("agent", tools_condition)
b2.add_edge("tools", "agent")
g2 = b2.compile(checkpointer=MemorySaver())

cfg2 = {"configurable": {"thread_id": "hitl-2"}}

try:
    g2.invoke({"messages": [("user", "删掉 /tmp/b.txt")]}, cfg2)
    print("② 正常返回（没停？）")
except Exception as e:
    print("② 收到暂停信号:", type(e).__name__)        # GraphInterrupt = 停住了，不是错

st2 = g2.get_state(cfg2)
print("   停在:", st2.next)

# 人点了"拒绝" → Command(resume=...) 把值传回 interrupt() 处
out2 = g2.invoke(Command(resume="reject"), cfg2)
print("   结果:", out2["messages"][-1].content) 