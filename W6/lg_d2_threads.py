"""W6-D2 步骤1：thread_id 多会话隔离 —— 一张图，两个用户，互不干扰"""

import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list,add_messages]

@tool
def add(a: int,b: int)->int:
    """计算两个整数之和"""
    return a+b

@tool
def multiply(a:int,b: int)-> int:
    """计算两数之积"""
    return a*b

llm=ChatOpenAI(
     model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

llm_with_tools=llm.bind_tools([add,multiply])

def agent_node(state: AgentState)->dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder= StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tools",ToolNode([add,multiply]))
builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",tools_condition)
builder.add_edge("tools","agent")

# 同一张图，编译时只传一次 checkpointer；会话隔离靠 invoke 时传不同 thread_id
graph = builder.compile(checkpointer=MemorySaver())

# ── 两个用户，两个 thread ──

alice={"configurable": {"thread_id":"alice-001"}}
bob ={"configurable":{"thread_id":"bob-001"}}

graph.invoke({"messages":[("user","我叫爱丽丝，我喜欢猫")]},alice)
graph.invoke({"messages": [("user", "我是鲍勃")]}, bob)

#各自问各自己的信息
r_a =graph.invoke({"messages":[("user","我叫什么，我喜欢什么？")]},alice)
r_b=graph.invoke({"messages":[("user","我叫什么？我喜欢什么？")]},bob)
print("Alice记得:",r_a["messages"][-1].content)
print("Bob  记得:",r_b["messages"][-1].content)
