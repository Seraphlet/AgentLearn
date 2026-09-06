"""W5-D4: 一张 ReAct 图，四种看法（updates / values / messages / events）"""

import os, asyncio,time
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list,add_messages]

@tool
def add(a: int,b: int) -> int:
    """计算两个整数之和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数之积"""
    return a * b

tools=[add,multiply]

llm=ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
llm_with_tools=llm.bind_tools(tools)

def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder=StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",tools_condition)
builder.add_edge("tools","agent")
graph=builder.compile()

TEST_INPUT = {"messages": [("user", "帮我算 (15 + 27) 乘以 2")]}

# ── 看法①：updates（看结构）──

print("=== ① updates：谁改了什么 ===")
for event in graph.stream(TEST_INPUT, stream_mode="updates"):
    print("  ", event)

# ── 看法②：values（看全貌）──
print("\n=== ② values：每一步完整 state ===")
for i, snap in enumerate(graph.stream(TEST_INPUT, stream_mode="values")):
    print(f"  第{i}步 消息数={len(snap['messages'])}  最后一条={snap['messages'][-1].__class__.__name__}")


# ── 看法③：messages（打字机）──
print("\n=== ③ messages：token 级打字机 ===")
for chunk, metadata in graph.stream(TEST_INPUT, stream_mode="messages"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
        time.sleep(0.1)
print("\n")

# ── 看法④：astream_events（最细粒度，异步）──
print("=== ④ astream_events：token + 工具事件 ===")
async def run_events():
    async for ev in graph.astream_events(TEST_INPUT, version="v2"):
        if ev["event"] == "on_chat_model_stream":
            data = ev["data"]["chunk"].content
            if data:
                print(data, end="", flush=True)
        elif ev["event"] == "on_tool_start":
            print(f"\n  [tool开始] {ev['name']}")
        elif ev["event"] == "on_tool_end":
            print(f"  [tool结束] {ev['name']} → {ev['data'].get('output')}")

asyncio.run(run_events())
print("\n=== 完成 ===")