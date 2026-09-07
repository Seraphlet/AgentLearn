"""W5-D5: Pydantic state + reducer 机制——为 D6 MemoryManager 打底"""
"""W5-D5 收尾：自定义 reducer —— 累加 vs 覆盖的终极对照"""
import os
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ── 1. Pydantic state：messages 追加 / summary 覆盖 / 计数 ──

class RichState(BaseModel):
    messages: Annotated[list[Any],add_messages]=[]
    summary: str =Field(default="",description="对话摘要")
    tool_use_count: int =0

# ── 自定义 reducer：计数器（current + new）──
def increment(current: int,new: int)->int:
    return current+new

# ── 自定义 reducer：日志拼接 ──
def add_log(current: list[str],new: list[str])-> list[str]:
    return current+new

class CounterState(BaseModel):
    visits: Annotated[int,increment] = 0 # 有 reducer → 累加
    log: Annotated[list[str],add_log] = []  # 有 reducer → 拼接
    plain: int = 0 

def step1(state:CounterState) -> dict:
    return {"visits": 1, "log": ["step1 执行了"], "plain": 1}

def step2(state: CounterState) -> dict:
    return {"visits": 1, "log": ["step2 执行了"], "plain": 2}

c = StateGraph(CounterState)
c.add_node("step1", step1)
c.add_node("step2", step2)
c.add_edge(START, "step1")
c.add_edge("step1", "step2")
c.add_edge("step2", END)
gg = c.compile()

final = gg.invoke({})
print("visits:", final["visits"])   # 期望 2  （step1 +1，step2 再 +1 → 累加）
print("log:", final["log"])          # 期望 ['step1 执行了', 'step2 执行了']
print("plain:", final["plain"])      # 期望 2  （step2 覆盖 step1 的 1）





# ——————2.工具+LLM
@tool
def getweather(city: str) -> str:
    """查询城市天气(模拟)"""
    return f"{city}:🌤晴，25 ℃"

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ── 3. 节点：Pydantic state 直接属性访问 state.messages ──

def agent_node(state: RichState) ->dict:
    return {"messages": [llm.bind_tools([getweather]).invoke(state.messages)]}

def tag_node(state: RichState) ->dict:
    """演示覆盖式字段：summary 是普通字段，每次覆盖"""
    print(f"[tag] 执行了！进入时 tool_use_count = {state.tool_use_count}")
    ret = {"summary": "本图工具使用次数已刷新", "tool_use_count": state.tool_use_count + 1}
    print(f"[tag] 返回 tool_use_count = {ret['tool_use_count']}")
    return ret

# ── 4. 构图 ⚠️ 注意：agent 有分叉，只挂条件边，不要再加普通 END 边！ ──
builder=StateGraph(RichState)
builder.add_node("agent",agent_node)
builder.add_node("tools",ToolNode([getweather]))
builder.add_node("tag",tag_node)
builder.add_edge(START,"agent")
builder.add_edge("tools","agent")  # 回边：工具结果喂回 agent


# ⚠️ 正确写法：END 写进条件边的 path_map，而不是给 agent 加普通 END 边
# 解释：agent 后面要么去 tools（有 tool_calls）要么直接结束（无 tool_calls），
#       这个分叉只能由 tools_condition 决定；再加 add_edge("agent", END)
#       等于同时给 agent 两条路 → 构图冲突/行为异常

builder.add_conditional_edges("agent",tools_condition,{"tools":"tools",END:"tag"})
# 有 tool_calls → 去 tools（继续循环）
# 无 tool_calls → 先去 tag 收尾（不再是直接 END）
builder.add_edge("tag", END)   # 这时 tag→END 才有意义：tag 写完 → 图结束
graph = builder.compile()

# ── 5. 跑通验证 ──
result = graph.invoke({"messages": [("user", "北京天气怎么样")]})
print("=== 完整链路 ===")
for m in result["messages"]:
    m.pretty_print()
print("=== state 最终状态 ===")
print("summary:", result.get("summary"))
print("tool_use_count:", result.get("tool_use_count"))


