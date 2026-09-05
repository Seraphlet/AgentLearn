import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ── 1. State：messages 用 reducer 自动累加 
class AgentState(TypedDict):
    messages: Annotated[list,add_messages]


# ── 2. 工具（@tool 从签名+docstring 自动生成 schema，docstring 是给模型看的！）──

@tool 
def add(a:int,b: int) -> int:
    """计算两个整数之和"""
    return a+b

@tool
def multiply(a: int,b: int) -> int:
    """计算两个整数之积"""
    return a*b

tools = [add,multiply]

# ── 3. LLM（复用你 W2 的多厂商思路，只换 base_url/model）──

llm= ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

llm_with_tools=llm.bind_tools(tools) #关键，绑了工具的llm才会返回tool_calls

# # ── 4. agent 节点（含防死循环：消息超 10 条强制收口）──

def agent_node(state: AgentState) -> dict:
    if len(state["messages"]) > 10:
        forced={"role": "user","content": "请基于以上共计结果直接给出最终答案，不要在调用工具。"}
        return {"messages": [llm_with_tools.invoke(state["messages"] + forced)]}
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ── 5. 构图：ReAct 经典拓扑 ──

builder =StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tools",ToolNode(tools)) # 工具执行节点


builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",tools_condition)# 有 tool_calls → tools，否则 → END

builder.add_edge("tools","agent") # ⚠️ 回边！没有它工具跑完就结束，永远没有最终答案

graph=builder.compile()
# ── 6. 链式任务验收：需要"先 add 再 multiply"两次工具调用 ──
result=graph.invoke({"messages": [("user", "帮我算一下 (12 + 8) 乘以 3")]})
print("=== 完整链路（肉眼确认）===")
for m in result["messages"]:
    m.pretty_print()

# ── 7. stream 看 agent/tools 交替 ──
print("\n=== stream 看节点流转 ===")

for event in graph.stream({"messages": [("user", "3 加 5 等于几")]}):
    print(event)

from langchain.agents import create_agent
prebuilt_agent=create_agent(llm_with_tools,tools)
result2 =prebuilt_agent.invoke({"messages": [("user","(12+8)*3 等于几")]})
print("=== prebuilt 版输出 ===")
for m in result2["messages"]:
    m.pretty_print()
