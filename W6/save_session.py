import os,asyncio
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
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

def main():
    with SqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph = builder.compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "persist-1"}}
        graph.invoke({"messages": [("user", "我叫小明，住在北京")]}, cfg)
        print("已写入会话")




def main1():
    with SqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph = builder.compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "persist-1"}}
        r = graph.invoke({"messages": [("user", "我叫什么？住哪？")]}, cfg)
        print("重启后回复:", r["messages"][-1].content)   # 应答出小明/北京

def main2():
    with SqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph =builder.compile(checkpointer=saver)
        cfg = {"configurable":{"thread_id":"persist-1"}}

        #get_state:读当前存档
        state = graph.get_state(cfg)
        print("消息总数:",len(state.values["messages"])) #应>2(小明那句+重启那句)
        print("下一步节点:",state.next) #应为空 ————没中断过(HITL时才非空)

        # ── ② get_state_history：看时间线（新→旧）──

        print("历史checkpoint时间线")
        for i,snap in enumerate(graph.get_state_history(cfg)):
            cid =snap.config["configurable"]["checkpoint_id"]
            print(f" #{i} checkpoint_id={cid[:8]}消息数={len(snap.values["messages"])}")

        # ── ③ update_state：外部改档（人从图外注入）──
        graph.update_state(cfg, {"messages": [("user", "顺便记住：我喜欢喝美式咖啡")]})

        # ── ④ 验证注入真的被 agent 感知 ──
        r = graph.invoke({"messages": [("user", "我喜欢喝什么？")]}, cfg)
        print("注入后回复:", r["messages"][-1].content)   # 应答出美式咖啡

main2()
