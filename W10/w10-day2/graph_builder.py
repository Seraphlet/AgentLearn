# graph_builder.py

from pyexpat.errors import messages
from typing import Annotated, TypedDict
import os
from cv2 import add
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
class ChatState(TypedDict):
    messages: Annotated[list,add_messages] 

llm=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
def build_graph(checkpoint=None):
    """和 FastAPI 零耦合：既能被服务调用，也能被脚本/测试调用"""
    def agent(state:ChatState):
        message=llm.invoke(state["messages"])
        print("agent",message.text,"\n")
        return {"messages":[message]}
    def respond(state: ChatState):
        turns = len(state["messages"])
        text = text_of(state["messages"][-1])
        print(text)
        return {"messages" : [("ai",f"（第 {turns} 条消息）你说的是：{text}")]}
    g=StateGraph(ChatState)
    g.add_node("respond",respond)
    g.add_node("agent",agent)
    g.add_edge(START,"agent")
    g.add_edge("agent","respond")
    g.add_edge("respond",END)
    return g.compile(checkpointer=checkpoint)

def text_of(message) -> str:
    """兼容 str 和多模态块结构两种 content"""
    content=getattr(message,"content",None)
    if isinstance(content,str):
        return content
    if isinstance(content,list):
        return "".join(b.get("text","") for b in content if isinstance(b,dict))
    return ""

def extract_text(messages) -> str:
    """从后往前找第一条有文字的 AI 消息 —— 接真图时这 10 行不是装饰（见魔鬼代言人）"""
    for m in reversed(messages):
        text = "".join(text_of(m).split())
        
        if text:
            return text
    return ""


