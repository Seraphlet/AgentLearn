import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 回项目根，同 W4

from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from memory.memory_manager import MemoryManager      # W3 组件，原样复用

mm=MemoryManager(window=8,max_tokens=1500,keep=4)

prompt=ChatPromptTemplate.from_messages([
     ("system", "你是有记忆，能调工具的助手。历史对话和检索到用户事实、喜好和用户说记下的内容都在history里"),
     ("placeholder", "{history}"), 
     ("human", "{question}"),
])
llm=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def _lc_to(m:dict):
    role,content=m.get("role","user"),m.get("content","")
    print(m.__dir__)
    if role=="system":
        return SystemMessage(content=content)
    if role =="user":
        return HumanMessage(content=content)
    return AIMessage(content=content)
    
chain = prompt | llm

class State(TypedDict):
    question:str
    answer:str

def run_w4_chain(state:State)->dict:
     """节点 = 薄壳：取记忆 → 跑 W4 链 → 写回记忆。业务逻辑全在壳外，一行没改"""
     history =[_lc_to(m) for m in mm.get_context(query=state["question"])]
     ai = chain.invoke({"history":history,"question":state["question"]})
     mm.add({"role": "user", "content": state["question"]})  # 记忆照常写回
     
     mm.add({"role": "assistant", "content": ai.content})
 
     return {"answer": ai.content} # 只把 answer 更新进 State

builder = StateGraph(State)
builder.add_node("w4_chain", run_w4_chain)
builder.add_edge(START, "w4_chain")
builder.add_edge("w4_chain", END)
graph = builder.compile()

if __name__ == "__main__":
    print("=== W5-D1 步骤 C：W4 链包成 LangGraph 节点 ===\n")
    print(graph.get_graph().draw_ascii())          # 应看到 START → w4_chain → END

    # 3 连问：①纯问答 ②写记忆 ③考记忆（证明图外组件 mm 照常工作）
    for q in ["你好，简单介绍下你自己",
              "记住：我最喜欢的模型是 DeepSeek",
              "我最喜欢的模型是什么？"]:
        out = graph.invoke({"question": q})        # 每次从 START 全量跑一遍
        print(f"Q: {q}\nA: {out['answer']}\n")

