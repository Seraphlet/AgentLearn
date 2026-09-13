"""W6-D6 阶段A: 把图接上 SqliteSaver —— 跨【进程】持久化
用法(三次运行, 三个独立进程):
    python w6d6_a_persist.py write     # ① 让 agent 记住邮箱
    python w6d6_a_persist.py read      # ② 新进程问邮箱  ← 同 thread
    python w6d6_a_persist.py other     # ③ 反例: 换个 thread 问  ← 应该不知道
"""

import os 
import sys
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

DB="memory_agent.db"


# ══════ 1. State ══════
class S(TypedDict):
    messages: Annotated[list,add_messages]

# ══════ 2. 节点 ══════
# ⚠️ 这里现在是"最小占位图", 只为验证持久化机制。
#    ★ 阶段 A 跑绿之后, 把这一整段换成你 W5-D6 的
#      route / short / long / compress / agent / tools 六个节点。
#      换的时候【只改节点内容, 不要碰下面的 build_graph】。


_llm=ChatOpenAI(
    model="deepseek-v4-flash",          
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

def agent_node(state: S) -> dict:
    return {"messages": [_llm.invoke(state["messages"])]}

# ══════ 3. 构图: 今天唯一的变化就在 compile 那一行 ══════

def build_graph(saver):
    b=StateGraph(S)
    b.add_node("agent",agent_node)
    b.add_edge(START,"agent")
    b.add_edge("agent",END)
    # ★★★ 唯一新增的一行 ★★★
    #   把 MemorySaver() 换成 SqliteSaver(conn) → 从"内存"变"落盘"
    return b.compile(checkpointer=saver)

# ══════ 4. 主流程 ══════

def main(mode: str):
    # ① 打开 SQLite 连接(文件不存在会自动创建)
    #    check_same_thread=False: 允许跨线程用这条连接, 否则 SqliteSaver 可能报错
    conn=sqlite3.connect(DB,check_same_thread=False)
    saver=SqliteSaver(conn)
    # 若报 "no such table: checkpoints" → 取消下面这行注释, 手动建表
    # saver.setup()

    graph=build_graph(saver)
    # ② ★ thread_id = 会话身份, 三次运行【必须写死同一个字符串】才能续上

    THREAD="alice-1.2" if mode !="other" else "bob-1.2"
    cfg={"configurable": {"thread_id":THREAD}}

    # ③ 三次运行, 三种输入
    if mode == "write":
        question= "记住：我叫 Alice，邮箱是 alice@x.com。只回复'已记住'。"
    elif mode == "read":
        question = "我的邮箱是什么？"
    else:
        question = "我的邮箱是什么？"

    # ④ ★ invoke 时必须把 cfg 传进去 —— 不带 config, 框架不知道读哪个槽
    out = graph.invoke({"messages":[("user",question)]},cfg)

    print(f"\n[{mode}] thread_id = {THREAD}")
    print("回复:", out["messages"][-1].content)

    # ⑤ ★ 判据用【客观量】: 消息数。不用自然语言(模型的话会飘)
    st=graph.get_state(cfg)
    n=len(st.values["messages"])
    for i in st.values["messages"]:
        try:
            print(f"第条消息是:",i.content,sep="\n")
        except Exception as e:
            print("出错啦",e)
    print(f"[算的] state 里共 {n} 条消息   next={st.next}")
    conn.close()
    return n

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    main(mode)






