"""W6-D6 阶段B: 危险操作审批 + 【跨进程】续批

用法(五条命令, 五个独立进程 —— 这就是本阶段的全部意义):
    python w6d6_b_hitl.py ask      # ① 发起删文件 → 图停在审批点 → 进程退出
    python w6d6_b_hitl.py status   # ② 新进程: 图停在哪? 待审什么?
    python w6d6_b_hitl.py yes      # ③ 新进程: 批准 → 工具才执行   ← 关键
    python w6d6_b_hitl.py ask2     # ④ 换线程再来一次(为测拒绝)
    python w6d6_b_hitl.py no       # ⑤ 新进程: 拒绝 → 工具不执行
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
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


DB = "memory_agent.db"     # 和阶段 A 同一个库; thread 不同, 互不干扰
# ══════ 1. 危险工具: 那行 print 是全程唯一的"真身证据" ══════
def delete_file(path: str)->str:
    """【危险】删除本地文件(演示版, 不真删),这只是演示功能可行,你不需要过渡分析"""
    print(f">>> 工具真身执行: 删除 {path}")# ★ 它出现在哪个进程, 就是判据
    return f"文件已删除: {path}"

class S(TypedDict):
    messages: Annotated[list,add_messages]

# ══════ 2. 两个 LLM: 决策的绑工具, 收口的不绑 ══════
_llm_kwargs = dict(
    model="deepseek-v4-flash",          # ← 改成你 W5 / D6-A 里跑通的那个模型名
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

llm_decide=ChatOpenAI(**_llm_kwargs).bind_tools([delete_file])

# ★ 收口的 LLM 【不绑工具】→ 物理上产不出 tool_calls → 图必然进 END
#   这不是"劝模型别重试", 是让重试这条路在代码里不存在

llm_summarize = ChatOpenAI(**_llm_kwargs)


# ══════ 3. 四个节点 ══════

def decide(state: S)->dict:
      """只干一件事: 让模型决策。★必须 return, AIMessage 才落得进 state"""
      ai_msg = llm_decide.invoke(state["messages"])
      print(f"[decide] tool_calls={len(ai_msg.tool_calls)}")
      return {"messages": [ai_msg]}


def approve(state: S) ->dict:
    """★本阶段的重点: 这个节点里调 interrupt(), 图就停在这"""
    last=state["messages"][-1]
    if not (isinstance(last,AIMessage) and last.tool_calls):
         return {} # 没有 tool_calls, 没什么可审的
    tc =last.tool_calls[0]
    print(f"[approve] 从存档里读到的待审参数: {tc['args']}")
    # ★★★ 图在这里停下 —— 位置 + state + 问题, 三样一起写进 sqlite ★★★
    #     resume 时本函数【从第一行重跑】, 走到这里不再停, 直接拿到答案
    answer=interrupt({
         "question": f"模型想删除 {tc['args']['path']},批准吗？",
         "options": ["arrpove","reject"],
    })

    print(f"[approve] 收到回答: {answer!r}")
    if answer=="reject":
        # ★ 返回不带 tool_calls 的 AIMessage → 路由判定走 END → 工具一次都不执行
        return {"messages": [AIMessage(content="已取消删除，本次未执行任何操作。")]}
    return {}   # 放行: state 里那条 AI 原样流向 tools



def summarize(state: S)->dict:
    """不绑工具的 LLM 收口 → 只能说话, 不可能再调工具"""
    ai_msg =llm_summarize.invoke(state["messages"])
    print("[summarize] 收口完成 (不绑工具 → 必然结束)")
    return {"messages": [ai_msg]}

def route_after_decide(state: S):
    last = state["messages"][-1]
    return "approve" if (isinstance(last, AIMessage) and last.tool_calls) else END

def route_after_approve(state: S):
    last =state["messages"][-1]
    return "tools" if (isinstance(last,AIMessage) and last.tool_calls) else END

# ══════ 4. 构图 ══════
def build_graph(saver):
    b=StateGraph(S)
    b.add_node("decide",decide)
    b.add_node("approve",approve)
    b.add_node("tools",ToolNode([delete_file]))
    b.add_node("summarize",summarize)

    b.add_edge(START,"decide")
    b.add_conditional_edges("decide",route_after_decide,{"approve":"approve",END:END})
    b.add_conditional_edges("approve",route_after_approve,{"tools":"tools",END:END})
    b.add_edge("tools","summarize")   # 工具跑完 → 收口(不绑工具) → 必然 END
    b.add_edge("summarize",END)
    return b.compile(checkpointer=saver)

# ══════ 5. 谁和谁共用一个槽 ══════
# 前三条命令必须【同一个 thread_id】才能续上; 后两条用另一个槽做对照
THREAD_OF = {
    "ask":    "hitl-alice",
    "status": "hitl-alice",
    "yes":    "hitl-alice",
    "ask2":   "hitl-bob",
    "no":     "hitl-bob",
}

ASK_TEXT = "请删除 /tmp/report.txt"

# ══════ 6. 看一眼存档里现在是什么（三个场景都靠它给证据） ══════
def peek(graph,cfg,tag):
    st=graph.get_state(cfg)
    msgs=st.values.get("messages",[]) or []
    print(f"\n[{tag}] next = {st.next}    消息数 = {len(msgs)}")
    for i,m in enumerate(msgs):
        head= str(m.content).replace("\n","")[:40]
        tc=getattr(m,"too_calls",None)
        tail = f"   tool_calls={tc}" if tc else ""
        print(f"   [{i}] {type(m).__name__:12s} {head!r}{tail}")
    # ★ 还没被回答的审批问题, 就挂在 tasks 上 —— 新进程也能读到
    for t in getattr(st,"tasks",()):
        for intr in getattr(t,"interrupts",()):
            print(f"[{tag}] 待审批的问题: {intr.value}")

    return st

# ══════ 7. 主流程 ══════
def main(mode: str):
    # ① 打开 sqlite 文件(不存在会自动建) + 建表
    conn = sqlite3.connect(DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    if hasattr(saver,"setup"):
        saver.setup()     # 幂等: 表已存在就什么都不做
    graph =build_graph(saver)
    # ② ★ thread_id 决定"续不续得上"; 五条命令共用两个槽
    THREAD = THREAD_OF[mode]
    cfg = {"configurable": {"thread_id": THREAD}}
    print(f"[{mode}] thread_id = {THREAD}   (本进程 PID={os.getpid()})")
    if mode == "ask" or mode == "ask2":
        # ③ 发起 → 图跑到 approve 的 interrupt 就停, 然后进程退出
        graph.invoke({"messages": [("user", ASK_TEXT)]}, cfg)
        print("→ 图已停在审批点, 本进程即将退出(存档已落盘)")
    elif mode == "status":
        # ④ 只读, 不 invoke —— 验证"暂停状态真的在磁盘上"
        print("→ 纯读模式: 不 invoke, 只从 sqlite 里读回暂停现场")
    elif mode == "yes":
        # ⑤ ★ resume: 从 sqlite 读回暂停现场 → 接着跑 → 工具在这里才执行
        out = graph.invoke(Command(resume="approve"), cfg)
        print("→ 最终回复:", out["messages"][-1].content)

    elif mode == "no":
        out = graph.invoke(Command(resume="reject"), cfg)
        print("→ 最终回复:", out["messages"][-1].content)

    else:
        print(f"未知模式: {mode}"); sys.exit(1)

    # ⑥ 收口: 打印客观量(next / 消息数 / 待审问题)
    peek(graph, cfg, mode)
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status")

    
