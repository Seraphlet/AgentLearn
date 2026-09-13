"""W6-D6 阶段C: 跨会话记忆 + 审批 端到端(复用 A/B 的地基, 不学新机制)

七个独立进程 —— 这就是本阶段的全部意义:
    python w6d6_c_e2e.py remember   # ① alice-1: 让它记住邮箱 → 应停在审批点
    python w6d6_c_e2e.py yes        # ② 新进程: 批准 → 才真正写进长期记忆库
    python w6d6_c_e2e.py recall     # ③ 新进程(同 thread): 问邮箱 → 靠 checkpoint 答出
    python w6d6_c_e2e.py bob        # ④ bob-1(新 thread): 让它记住工号 → 应停在审批点
    python w6d6_c_e2e.py no         # ⑤ 新进程: 拒绝 → 库里不能出现工号
    python w6d6_c_e2e.py carol      # ⑥ carol-1(全新 thread): 问邮箱 → 只能靠长期记忆库
    python w6d6_c_e2e.py verify     # ⑦ 断言: 直接读库文件, 看谁在谁不在
"""
import os
import sys
import json
import time
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
 
from langchain_openai import ChatOpenAI

DB = "memory_agent.db" # 和阶段A/B同一个库; thread 不同, 互不干扰
MEM = "w6d6_long_term_mem.json"   # ★ 长期记忆库: 全局共享, 【不绑】thread_id
# ══════ 1. 长期记忆库(你 W5 MemoryManager 的最小替身: 一个 json 文件) ══════
def _load_mem() ->list:
    if not os.path.exists(MEM):
        return []
    with open(MEM,"r",encoding="utf-8") as f:
         return json.load(f)
    

def _save_mem(items:list) ->None:
    with open(MEM,"w",encoding="utf-8") as f:
            json.dump(items,f,ensure_ascii=False,indent=2)

# ══════ 2. 两个工具: 一个只读(安全) / 一个写入(危险) ══════
@tool
def search_memory(query:str) ->str:
    """【安全·只读】在长期记忆库里检索(跨会话共享)。只读操作, 不需要审批。"""
    items = _load_mem()
    print(f">>> [安全工具] 检索 {query!r} → 库里共 {len(items)} 条")
    if not items:
         return "长期记忆是空的"
    return "长期记忆库内容:\n" + "\n".join(f"- {it['text']}" for it in items)
    # ★ 演示版: 把整库丢给模型自己挑。真版这里换成你 W5 的 memory_retriever(向量检索)

@tool
def remember_fact(text:str)->str:
    """【危险·写入】把一条事实写入长期记忆库(跨会话共享)。写操作, 执行前必须人工审批。"""
    items = _load_mem()
    items.append({"text": text,"ts":time.time()})
    _save_mem(items)
    print(f">>> [危险工具真身] 已写入长期记忆: {text}")
    return f"已记住：{text}"

# ★★ 分级审批的核心: 只有"写"进这个名单, 读的不进 ★★
DANGEROUS = {"remember_fact"}


class S(TypedDict):
    messages: Annotated[list,add_messages]

# ══════ 3. 两个 LLM: 决策的绑工具, 收口的不绑 ══════
_llm_kwargs = dict(
    model="deepseek-v4-flash",              # ← 和阶段B跑通的那个保持一致, 别换
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 两个工具 → 强制串行, 避开多 tool_calls 回填校验的坑

llm_decide = ChatOpenAI(**_llm_kwargs).bind_tools(
    [search_memory, remember_fact],
    parallel_tool_calls=False,
) # 若你的版本报错, 就删掉这一行

llm_summarize = ChatOpenAI(**_llm_kwargs) #不帮工具 → 收口时物理上产不出 tool_calls

SYS =(
    "你是一个带长期记忆的助手。规则:\n"
    "1. 用户说『记住：X』时，你必须【立即调用 remember_fact 工具】把 X 写进长期记忆库；"
    "不允许只回复『好的我记住了』而不调用工具。\n"
    "2. 用户问『之前记过什么 / 我的 XX 是什么』而你上下文里没有答案时，"
    "必须【调用 search_memory 工具】去长期记忆库查，不要凭猜测回答。\n"
    "3. 调完工具后，用一句简短中文总结结果。"
)

# ══════ 4. 节点 ══════
def decide(state: S)-> dict:
     ai_msg=llm_decide.invoke(state["messages"])
     names=[tc["name"] for tc in ai_msg.tool_calls]
     print(f"[decide] tool_calls={len(ai_msg.tool_calls)} {names}")
     return {"messages": [ai_msg]}

def approve(state: S) -> dict:
    last = state["messages"][-1]
    if not (isinstance(last,AIMessage) and last.tool_calls):
         return {}
    todo = [tc["name"] for tc in last.tool_calls]
    print(f"[approve] 待审批的工具调用：{todo}")
    # ★ 重放: resume 时本函数从第一行重跑, 所以这行会打印两次(和阶段B一样, 正常)
    answer = interrupt({
        "question": f"模型想执行 {todo}，批准吗？",   # ← 这次别拼错 :)
        "options": ["approve", "reject"],
    })
    print(f"[approve] 收到回答：{answer!r}" )
    if answer == "reject":
        # ★ 最小修复: 回一条"取消回执"(ToolMessage), 而不是另起一条 AIMessage。
        #   协议硬约束: 模型声明了几条 tool_calls, 就必须回填几条 tool_call_id 对应的 ToolMessage。
        #   原来回 AIMessage → tool_calls 永远没回执 → 这条 thread 的历史被毒化,
        #   下次再 invoke 同一个 thread_id 时, 这段历史被重新发给模型 → 400。
        return {"messages": [
            ToolMessage(
                content=f"用户拒绝执行本次调用，已取消（{tc['name']} 未真正执行）",
                tool_call_id=tc["id"],          # ★ 必须一一对应, 这是回执的关键字段
            )
            for tc in last.tool_calls
        ]}
    return {}                        # 放行: state 里那条 AI 原样流向 tools

def summarize(state: S) ->dict:
    ai_msg = llm_summarize.invoke(state["messages"])
    print("[summarize] 收口完成 (不绑工具 → 必然结束)")
    return {"messages": [ai_msg]}

# ══════ 5. 路由: ★分级审批就在这两个函数里 ★ ══════
def route_after_decide(state: S):
    last = state["messages"][-1]
    if not (isinstance(last,AIMessage) and last.tool_calls):
        return END
    if any(tc["name"] in DANGEROUS for tc in last.tool_calls):
         return "approve" # 含危险工具 → 先审
    return "tools"   # 全是安全工具 → 直接跑(不然每次检索都问人, 体验崩)

def route_after_approve(state: S):
    last = state["messages"][-1]
    return "tools" if (isinstance(last,AIMessage) and last.tool_calls) else END

# ══════ 6. 构图 ══════
def build_graph(saver):
    b=StateGraph(S)
    b.add_node("decide",decide)
    b.add_node("approve",approve)
    b.add_node("tools",ToolNode([search_memory,remember_fact]))
    b.add_node("summarize",summarize)
    b.add_edge(START,"decide")
    b.add_conditional_edges("decide", route_after_decide,{"approve":"approve","tools":"tools",END:END})
    b.add_conditional_edges("approve",route_after_approve,{"tools":"tools",END:END})
    b.add_edge("tools","summarize")
    b.add_edge("summarize",END)
    return b.compile(checkpointer=saver)

# ══════ 7. 谁和谁共用一个槽 ══════
THREAD_OF = {
    "remember": "alice-1", "yes": "alice-1", "status": "alice-1", "recall": "alice-1",
    "bob": "bob-1",        "no": "bob-1",
    "carol": "carol-1",            # ★ 全新会话: 验证"长期记忆能跨会话"
}

def peek(graph,cfg,tag):
    st=graph.get_state(cfg)
    msgs = st.values.get("messages",[]) or []
    print(f"\n[{tag}] next = {st.next}    消息数 = {len(msgs)}")
    for i,m in enumerate(msgs):
        head = str(m.content).replace("\n"," ")[:44]
        tc=getattr(m,"tool_calls",None)
        print(f"   [{i}] {type(m).__name__:14s} {head!r}" + (f"  tool_calls={tc}" if tc else ""))
    for t in getattr(st,"tasks",()):
        for intr in getattr(t,"interrupts",()):
            print(f"[{tag}] 待审批的问题: {intr.value}")
    return st

# ══════ 8. ★C 阶段的真正铁证: 直接读库文件, 不看 print ══════
def verify():
    items = _load_mem()
    print("长期记忆库当前内容：")
    for it in items:
        print("   -", it["text"])
    if not items:
        print("   (空)")
    texts = [it["text"] for it in items]
    assert any("alice@x.com" in t for t in texts), \
        "❌ alice 的邮箱不在库里 → ② 批准路径失败"
    assert not any("X99" in t for t in texts), \
        "❌ bob 的工号在库里 → ⑤ 拒绝路径失败(越权写入!)"
    print("\n✅ 断言: 批准的那条在库里, 拒绝的那条不在库里")

# ══════ 9. 主流程 ══════
FIRST_ASK = {
    "remember": "记住：我是 Alice，邮箱 alice@x.com",
    "bob":      "记住：我的工号是 X99",
    "carol":    "我不记得之前存过什么了，帮我查一下长期记忆库里有没有关于邮箱的记录",
}

def main(mode: str):
    if mode == "verify":
        return verify()

    conn= sqlite3.connect(DB,check_same_thread=False)
    saver = SqliteSaver(conn)
    if hasattr(saver,"setup"):
        saver.setup()
    graph = build_graph(saver)

    THREAD = THREAD_OF[mode]
    cfg = {"configurable": {"thread_id": THREAD}}
    print(f"[{mode}] thread_id = {THREAD}   (本进程 PID={os.getpid()})")

    if mode in FIRST_ASK:
        graph.invoke({"messages": [SystemMessage(content=SYS), ("user", FIRST_ASK[mode])]}, cfg)
        print(f"→ [{mode}] 本轮结束; 看下面 next 判断是停了还是跑完了")
    elif mode == "status":
        print("→ 纯读模式: 不 invoke, 只从 sqlite 里读回暂停现场")
    elif mode == "yes":
        out = graph.invoke(Command(resume="approve"), cfg)
        print("→ 最终回复:", out["messages"][-1].content)
    elif mode == "no":
        out = graph.invoke(Command(resume="reject"), cfg)
        print("→ 最终回复:", out["messages"][-1].content)
    elif mode == "recall":
        # ★ 只传新问题; cfg 会先把 checkpoint 里的历史(SYS + 前文)读回来
        out = graph.invoke({"messages": [("user", "我的邮箱是什么？")]}, cfg)
        print("→ 最终回复:", out["messages"][-1].content)
    else:
        print(f"未知模式: {mode}"); sys.exit(1)

    peek(graph, cfg, mode)
    conn.close()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status")


