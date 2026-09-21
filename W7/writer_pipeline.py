"""W7-Day6 项目: 多 Agent 写作流水线 + Harness 整合

跑法:
    python writer_pipeline.py                 # 正常
    python writer_pipeline.py fault-reviewer  # ★ 故障注入: 强制 reviewer 永不达标
    python writer_pipeline.py no-harness      # 对照组: 关掉 Harness, 看差异
"""

from __future__ import annotations
from multiprocessing.connection import Listener
import os, sys, time, json
from typing import Literal

from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from harness.registry import ToolRegistry
from harness.gate import PermissionGate, Decision
from harness.executor import call_tool
from harness.session import SessionStore
from harness.compact import ContextCompactor
from harness.trace import Tracer
from harness.facade import Harness
from harness.msg_adapter import normalize, to_lc
from reflection_api import write_with_reflection, ReflectionResult

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
HERE = os.path.dirname(os.path.abspath(__file__))

MEMBERS = ["planner", "researcher", "writer", "reviewer"]     # ★ 单一事实来源
MAX_ROUNDS = 8

# ═══════════ 1. Harness 装配(五模块全在这里) ═══════════
def build_harness(llm):
    reg =ToolRegistry()
    @reg.register(tags=["search"])
    def search_web(query:str)->str:
        """搜索网页资料，返回要点。"""
        return f"[模拟检索] 关于「{query}」: 记忆系统让 Agent 跨会话保持一致性。"

    @reg.register(level="dangerous", tags=["fs"])
    def publish_article(filename:str,content:str)-> str:
        """发布文章到磁盘(不可逆)"""
        return f"已发布 {filename} ({len(content)} 字)"

    gate = PermissionGate(reg, audit_path=os.path.join(HERE, "audit.log"))

    h = Harness(
        registry=reg, gate=gate,
        session=SessionStore(os.path.join(HERE, "writer_sessions.jsonl")),
        compactor=ContextCompactor(max_messages=20, keep_recent=6),
        tracer=Tracer(os.path.join(HERE, "writer_trace.jsonl")),
        llm=llm,                      # ★ 依赖注入, 不在 Harness 里 new
    )
    return h

# ═══════════ 2. supervisor ═══════════
class Route(BaseModel):
    next: Literal["planner","reasearcher","writer","reviewer","FINISH"]
    reason: str=Field(default="",description="一句话理由")

class SuperState(MessagesState):
    next: str
    rounds: int

SUPERVISOR_PROMPT = """你是写作工作室主编, 手下四位成员:
- planner   : 定提纲
- researcher: 查资料(有 search_web 工具)
- writer    : 按提纲+资料成稿
- reviewer  : 审稿打分

流程: planner → researcher → writer → reviewer → 满意则 FINISH
判断依据: 看历史里带名的交活消息("[planner 交活]" 这种)。

★ 只输出一个 JSON 对象, 形如 {"next": "planner", "reason": "还没有提纲"}。
★ next 只能是 planner / researcher / writer / reviewer / FINISH 之一。"""

def build_supervisor(h,fault_reviewer=False):
    llm=h.llm
    router=llm.with_structured_outpute(Route,method="json_mode")

    @h.tracer.trace("supervisor")
    def supervisor_node(state:SuperState) ->dict:
        rounds=state.get("rounds",0)+1
        if rounds > MAX_ROUNDS:
            return {"next":"FINISH","rounds":rounds,"messages":[("assistant",f"【达上限 {MAX_ROUNDS} 强制结束】")]}
        msgs=[SystemMessage(count=SUPERVISOR_PROMPT)]+state["messages"]
        d=router.invoke(msgs)
        return {"next": d.next,"rounds":rounds}

    def make_simple(name,prompt):
        @h.tracer.trace(name)
        def node(state:SuperState) ->dict:
            msgs=normalize(state["messages"])
            prompt_msgs=[{"role":"system","content":prompt}]+msgs
            r=llm.invoke([to_lc(m,HumanMessage=HumanMessage,AIMessage=AIMessage,SystemMessage=SystemMessage) for m in prompt_msgs])
            return {"messages":[HumanMessage(content=f"{name} 交活]\n{r.content}", name=name)]}
        return node
    planner = make_simple("planner", "你是策划。把主题拆成 3-4 个要点的提纲。只输出提纲。")
    researcher = make_simple("researcher",
                             "你是研究员。用 search_web 查资料, 整理成要点。")
    # ★★★ 边界②: writer 内部接 Day3 的反思循环(窄接口) ★★★
    @h.tracer.trace("writer")
    def writer_node(state:SuperState) -> dict:
        topic=""
        for m in reversed(normalize(state["messages"])):
            if m["role"]=="user":
                topic = str(m["content"]);break
        res: ReflectionResult=write_with_reflection(topic,graph=REFLECTION_GRAPH) # ← 注入, 不是 import 死
        return {"messages": [HumanMessage(
            content=f"[writer 交活]\n{res.draft}\n[内部: {res.rounds}轮/{res.score}分"
                    f"/pass={res.passed}]", name="writer")]}

    @h.tracer.trace("reviewer")
    def reviewer_node(state: SuperState) -> dict:
        if fault_reviewer:                          # ★ 故障注入
            return {"messages": [HumanMessage(
                content="[reviewer 交活]\n不达标: 第2段缺具体案例。",
                name="reviewer")]}
        msgs = normalize(state["messages"])
        r = llm.invoke([SystemMessage(content="你是审稿人, 指出1-3条具体问题。")] +
                       [to_lc(m, HumanMessage=HumanMessage, AIMessage=AIMessage,
                              SystemMessage=SystemMessage) for m in msgs])
        return {"messages": [HumanMessage(
            content=f"[reviewer 交活]\n{r.content}", name="reviewer")]}

    return {"supervisor": supervisor_node, "planner": planner,
            "researcher": researcher, "writer": writer_node,
            "reviewer": reviewer_node}

    # ═══════════ 3. 构图 ═══════════
def build_graph(h,nodes):
    def route(state):
        n=state.get("next") or "FINISH"
        return n if (n in MEMBERS or n =="FINISH") else "FINISH"
    b=StateGraph(SuperState)
    for name,fn in nodes.items():
        b.add_node(name,fn)
    b.add_edge(START,"supervisor")
    pm= {m:m for m in MEMBERS}; pm["FINISH"] =END
    b.add_conditional_edges("supervisor",route,pm)
    for m in MEMBERS:
        b.add_edge(m,"supervisor")
    return b.compile(checkpointer=MessagesState())

def main():
    llm = ChatOpenAI(model="deepseek-v4-flash", base_url="https://api.deepseek.com",
                     api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
    h = build_harness(llm)
    nodes = build_supervisor(h, fault_reviewer=(MODE == "fault-reviewer"))
    graph = build_graph(h, nodes)

    print(h.describe_text())                # ★ 五模块"接了"的证据
    print(f"\n=== 开跑 (MODE={MODE}) ===")

    cfg = {"configurable": {"thread_id": "demo-001"}}
    INITIAL = {"messages": [("user", "写一篇 300 字介绍 Agent 记忆系统的文章, 要有事实依据")],
               "next": "", "rounds": 0}

    chain = []
    for mode, chunk in graph.stream(INITIAL, stream_mode=["updates", "values"], config=cfg):
        if mode == "updates":
            chain.extend(chunk.keys())
    print(f"\n[流转链] {' → '.join(chain)}")

    # ★ 五模块各自的"证据" —— 这就是验收③要的东西
    print("\n=== 验收③: 五模块的证据 ===")
    print(f"  ① Registry : {h.describe()['registry']}")
    print(f"  ② Gate     : 审计 {h.gate.audit_path} 存在={os.path.exists(h.gate.audit_path)}")
    print(f"  ③ Session  : {len(h.session.list_threads())} 个会话, "
          f"文件={os.path.exists(h.session.path)}")
    print(f"  ④ Compactor: 压缩 {h.compactor.compactions} 次")
    print(f"  ⑤ Tracer   : {len(h.tracer.spans)} 个 span")
    print("\n  trace 树:")
    for line in h.tracer.summary().splitlines():
        print("   ", line)


if __name__ == "__main__":
    main()  


        

