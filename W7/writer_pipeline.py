"""W7-Day6 项目: 多 Agent 写作流水线 + Harness 整合

跑法(在 W7 目录下):
    python writer_pipeline.py                 # 正常
    python writer_pipeline.py fault-reviewer  # ★ 故障注入: 强制 reviewer 永不达标
    python writer_pipeline.py no-harness      # 对照组: 摘掉 Registry/Gate/落盘 Session/压缩

三个边界:
    ① researcher 查资料 → 必须穿过 Gate + executor(不许旁路 registry.invoke)
    ② writer    写稿   → 进反思子图(窄接口, 只回 5 个字段)
    ③ supervisor 决策  → 存全量 + 读时压(投影), 拿压缩后的视图喂模型
"""

from __future__ import annotations
import os, sys
from typing import Literal

from dotenv import load_dotenv
load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                      # ★ 从仓库根启动也能找到 harness/
    sys.path.insert(0, HERE)

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from harness.registry import ToolRegistry
from harness.gate import PermissionGate
from harness.executor import call_tool
from harness.session import SessionStore
from harness.compact import ContextCompactor
from harness.trace import Tracer
from harness.facade import Harness
from harness.msg_adapter import normalize, to_lc
from reflection_api import build_reflection_graph, write_with_reflection, ReflectionResult

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
MEMBERS = ["planner", "researcher", "writer", "reviewer"]     # ★ 单一事实来源
MAX_ROUNDS = 8
THREAD = "demo-001"
# ★ to_lc 的消息类一次性备齐: 少传一个 ToolMessage, 遇到 tool 消息就会炸
LC = dict(HumanMessage=HumanMessage, AIMessage=AIMessage,
          SystemMessage=SystemMessage, ToolMessage=ToolMessage)

# ═══════════ 1. Harness 装配(五模块全在这里) ═══════════
def build_harness(llm, enabled=True):
    if not enabled:                     # 对照组: 只剩 Tracer(装饰器还要能跑)
        return Harness(registry=None, gate=None,
                       session=SessionStore(None),                    # 纯内存, 一关就没
                       compactor=ContextCompactor(max_messages=10**9),  # 永不压
                       tracer=Tracer(None), llm=llm)

    reg = ToolRegistry()
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


def _topic_of(state) -> str:
    """用户最初给的主题。

    ★ 坑: 成员交活也走 HumanMessage, 只是带了 name —— 所以 role=="user" 不代表
      "人类用户"。必须认「第一条没有 name 的 user 消息」, 否则拿到的是上一位成员的
      交活长文(researcher 会照着提纲去检索, writer 会照着资料去写稿)。
    """
    for m in normalize(state["messages"]):
        if m["role"] == "user" and not m.get("name"):
            return str(m["content"])
    return ""


def _has_done(state, name: str) -> bool:
    """历史里有没有 "[xxx 交活]" —— supervisor 的"交活消息"协议"""
    tag = f"[{name} 交活]"
    return any(tag in str(m.get("content") or "") for m in normalize(state["messages"]))

# ═══════════ 2. supervisor ═══════════
class Route(BaseModel):
    next: Literal["planner","researcher","writer","reviewer","FINISH"]
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
★ 硬规则: 历史里没有 "[reviewer 交活]" 就不许 FINISH(稿子没审过不算完成)。

★ 只输出一个 JSON 对象, 形如 {"next": "planner", "reason": "还没有提纲"}。
★ next 只能是 planner / researcher / writer / reviewer / FINISH 之一。"""

def build_supervisor(h, reflection_graph=None, fault_reviewer=False):
    llm=h.llm
    router=llm.with_structured_output(Route,method="json_mode")

    @h.tracer.trace("supervisor")
    def supervisor_node(state:SuperState) ->dict:
        rounds=state.get("rounds",0)+1
        if rounds > MAX_ROUNDS:
            return {"next":"FINISH","rounds":rounds,
                    "messages":[HumanMessage(content=f"【达上限 {MAX_ROUNDS} 强制结束】")]}

        # ★ 边界③: ③存全量(存档) → ④读时压(投影), 喂给模型的是投影后的视图
        history = normalize(state["messages"])
        h.save(THREAD, history)
        view = h.load_for_llm(THREAD) or history
        msgs = [SystemMessage(content=SUPERVISOR_PROMPT)] + [to_lc(m, **LC) for m in view]

        d=router.invoke(msgs)
        nxt=d.next
        # ★ 程序兜底(和 critic 的 passed_by_code 同一个套路): 模型看到 writer 交活里
        #   "[内部: ... pass=True]" 就爱直接收工, 于是 reviewer 永远轮不到。
        #   流程写死了 writer → reviewer, 那就由程序保证 —— 没审过不许 FINISH。
        if nxt=="FINISH" and not _has_done(state,"reviewer"):
            nxt="reviewer"
        return {"next": nxt,"rounds":rounds}

    def make_simple(name,prompt):
        @h.tracer.trace(name)
        def node(state:SuperState) ->dict:
            msgs=normalize(state["messages"])
            prompt_msgs=[{"role":"system","content":prompt}]+msgs
            r=llm.invoke([to_lc(m, **LC) for m in prompt_msgs])
            return {"messages":[HumanMessage(content=f"[{name} 交活]\n{r.content}", name=name)]}
        return node
    planner = make_simple("planner", "你是策划。把主题拆成 3-4 个要点的提纲。只输出提纲。")

    @h.tracer.trace("researcher")
    def researcher_node(state:SuperState) ->dict:
        # ★ 边界①: 查资料必须穿过 Gate + executor —— 不许旁路 registry.invoke
        topic=_topic_of(state)
        if h.registry is not None and h.registry.exists("search_web"):
            r=call_tool(h.registry, h.gate, "search_web", {"query":topic}, actor="researcher")
            material = r.get("result","") if r["ok"] else f"(检索被拦/失败: {r.get('error')})"
        else:
            material = "(对照组: 没接 Harness, 手上没有检索工具)"
        msgs=[{"role":"system",
               "content":"你是研究员。把主题和下方资料整理成要点, 只输出要点。\n\n"
                         f"【检索结果】{material}"}]+normalize(state["messages"])
        r=llm.invoke([to_lc(m, **LC) for m in msgs])
        return {"messages":[HumanMessage(content=f"[researcher 交活]\n{r.content}", name="researcher")]}

    # ★★★ 边界②: writer 内部接 Day3 的反思循环(窄接口) ★★★
    @h.tracer.trace("writer")
    def writer_node(state:SuperState) -> dict:
        topic=_topic_of(state)
        res: ReflectionResult=write_with_reflection(topic, graph=reflection_graph) # ← 注入, 不是 import 死
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
                       [to_lc(m, **LC) for m in msgs])
        return {"messages": [HumanMessage(
            content=f"[reviewer 交活]\n{r.content}", name="reviewer")]}

    return {"supervisor": supervisor_node, "planner": planner,
            "researcher": researcher_node, "writer": writer_node,
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
    # ★ 要按 thread_id 存档, 必须给一个真的 checkpointer(MessagesState 不是)
    return b.compile(checkpointer=MemorySaver())

def main():
    llm = ChatOpenAI(model="deepseek-v4-flash", base_url="https://api.deepseek.com",
                     api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
    h = build_harness(llm, enabled=(MODE != "no-harness"))
    reflection = build_reflection_graph()      # ★ 装配期造好子图, 注入给 writer 节点
    nodes = build_supervisor(h, reflection_graph=reflection,
                             fault_reviewer=(MODE == "fault-reviewer"))
    graph = build_graph(h, nodes)

    print(h.describe_text())                # ★ 五模块"接了"的证据
    print(f"\n=== 开跑 (MODE={MODE}) ===")

    cfg = {"configurable": {"thread_id": THREAD}}
    INITIAL = {"messages": [("user", "写一篇 300 字介绍 Agent 记忆系统的文章, 要有事实依据")],
               "next": "", "rounds": 0}

    chain = []
    for mode, chunk in graph.stream(INITIAL, stream_mode=["updates", "values"], config=cfg):
        if mode == "updates":
            chain.extend(chunk.keys())
    print(f"\n[流转链] {' → '.join(chain)}")

    # ★ 五模块各自的"证据" —— 这就是验收③要的东西
    d = h.describe()
    audit = d["gate"].get("audit_path")
    print("\n=== 验收③: 五模块的证据 ===")
    print(f"  ① Registry : {d['registry']}")
    print(f"  ② Gate     : 审计 {audit} 存在={os.path.exists(audit) if audit else False}")
    print(f"  ③ Session  : {d['session']['threads']} 个会话, 文件={d['session']['path']}")
    print(f"  ④ Compactor: 压缩 {d['compaction']['compactions']} 次 "
          f"(阈值 max={d['compaction']['max_messages']}, "
          f"本次历史 {len(h.session.messages(THREAD))} 条)")
    print(f"  ⑤ Tracer   : {d['trace']['spans']} 个 span")
    print("\n  trace 树:")
    for line in h.tracer.summary().splitlines():
        print("   ", line)


if __name__ == "__main__":
    main()
