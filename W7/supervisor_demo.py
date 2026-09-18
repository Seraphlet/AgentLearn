"""W7-Day2 项目: Supervisor 模式 3-Worker 流水线
=================================================================
结构化输出路由(json_mode) + FINISH 收口 + 轮数兜底 + 拓扑导出 + 断言自检

跑法(Windows CMD):
    python w7\supervisor_demo.py                  # 正常模式
    python w7\supervisor_demo.py fallback         # 兜底测试(MAX_ROUNDS=1)

自杀测试(证明断言真的会报警 —— 期望看到 ❌):
    set FAULT=no_name    && python w7\supervisor_demo.py fallback
    set FAULT=no_deliver && python w7\supervisor_demo.py fallback
    set FAULT=

PowerShell 用: $env:FAULT="no_name"  /  Remove-Item Env:FAULT

依赖: pip install langgraph langchain-openai langchain-core python-dotenv
.env:  DEEPSEEK_API_KEY=sk-xxxx
=================================================================
"""
import os
import sys
import time
from typing import Literal, get_args

from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.agents import create_agent


# ══════════════════ 0. 配置 ══════════════════
# ★ 相对清单原版的 4 处改动(面试可讲):
#   ① method="json_mode"  ← DeepSeek 不支持 json_schema(默认) 和强制 tool_choice
#   ② prompt 里必须有 "JSON" 字样  ← json_mode 的硬性要求
#   ③ MODE 作为模式单一事实来源, 不用 MAX_ROUNDS>1 去代理
#   ④ 兜底判断放在调 LLM 之前  ← 省掉那一轮白花的钱

MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
MAX_ROUNDS = 1 if MODE == "fallback" else 8
FAULT = os.getenv("FAULT") or None          # 仅用于自杀测试

MEMBERS = ["researcher", "writer", "reviewer"]      # ★ 单一事实来源


def make_llm(temperature: float = 0):
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,     # 0 = 决策尽量稳, 少飘
    )


llm = make_llm()           # worker 用
router_llm = make_llm()    # supervisor 用(★不能 bind_tools)


# ══════════════════ 1. state + Route ══════════════════
class SuperState(MessagesState):
    """在 MessagesState(自带 messages) 上加两个自定义字段"""
    next: str        # supervisor 的决策
    rounds: int      # 轮数计数(兜底用)


class Route(BaseModel):
    """supervisor 的决策只能从这里选 —— FINISH 也是一等公民"""
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    reason: str = Field(default="", description="一句话理由, 方便调试")


def check_single_source_of_truth():
    """防呆: 花名册和 Literal 不一致时立刻炸, 而不是跑到一半才崩"""
    opts = set(get_args(Route.model_fields["next"].annotation))
    assert opts == set(MEMBERS) | {"FINISH"}, \
        f"名单和 Literal 不一致! Literal={opts}, MEMBERS={set(MEMBERS)}"
    print(f"[check] 名单一致性 OK: {sorted(opts)}")


# ★ 改动①: DeepSeek 实测可用的是 json_mode
#   function_calling → 400 'Thinking mode does not support this tool_choice'
#   json_schema(默认) → 400 'This response_format type is unavailable now'
router = router_llm.with_structured_output(Route, method="json_mode")


# ══════════════════ 2. 三个 worker ══════════════════
@tool
def web_search(query: str) -> str:
    """搜索网页资料, 返回要点。"""
    print(f"        >>> [web_search 真身被调用] query={query!r}")
    return (f"[模拟检索结果] 关于「{query}」: LangGraph 是 LangChain 团队开发的、"
            f"用于构建有状态多步 Agent 的框架; 核心是 StateGraph + 条件边 + checkpointer。")


WORKER_PROMPTS = {
    "researcher": "你是研究员。用 web_search 查资料, 把关键要点整理成 2-3 条, 不要写完整文章。",
    "writer": ("你是写手。把对话历史里 researcher 交的资料整合成一篇 300 字左右的中文文章。"
               "只写文章, 不要写评论或修改意见。"),
    "reviewer": ("你是审稿人。检查上一位 writer 写的文章, 指出事实性/结构/表达上的具体问题(1-3 条)。"
                 "如果文章已经可以定稿, 明确说「可以定稿」。"),
}


def make_worker(name: str, tools: list, prompt: str):
    """工厂: 造一个 worker 节点。每个 worker = 独立 agent + 独立 system prompt"""
    agent = create_agent(llm, tools, system_prompt=prompt)
    # 老版本 langgraph 报 TypeError → 改成 state_modifier=prompt

    def node(state: SuperState) -> dict:
        t0 = time.time()
        result = agent.invoke({"messages": state["messages"]})
        dt = time.time() - t0

        last = result["messages"][-1]
        text = str(last.content)
        print(f"[{name}] {dt:.1f}s, 产出 {len(text)} 字")

        # ── 故障注入(只给自杀测试用, 正常跑时 FAULT=None) ──
        if FAULT == "no_deliver":
            print(f"        [FAULT] 假装 {name} 没交活")
            return {"messages": []}                       # 制造"空集合"
        msg_name = None if FAULT == "no_name" else name    # 制造"没带 name"

        # ③ 带名消息: supervisor 一眼看出谁交的活
        return {"messages": [HumanMessage(content=f"[{name} 交活]\n{text}", name=msg_name)]}

    node.__name__ = name
    return node


worker_nodes = {
    "researcher": make_worker("researcher", [web_search], WORKER_PROMPTS["researcher"]),
    "writer":     make_worker("writer", [], WORKER_PROMPTS["writer"]),
    "reviewer":   make_worker("reviewer", [], WORKER_PROMPTS["reviewer"]),
}


# ══════════════════ 3. supervisor 节点 ══════════════════
SUPERVISOR_PROMPT = """你是写作工作室的主编(supervisor), 手下三位成员:

- researcher: 负责查资料, 只交资料要点, 不写正文
- writer:     负责把资料整合成一篇结构清晰的中文文章
- reviewer:   负责审稿, 指出文章的问题

你的职责: 每一步只决定「下一步交给谁」。不要替任何成员干活, 不要自己写文章。

标准流程: researcher 拿资料 → writer 成稿 → reviewer 审稿
         → 若 reviewer 提出重大问题, 可以再回 writer 修改
         → 都满意后输出 FINISH

判断依据: 看历史里带名的交活消息("[researcher 交活]" 这种)。
已经交过活的不必重复派。

★ 输出格式: 只输出一个 JSON 对象, 形如 {"next": "researcher", "reason": "还没有资料"}。
★ next 只能是 researcher / writer / reviewer / FINISH 之一。
"""                                              # ★ 改动②: 这两行不能省(json_mode 硬要求)


def supervisor_node(state: SuperState) -> dict:
    rounds = state.get("rounds", 0) + 1

    # ★ 改动④: 超限时【不再调 LLM】直接收口 —— 原版先调再判断, 那轮钱白花
    if rounds > MAX_ROUNDS:
        print(f"[supervisor] round={rounds} 超过上限 {MAX_ROUNDS} → 强制 FINISH")
        return {
            "next": "FINISH",
            "rounds": rounds,
            "messages": [("assistant", f"【已达最大轮数 {MAX_ROUNDS}, 强制结束】")],
        }

    msgs = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision = router.invoke(msgs)
    print(f"[supervisor] round={rounds}  next={decision.next}   ({decision.reason})")
    return {"next": decision.next, "rounds": rounds}


# ══════════════════ 4. 构图 ══════════════════
def route_from_supervisor(state: SuperState):
    nxt = state.get("next") or "FINISH"
    # 赌恶不赌善: 万一是野值, 兜到 END, 别让图崩在半路
    return nxt if (nxt in MEMBERS or nxt == "FINISH") else "FINISH"


def build_graph():
    b = StateGraph(SuperState)
    b.add_node("supervisor", supervisor_node)
    for name, node in worker_nodes.items():
        b.add_node(name, node)

    b.add_edge(START, "supervisor")

    path_map = {m: m for m in MEMBERS}
    path_map["FINISH"] = END
    b.add_conditional_edges("supervisor", route_from_supervisor, path_map)

    # ★ 回边: 每个 worker 干完都回到 supervisor 再决策(少了这条 = 各干各的)
    for name in MEMBERS:
        b.add_edge(name, "supervisor")

    return b.compile()


# ══════════════════ 5. 检查和主流程 ══════════════════
INITIAL = {
    "messages": [("user", "写一篇 300 字介绍 LangGraph 的文章, 要有事实依据")],
    "next": "",
    "rounds": 0,      # ★ 别忘了初始化, 否则首次取值报错
}


def run_checks(checks, verbose=True):
    """跑完全部检查, 不因第一条失败就停 —— 一次看到所有问题"""
    failures = []
    for name, fn in checks:
        try:
            detail = fn() or ""
            if verbose:
                print(f"  ✅ {name}" + (f"   — {detail}" if detail else ""))
        except AssertionError as e:
            failures.append((name, str(e)))
            if verbose:
                print(f"  ❌ {name}   — {e}")
    return failures


def build_checks(out, chain, mermaid):
    sup_calls = chain.count("supervisor")
    names_seen = [getattr(m, "name", None) for m in out["messages"]]
    workers_used = sorted({n for n in names_seen if n in MEMBERS})

    # 精确前缀匹配, 不用子串(正文里出现"交活"两字会误判)
    prefixes = tuple(f"[{w} 交活]" for w in MEMBERS)
    delivered = [m for m in out["messages"]
                 if str(getattr(m, "content", "")).startswith(prefixes)]

    checks = []

    # ── A. 交付物② 收口 ──
    def c_finish():
        assert out["next"] == "FINISH", f"没收口! next={out['next']!r}"
        return f"next=FINISH, rounds={out['rounds']}"
    checks.append(("A1 收口: next == FINISH", c_finish))

    # ── A2. 计数器自洽(只在此刻有意义: chain 与 out 出自同一次运行) ──
    def c_rounds():
        assert out["rounds"] == sup_calls, \
            f"rounds={out['rounds']} ≠ supervisor 实际执行 {sup_calls} 次(计数逻辑坏了)"
        return f"rounds == sup_calls == {sup_calls}"
    checks.append(("A2 计数器自洽: rounds == sup_calls", c_rounds))

    # ── B. worker → supervisor 通信 ──
    def c_delivered_nonempty():
        # ★ 先证非空: 否则下面的循环一句都不执行 = 没验证(真空真)
        assert delivered, ("一条带名交活消息都没有 → worker→supervisor 转消息那步没生效"
                          "(空集合会静默通过, 这是最容易漏的洞)")
        return f"{len(delivered)} 条交活消息"
    checks.append(("B1 交活消息非空", c_delivered_nonempty))

    def c_names_legal():
        # 只查 None 太弱: 名字打错成 reseachr 也过, 必须对着 MEMBERS 查
        bad = [getattr(m, "name", None) for m in delivered
               if getattr(m, "name", None) not in MEMBERS]
        assert not bad, f"交活消息 name 不在花名册里: {bad}"
        return f"name 全部合法: {[m.name for m in delivered]}"
    checks.append(("B2 交活消息 name ∈ MEMBERS", c_names_legal))

    def c_count_matches():
        # 推导不变量: 每次派活 → 恰好一个 worker → 恰好一条交活消息
        assert len(delivered) == sup_calls - 1, \
            f"交活消息 {len(delivered)} 条, 期望 {sup_calls - 1} 条(每次派活应恰好产出一条)"
        return f"{len(delivered)} == {sup_calls} - 1"
    checks.append(("B3 交活数 == sup_calls - 1", c_count_matches))

    # ── C. 交付物③ 拓扑导出 ──
    def c_mermaid():
        assert mermaid, "mermaid 导出为空"
        missing = [n for n in ["supervisor"] + MEMBERS if n not in mermaid]
        assert not missing, f"mermaid 里缺节点: {missing}"
        has_end = "__end__" in mermaid
        return f"节点齐全, END 节点{'有' if has_end else '未识别(请人眼核对)'}"
    checks.append(("C1 mermaid 含全部节点", c_mermaid))

    # ── D. 分模式检查 ──
    if MODE == "normal":
        def c_two_workers():
            assert len(workers_used) >= 2, \
                f"正常模式只用了 {workers_used}, 不满足 ≥2 个 worker(交付物①)"
            return f"workers_used={workers_used}"
        checks.append(("D1 [normal] 至少经过 2 个 worker", c_two_workers))

    else:
        def c_forced():
            forced = any("强制结束" in str(m.content) for m in out["messages"])
            assert forced, "兜底没生效: 消息里没有【强制结束】标记"
            return "看到【强制结束】标记"
        checks.append(("D1 [fallback] 出现强制结束标记", c_forced))

        def c_over_limit():
            assert out["rounds"] > MAX_ROUNDS, \
                f"兜底没生效: rounds={out['rounds']} 没超上限 {MAX_ROUNDS}"
            return f"rounds={out['rounds']} > MAX_ROUNDS={MAX_ROUNDS}"
        checks.append(("D2 [fallback] 轮数确实超限", c_over_limit))

        def c_exact_rounds():
            # ★ 确定性断言, 不依赖模型怎么走
            assert sup_calls == MAX_ROUNDS + 1, \
                (f"兜底路径轮数不对: sup_calls={sup_calls}, 期望 {MAX_ROUNDS + 1}。"
                 f"若第 1 轮就 FINISH, 说明这个测试用例失效了(任务太简单) —— "
                 f"换输入, 别改断言")
            return f"sup_calls={sup_calls} == MAX_ROUNDS+1"
        checks.append(("D3 [fallback] sup_calls == MAX_ROUNDS+1", c_exact_rounds))

    info = dict(sup_calls=sup_calls, delivered=len(delivered), workers_used=workers_used)
    return checks, info


def main():
    check_single_source_of_truth()          # 预检: 把 bug 拦在跑之前

    graph = build_graph()
    print(f"\n=== 开始跑 (MODE={MODE}, MAX_ROUNDS={MAX_ROUNDS}"
          f"{', FAULT=' + FAULT if FAULT else ''}) ===")

    # ★ 一次运行同时拿 chain(updates) 和终态(values) —— 不做跨运行对比
    chain = []
    final_state = None
    for mode, chunk in graph.stream(INITIAL, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node_name in chunk:
                chain.append(node_name)
        elif mode == "values":
            final_state = chunk          # values 每个事件是完整 state 快照
    out = final_state

    print(f"\n[流转链] {' → '.join(chain)}")

    print("\n=== 成品 ===")
    print(str(out["messages"][-1].content)[:400])
    print(f"[结果] next={out['next']}  rounds={out['rounds']}")

    # 交付物③: 导出拓扑图
    mermaid = graph.get_graph().draw_mermaid()
    path = "w7_supervisor_topology.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("```mermaid\n" + mermaid + "\n```\n")
    print(f"[导出] {path}  (贴到 mermaid.live 渲染, 和 Day1 图纸并排对比)")

    # 跑全部检查
    checks, info = build_checks(out, chain, mermaid)
    print(f"\n=== 检查 ({len(checks)} 项) ===")
    failures = run_checks(checks)

    print("\n" + "=" * 54)
    if failures:
        print(f"[结果] {len(failures)} 项未通过:")
        for n, msg in failures:
            print(f"   ✗ {n}\n     {msg}")
        sys.exit(1)          # 非零退出码 → 批量采样才能自动判断
    print(f"[结果] 全部 {len(checks)} 项通过  |  {info}")
    sys.exit(0)


if __name__ == "__main__":
    main()