"""Day5 验收: Session Store + Context Compaction + Tracer + Harness 门面

    python test_harness2.py
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from harness.session import SessionStore
from harness.compact import ContextCompactor, _safe_cut
from harness.trace import Tracer
from harness.facade import Harness

TMP = os.path.join(HERE, "_tmp_d5")
T = []          # 用例表


def case(fn):
    T.append(fn)
    return fn


def _dialog(n_rounds, with_system=True):
    m = []
    if with_system:
        m.append({"role": "system", "content": "你是助手"})
    for i in range(n_rounds):
        m.append({"role": "user", "content": f"问题{i}"})
        m.append({"role": "assistant", "content": f"回答{i}"})
    return m


def _tool_dialog():
    """一个含 tool_call / tool 结果的块: 9 条(system + 4 对 + 1)"""
    return [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "查天气"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "get_weather", "args": {"city": "北京"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "晴 25度"},
        {"role": "assistant", "content": "北京晴 25 度"},
        {"role": "user", "content": "那上海呢"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "name": "get_weather", "args": {"city": "上海"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "阴 22度"},
        {"role": "assistant", "content": "上海阴 22 度"},
    ]


# ============================================================ Session
@case
def t_session_round_trip():
    s = SessionStore(None)
    s.append("t1", [{"role": "user", "content": "你好"}])
    assert s.messages("t1") == [{"role": "user", "content": "你好"}]
    assert s.latest("t1")["seq"] == 0


@case
def t_session_restart_recover():
    """验收①: 重启进程会话可恢复(换个新实例 = 模拟重启)"""
    p = os.path.join(TMP, "restart.jsonl")
    SessionStore(p).append(
        "chat-1",
        [{"role": "user", "content": "我叫小明"},
         {"role": "assistant", "content": "你好小明"}],
        meta={"rounds": 1})
    b = SessionStore(p)                    # ★ 全新实例, 内存全丢
    msgs = b.messages("chat-1")
    assert len(msgs) == 2 and msgs[0]["content"] == "我叫小明", "重启后内容丢了"
    assert b.latest("chat-1")["meta"]["rounds"] == 1


@case
def t_session_multi_thread_isolated():
    """多会话隔离: A 存的东西 B 取不到"""
    s = SessionStore(os.path.join(TMP, "multi.jsonl"))
    s.append("A", [{"role": "user", "content": "A 的密话"}])
    s.append("B", [{"role": "user", "content": "B 的密话"}])
    assert len(s.load("A")) == 1 and len(s.load("B")) == 1
    assert s.messages("A")[0]["content"] == "A 的密话"
    assert s.list_threads() == ["A", "B"]


@case
def t_session_survives_corrupt_tail():
    """★ 崩溃恢复: 尾部有写坏的行, 读取不能炸, 且要计数, 之后还能继续写"""
    p = os.path.join(TMP, "corrupt.jsonl")
    s = SessionStore(p)
    s.append("t1", [{"role": "user", "content": "第一条"}])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"ts":123,"thread_id":"t1","mess')     # 模拟"写一半断电"
    s2 = SessionStore(p)
    assert len(s2.load("t1")) == 1, "坏行应该被跳过"
    assert s2.bad_lines == 1, f"坏行该被计数, 实际 {s2.bad_lines}"
    s2.append("t1", [{"role": "user", "content": "第二条"}])
    assert len(SessionStore(p).load("t1")) == 2, "坏行之后应该还能继续写"


@case
def t_session_delete_is_atomic_and_complete():
    s = SessionStore(os.path.join(TMP, "del.jsonl"))
    s.append("A", [{"role": "user", "content": "a"}])
    s.append("B", [{"role": "user", "content": "b"}])
    s.delete("A")
    assert s.list_threads() == ["B"]
    assert not os.path.exists(s.path + ".tmp"), "临时文件该被 replace 掉, 不该残留"


# ============================================================ Compactor
@case
def t_compact_below_threshold_returns_none():
    """避坑: 不超预算绝不动手(返回 None, 不是返回原列表)"""
    c = ContextCompactor(max_messages=20, keep_recent=4)
    assert c.maybe_compact(_dialog(3)) is None          # 7 条 < 20
    assert c.compactions == 0, "没超预算却计了一次压缩"


@case
def t_compact_shrinks():
    """避坑: 压缩后必须【变少】, 不是变多"""
    msgs = _dialog(20)                                  # 41 条
    c = ContextCompactor(max_messages=20, keep_recent=6)
    out = c.maybe_compact(msgs)
    assert out is not None
    assert len(out) < len(msgs), f"压缩后反而变多: {len(msgs)} -> {len(out)}"
    assert c.compactions == 1


@case
def t_compact_keeps_system_and_recent():
    msgs = _dialog(20)
    out = ContextCompactor(20, 6).maybe_compact(msgs)
    assert out[0]["role"] == "system" and out[0]["content"] == "你是助手"
    assert [m["content"] for m in out[-6:]] == [m["content"] for m in msgs[-6:]], \
        "最近 6 条被动过"


@case
def t_compact_uses_injected_summarizer():
    """依赖注入: 有 LLM 用 LLM, 没有用机械摘要"""
    seen = {}

    def fake(old):
        seen["n"] = len(old)
        return "【假摘要】早期聊了天气"

    out = ContextCompactor(20, 6).maybe_compact(_dialog(20), summarize=fake)
    assert seen.get("n", 0) > 0, "注入的 summarize 没被调用"
    assert any("假摘要" in (m.get("content") or "") for m in out), "摘要没进消息"


@case
def t_compact_tool_pair_not_split():
    """★★ 压缩绝不能把 assistant.tool_calls 和它的 tool 结果拆开"""
    msgs = _tool_dialog() * 4                            # 36 条
    c = ContextCompactor(max_messages=20, keep_recent=6)

    # 先证明"这个数据真的会切在 tool 上" —— 否则这条断言没牙
    body = msgs[1:]
    raw_cut = len(body) - 6
    assert body[raw_cut]["role"] == "tool", \
        f"构造的数据没切到 tool 上(切到了 {body[raw_cut]['role']}), 这条测试就白测了"

    out = c.maybe_compact(msgs)
    assert out is not None
    called = {tc["id"] for m in out for tc in (m.get("tool_calls") or [])}
    for m in out:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in called, \
                f"孤儿 tool 结果: {m['tool_call_id']} 的 tool_calls 被切掉了"


@case
def t_safe_cut_moves_back_to_assistant():
    msgs = [{"role": "user", "c": 0}, {"role": "assistant", "c": 1},
            {"role": "tool", "c": 2}, {"role": "tool", "c": 3}]
    assert _safe_cut(msgs, 3) == 1, "_safe_cut 没退到 tool 对的开头"
    assert _safe_cut(msgs, 0) == 0, "_safe_cut 不该退到负数"


@case
def t_compact_is_projection_not_destroy():
    """★ 核心决策: 压缩是投影 —— 存档一个字没动"""
    s = SessionStore(os.path.join(TMP, "proj.jsonl"))
    msgs = _dialog(20)
    s.append("t1", msgs)
    c = ContextCompactor(20, 6)
    view = c.maybe_compact(s.messages("t1"))
    assert view is not None and len(view) < len(msgs)
    assert len(s.messages("t1")) == len(msgs), "★ 压缩动了存档!(应该存全量)"


# ============================================================ Tracer
@case
def t_trace_return_not_eaten():
    """★ 避坑: 装饰器吃掉返回值 -> 图直接报错"""
    tr = Tracer(None)

    @tr.trace("node_a")
    def node_a(state):
        return {"messages": ["hi"]}

    assert node_a({}) == {"messages": ["hi"]}, "返回值被吃了"


@case
def t_trace_records_node_ms_ok():
    """验收③: trace 有 node / ms / ok"""
    tr = Tracer(None)

    @tr.trace("node_a")
    def node_a(state):
        return 1

    node_a({})
    s = tr.spans[-1]
    assert s["name"] == "node_a" and s["ok"] is True
    assert isinstance(s["ms"], (int, float)) and s["ms"] >= 0


@case
def t_trace_exception_recorded_and_reraised():
    """异常也要记账, 而且必须继续往上抛(不能吞)"""
    tr = Tracer(None)

    @tr.trace("boom")
    def boom(state):
        raise ValueError("炸了")

    try:
        boom({})
        raise AssertionError("异常被吞了!")
    except ValueError:
        pass
    s = tr.spans[-1]
    assert s["ok"] is False and "ValueError" in s["err"], f"异常没记上: {s}"


@case
def t_trace_nesting_parent_child():
    """嵌套调用: 子 span 的 parent 指向父 span"""
    tr = Tracer(None)

    @tr.trace("inner")
    def inner():
        return 1

    @tr.trace("outer")
    def outer():
        return inner()

    outer()
    by = {s["name"]: s for s in tr.spans}
    assert len(tr.spans) == 2
    assert by["outer"]["parent_id"] is None
    assert by["inner"]["parent_id"] == by["outer"]["span_id"], "父子关系没接上"


@case
def t_trace_jsonl_written():
    p = os.path.join(TMP, "trace.jsonl")
    tr = Tracer(p)

    @tr.trace("n1")
    def n1():
        return 1

    n1()
    lines = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert lines and lines[0]["name"] == "n1" and "ms" in lines[0]
    assert tr.total_ms() >= 0


# ============================================================ 门面 / 铁律
@case
def t_facade_describe_five_modules():
    """验收④: describe() 输出五模块状态"""
    h = Harness(session=SessionStore(None),
                compactor=ContextCompactor(max_messages=10, keep_recent=3),
                tracer=Tracer(None))
    d = h.describe()
    for k in ("registry", "gate", "session", "compaction", "trace"):
        assert k in d, f"describe() 少了 {k}"
    assert d["compaction"]["max_messages"] == 10
    assert d["trace"]["present"] is True
    assert d["llm"]["injected"] is False, "不该凭空 new 一个 LLM"


@case
def t_facade_llm_must_be_injected():
    """避坑: Harness 里直接 new LLM -> 换模型要改多处"""
    class FakeLLM:
        model_name = "fake-model"

    h = Harness(llm=FakeLLM())
    assert h.describe()["llm"] == {"injected": True, "model": "fake-model"}


@case
def t_day5_modules_have_no_cycle():
    """避坑: 模块互相 import 成环 -> Day5 三模块必须互不依赖"""
    bad = {}
    for name in ("session", "compact", "trace"):
        p = os.path.join(HERE, "harness", name + ".py")
        tree = ast.parse(open(p, encoding="utf-8").read())
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("harness"):
                    deps.add(node.module.split(".")[-1])
                elif node.level:
                    deps.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                deps.update(a.name.split(".")[-1] for a in node.names
                            if a.name.startswith("harness."))
        deps.discard(name)
        if deps:
            bad[name] = sorted(deps)
    assert not bad, f"Day5 模块之间互相依赖了: {bad}"


@case
def t_end_to_end_store_compact_trace():
    """三件套串起来: 存全量 -> 读时压 -> 每步有 trace"""
    h = Harness(session=SessionStore(os.path.join(TMP, "e2e.jsonl")),
                compactor=ContextCompactor(8, 3),
                tracer=Tracer(os.path.join(TMP, "e2e_trace.jsonl")))

    @h.tracer.trace("agent")
    def agent(messages):
        return messages + [{"role": "assistant", "content": "ok"}]

    msgs = []
    for i in range(10):
        msgs = agent(msgs + [{"role": "user", "content": f"q{i}"}])
        h.save("t1", msgs, meta={"round": i})           # ★ 存全量

    assert len(h.session.messages("t1")) == 20, "存的应该是全量"
    view = h.load_for_llm("t1")                          # ★ 读时压
    assert len(view) < 20, "读时没压下去"
    assert len(h.session.messages("t1")) == 20, "★ 读时压不该动存档"
    assert len(h.tracer.spans) == 10 and all(s["ok"] for s in h.tracer.spans)


# ============================================================ runner
def main() -> int:
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)

    print(f"跑 {len(T)} 个用例  (Python {sys.version.split()[0]})")
    ok, fails = 0, []
    for fn in T:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}")
            print(f"       {type(e).__name__}: {e}")
            fails.append(fn.__name__)

    # ★ 顺手把 trace 树打出来看一眼
    tr = Tracer(None)

    @tr.trace("supervisor")
    def sup():
        return inner()

    @tr.trace("worker")
    def inner():
        return 1

    sup()
    print("\n  --- trace 树长这样 ---")
    for line in tr.summary().splitlines():
        print("  " + line)

    print(f"\n  {ok}/{len(T)} 通过" + (f"   失败: {fails}" if fails else ""))
    return 0 if ok == len(T) else 1


if __name__ == "__main__":
    sys.exit(main())