"""test_harness.py —— 离线单测, 0 成本, 不依赖 LLM。

跑法(在 W7 目录):
    python test_harness.py
"""
from __future__ import annotations

import json
import os

from harness.registry import ToolRegistry, DuplicateToolError, ToolError
from harness.gate import PermissionGate, Decision
from harness.executor import call_tool, ExecutorBug

AUDIT = "test_audit.jsonl"
CALLS: list[str] = []          # ★ 副作用账本: 只有它知道"工具到底跑没跑"


# ----------------------------------------------------------------------
def build():
    """每个用例一套干净的 registry + gate"""
    CALLS.clear()
    if os.path.exists(AUDIT):
        os.remove(AUDIT)

    reg = ToolRegistry()

    @reg.register(tags=["web"])
    def read_news(url: str = ""):
        """读取公开网页(只读)"""
        CALLS.append("read_news")
        return f"内容 from {url}"

    @reg.register(level="dangerous", tags=["fs"])
    def delete_file(path: str):
        """删除文件(不可逆)"""
        CALLS.append("delete_file")
        return f"已删除 {path}"

    return reg, PermissionGate(reg, audit_path=AUDIT)


# ---------------------------- 验收① Registry ----------------------------
def t_registry():
    reg, _ = build()
    assert reg.list() == ["delete_file", "read_news"], reg.list()
    assert reg.exists("read_news") and not reg.exists("nope")
    assert reg.is_dangerous("delete_file") is True
    assert reg.is_dangerous("read_news") is False
    assert "delete_file" in [t.name for t in reg.search("fs")]
    assert reg.search("网页") != []
    assert reg.search("淘汰场") == []
    assert reg.get("read_news").description == "读取公开网页(只读)"


def t_registry_dup_and_unknown():
    reg, _ = build()
    try:
        reg.register("read_news")(lambda: 1)
        raise AssertionError("重名应报错, 但没报")
    except DuplicateToolError:
        pass
    try:
        reg.get("不存在")
        raise AssertionError("未注册应报错, 但没报")
    except ToolError:
        pass
    try:
        reg.register("bad", level="乱写")(lambda: 1)
        raise AssertionError("非法 level 应报错, 但没报")
    except ToolError:
        pass


def t_bare_decorator():
    reg = ToolRegistry()

    @reg.register
    def ping():
        """打个招呼"""
        return "pong"

    assert reg.list() == ["ping"], reg.list()
    assert reg.get("ping").description == "打个招呼"
    assert reg.invoke("ping") == "pong"


# ---------------------------- 验收② Gate 三态 ----------------------------
def t_gate_three_states():
    reg, gate = build()
    gate.deny = {"delete_file"}
    assert gate.check("read_news", "u").decision is Decision.ALLOW
    assert gate.check("delete_file", "u").decision is Decision.DENY
    assert gate.check("没这个工具", "u").decision is Decision.DENY

    reg2, gate2 = build()
    assert gate2.check("delete_file", "u").decision is Decision.NEED_APPROVAL
    gate2.allow = {"delete_file"}
    assert gate2.check("delete_file", "u").decision is Decision.ALLOW


def t_gate_blacklist_beats_whitelist():
    """策略决定: deny 优先于 allow, 否则"临时拉黑"会被白名单静默覆盖"""
    reg, gate = build()
    gate.deny, gate.allow = {"delete_file"}, {"delete_file"}
    assert gate.check("delete_file", "u").decision is Decision.DENY


# ------------------- 验收④ ★ 拒绝时"确实没执行" -------------------
def t_deny_really_not_executed():
    # 正例: 允许 → 账本必须留下记录(先证明这条路真的通)
    reg, gate = build()
    r = call_tool(reg, gate, "read_news", {"url": "x"}, actor="u")
    assert CALLS == ["read_news"], f"允许路径没执行: {CALLS}"
    assert r["ok"] and r["executed"], (r, CALLS)

    # 反例1: 黑名单 → 账本必须为空
    reg, gate = build()
    gate.deny = {"delete_file"}
    r = call_tool(reg, gate, "delete_file", {"path": "x"}, actor="u")
    assert r["ok"] is False and r["executed"] is False, r
    assert CALLS == [], f"★ 拒绝了却还是执行了: {CALLS}"

    # 反例2: 人工拒绝 → 账本必须为空
    reg, gate = build()
    r = call_tool(reg, gate, "delete_file", {"path": "x"}, actor="u",
                  approve=lambda t, why: False)
    assert r["ok"] is False and CALLS == [], f"★ 人工拒绝了却还是执行了: {CALLS}"

    # 反例3: 人工批准 → 账本必须有记录(证明审批这条路真的通)
    reg, gate = build()
    r = call_tool(reg, gate, "delete_file", {"path": "x"}, actor="u",
                  approve=lambda t, why: True)
    assert CALLS == ["delete_file"], CALLS
    assert r["ok"] and r["executed"], r


def t_never_silent_none():
    """★ 任何路径都不许返回 None —— 这次的 bug 就是隐式 None"""
    reg, gate = build()

    cases = [
        ("read_news", {}, None),
        ("delete_file", {"path": "x"}, None),
        ("delete_file", {"path": "x"}, lambda t, w: False),
        ("delete_file", {"path": "x"}, lambda t, w: True),
        ("没这个工具", {}, None),
    ]
    for name, a, ap in cases:
        r = call_tool(reg, gate, name, a, actor="u", approve=ap)
        assert isinstance(r, dict), f"{name} 返回了 {r!r}, 不是 dict"
        assert {"ok", "executed"} <= set(r), f"{name} 少了字段: {r}"


def t_tool_exception_is_recorded():
    """工具自己抛异常 → 必须记账 + 返回 dict, 不许消失得无痕"""
    CALLS.clear()
    if os.path.exists(AUDIT):
        os.remove(AUDIT)
    reg = ToolRegistry()

    @reg.register
    def boom():
        """一定会炸的工具"""
        CALLS.append("boom")
        raise ValueError("炸了")

    gate = PermissionGate(reg, audit_path=AUDIT)
    r = call_tool(reg, gate, "boom", {}, actor="u")

    assert r["ok"] is False and r["executed"] is False, r
    assert "ValueError" in r["error"], r
    assert CALLS == ["boom"], CALLS
    assert any(json.loads(l)["decision"] == "error"
               for l in open(AUDIT, encoding="utf-8")), "异常没被记账"


# ---------------------------- 验收③ 审计日志 ----------------------------
def t_audit():
    reg, gate = build()
    gate.deny = {"delete_file"}
    call_tool(reg, gate, "read_news", {}, actor="user-1")
    call_tool(reg, gate, "delete_file", {"path": "x"}, actor="user-1")

    recs = [json.loads(l) for l in open(AUDIT, encoding="utf-8")]
    for r in recs:
        assert {"ts", "actor", "tool", "decision"} <= set(r), r
    decisions = {r["decision"] for r in recs}
    assert "denied" in decisions, f"审计没记录被拦下的调用: {decisions}"
    assert "executed" in decisions, decisions           # ← 上次就挂在这
    assert any(r["actor"] == "user-1" for r in recs), "actor 没记上"


def t_audit_redacts_secrets():
    """审计不许泄露密钥 [3]: 只记参数摘要"""
    reg = ToolRegistry()

    @reg.register
    def call_api(api_key: str, query: str = ""):
        """调用外部 API"""
        return "ok"

    gate = PermissionGate(reg, audit_path=AUDIT)
    if os.path.exists(AUDIT):
        os.remove(AUDIT)
    call_tool(reg, gate, "call_api",
              {"api_key": "sk-真密钥-123", "query": "天气"}, actor="u")

    raw = open(AUDIT, encoding="utf-8").read()
    assert "sk-真密钥-123" not in raw, "★ 审计泄露了密钥"
    rec = json.loads(raw.strip().splitlines()[0])
    assert rec["args"]["api_key"] == "***", rec
    assert rec["args"]["query"] == "天气", rec


# ----------------------------------------------------------------------
def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("t_") and callable(v)]
    passed = 0
    for f in tests:
        try:
            f()
            print(f"  OK   {f.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n  {passed}/{len(tests)} 通过   (审计日志: {AUDIT})")


if __name__ == "__main__":
    _run()