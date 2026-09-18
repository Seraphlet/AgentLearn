"""call_tool —— 唯一的工具调用入口。
所有调用必须穿过 Gate, 不允许旁路。

【四条纪律】
1. 每个分支都 audit, 每个分支都 return dict —— 永不落空成隐式 None
2. 引入显式 allowed 放行位 —— 不放行就绝不许走到执行段
3. 出现没覆盖的决策 -> 抛 ExecutorBug, 不许静默
4. 异常也要 audit + return dict —— 可以处理, 但不许消失得无痕
"""
from __future__ import annotations

from .gate import Decision


class ExecutorBug(Exception):
    """executor 自己的错误(不是工具的错误), 必须炸出来"""


def _norm(decision) -> Decision:
    """把裸字符串 'allow' 或 Decision 枚举统一成 Decision。不认识的值 -> 炸。"""
    if isinstance(decision, Decision):
        return decision
    try:
        return Decision(decision)
    except ValueError as e:
        raise ExecutorBug(f"门禁返回了未知决策: {decision!r}") from e


def call_tool(registry, gate, tool_name: str, args: dict | None = None, *,
              actor: str = "agent", approve=None) -> dict:
    """
    返回统一形状: {"ok": bool, "executed": bool, "result"/"error": ...}
    "executed" 明确写在返回值里 —— 不逼调用方靠副作用去猜。
    """
    args = args or {}
    res = gate.check(tool_name, actor)
    decision = _norm(res.decision)          # 不靠 is 硬比枚举
    allowed = False                         # ★ 显式放行位, 默认不放行

    if decision is Decision.DENY:
        gate.audit(actor, tool_name, "denied", reason=res.reason, args=args)
        return {"ok": False, "executed": False,
                "error": f"被拒绝: {res.reason}"}

    elif decision is Decision.ALLOW:
        allowed = True

    elif decision is Decision.NEED_APPROVAL:
        granted = bool(approve(tool_name, res.reason)) if approve else False
        if not granted:
            gate.audit(actor, tool_name, "approval_rejected",
                       reason=res.reason, args=args)
            return {"ok": False, "executed": False, "error": "人工拒绝"}
        gate.audit(actor, tool_name, "approval_granted",
                   reason=res.reason, args=args)
        allowed = True

    else:
        raise ExecutorBug(f"未处理的门禁结果: {decision!r}")

    if not allowed:                          # ★ 兜底闸
        raise ExecutorBug("allowed 仍为 False 却走到了执行段")

    try:
        result = registry.invoke(tool_name, args)        # ★ 不读 .fn
    except Exception as e:
        gate.audit(actor, tool_name, "error",
                   reason=f"{type(e).__name__}: {e}", args=args)
        return {"ok": False, "executed": False,
                "error": f"{type(e).__name__}: {e}"}

    gate.audit(actor, tool_name, "executed", args=args)
    return {"ok": True, "executed": True, "result": result}