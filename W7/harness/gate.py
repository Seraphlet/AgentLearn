"""Permission Gate —— 门禁闸机。
唯一决定"这个调用能不能发"的地方。策略只有一份, 审计只有一个卡点。

【审计词表】定死这一套, 不许再混用:
    executed            放行且真的执行成功
    denied              被策略拦下, 未执行
    approval_granted    人工批准(之后会执行, 并再记一条 executed)
    approval_rejected   人工拒绝, 未执行
    error               放行了, 但工具自己抛异常
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEED_APPROVAL = "need_approval"


@dataclass
class GateResult:
    decision: Decision
    reason: str = ""


# 审计日志不许泄露密钥 [3]
SENSITIVE = ("key", "token", "secret", "password", "passwd", "pwd",
             "credential", "auth", "cookie", "session")


def redact(args: dict) -> dict:
    """参数摘要: 敏感字段打码, 长值截断"""
    out = {}
    for k, v in (args or {}).items():
        if any(s in str(k).lower() for s in SENSITIVE):
            out[k] = "***"
        else:
            s = str(v)
            out[k] = s if len(s) <= 60 else s[:57] + "..."
    return out


class PermissionGate:
    def __init__(self, registry, *, deny=(), allow=(), audit_path="audit.jsonl"):
        self.registry = registry
        self.deny = set(deny)        # 黑名单: 最高优先
        self.allow = set(allow)      # 危险工具白名单: 免审批直接放行
        self.audit_path = Path(audit_path)

    # ------------- 策略(纯函数式, 不执行任何工具, 可离线单测) -------------
    def check(self, tool_name: str, actor: str = "agent") -> GateResult:
        if not self.registry.exists(tool_name):
            return GateResult(Decision.DENY, f"工具未注册: {tool_name}")
        if tool_name in self.deny:
            return GateResult(Decision.DENY, "命中黑名单")
        if self.registry.is_dangerous(tool_name):        # ★ 不读 spec 内部字段
            if tool_name in self.allow:
                return GateResult(Decision.ALLOW, "危险工具已显式放行")
            return GateResult(Decision.NEED_APPROVAL, "危险工具需人工审批")
        return GateResult(Decision.ALLOW, "普通工具")

    # --------------------------- 审计 ---------------------------
    def audit(self, actor: str, tool: str, decision: str, *,
              reason: str = "", args: dict | None = None) -> dict:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "tool": tool,
            "decision": decision,
            "reason": reason,
            "args": redact(args or {}),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec