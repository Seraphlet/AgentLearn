from harness.registry import ToolRegistry
from harness.gate import PermissionGate
from harness.executor import call_tool
import os

CALLS = []
reg = ToolRegistry()

@reg.register(tags=["web"])
def read_news(url: str = ""):
    """读取公开网页(只读)"""
    CALLS.append("read_news")
    return "ok"

spec = reg.get("read_news")
print("① ToolSpec 字段:", list(vars(spec)) if hasattr(spec, "__dict__") else "无 __dict__")
print("② 有 .fn:", hasattr(spec, "fn"), "| 有 .func:", hasattr(spec, "func"))

AUDIT = "diag_audit.jsonl"
if os.path.exists(AUDIT): os.remove(AUDIT)
gate = PermissionGate(reg, audit_path=AUDIT)
r = call_tool(reg, gate, "read_news", {"url": "x"}, actor="u")

print("③ call_tool 返回值:", repr(r))
print("④ 账本 CALLS:", CALLS)
print("⑤ 审计文件:", open(AUDIT, encoding="utf-8").read().strip() if os.path.exists(AUDIT) else "(不存在)")
