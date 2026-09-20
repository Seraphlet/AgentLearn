"""Harness —— 五模块门面(Facade)

类比: 汽车中控台。
  发动机、变速箱、刹车各自是独立零件(五模块), 但你不能让司机一只手抓一个零件。
  Harness 就是中控台: 它自己不做任何事, 只负责把零件装到该在的位置。

★ 两条铁律:
  1) 这是唯一同时 import 五个模块的地方; 模块之间互不依赖(防循环导入)  [避坑表]
  2) LLM 从外部注入, 不在内部 new —— 换模型只改一处                  [避坑表]

五模块对应:
  Tool Registry      -> self.registry
  Permission Gate    -> self.gate
  Session Store      -> self.session
  Context Compaction -> self.compactor
  Tracer             -> self.tracer
"""
from __future__ import annotations

from .compact import ContextCompactor
from .session import SessionStore
from .trace import Tracer


class Harness:
    def __init__(self, *, registry=None, gate=None, session=None,
                 compactor=None, tracer=None, llm=None):
        # ★ 全部可注入; 没给的就用最保守的默认(内存版), 保证开箱可用
        self.registry = registry
        self.gate = gate
        self.session = session if session is not None else SessionStore(None)
        self.compactor = compactor if compactor is not None else ContextCompactor()
        self.tracer = tracer if tracer is not None else Tracer(None)
        self.llm = llm          # ★ 依赖注入; 绝不在内部 new ChatOpenAI

    # ---------------------------------------------------------------- 状态
    def describe(self):
        """验收④: 输出五模块状态"""
        return {
            "registry": self._registry_state(),
            "gate": self._gate_state(),
            "session": {
                "present": self.session is not None,
                "backend": "memory" if getattr(self.session, "path", None) is None else "jsonl",
                "path": getattr(self.session, "path", None),
                "threads": len(self.session.list_threads()) if self.session else 0,
            },
            "compaction": {
                "present": self.compactor is not None,
                "max_messages": getattr(self.compactor, "max_messages", None),
                "keep_recent": getattr(self.compactor, "keep_recent", None),
                "compactions": getattr(self.compactor, "compactions", 0),
            },
            "trace": {
                "present": self.tracer is not None,
                "path": getattr(self.tracer, "path", None),
                "spans": len(getattr(self.tracer, "spans", [])),
                "total_ms": self.tracer.total_ms() if self.tracer else 0,
            },
            "llm": {
                "injected": self.llm is not None,
                "model": getattr(self.llm, "model_name", None),
            },
        }

    def describe_text(self):
        d = self.describe()
        r, g = d["registry"], d["gate"]
        s, c, t = d["session"], d["compaction"], d["trace"]
        return "\n".join([
            "Harness 五模块状态",
            f"  ① Tool Registry      : {'OK' if r['present'] else '未接'} "
            f"工具 {r['tools']} 个 {r['names']}",
            f"  ② Permission Gate    : {'OK' if g['present'] else '未接'} "
            f"deny={g.get('deny')} allow={g.get('allow')}",
            f"  ③ Session Store      : {'OK' if s['present'] else '缺'} "
            f"{s['backend']} 会话 {s['threads']} 个",
            f"  ④ Context Compaction : {'OK' if c['present'] else '缺'} "
            f"max={c['max_messages']} keep={c['keep_recent']} 已压 {c['compactions']} 次",
            f"  ⑤ Tracer             : {'OK' if t['present'] else '缺'} "
            f"{t['spans']} 个 span, 顶层总耗时 {t['total_ms']}ms",
        ])

    # ---------------------------------------------------------------- 便利方法
    def save(self, thread_id, messages, meta=None):
        """存全量(不压缩) —— 压缩发生在 save_messages 里, 见注释"""
        return self.session.append(thread_id, messages, meta=meta)

    def load_for_llm(self, thread_id, summarize=None):
        """★ 读时压: 拿这份发给模型; 存档一个字没动"""
        msgs = self.session.messages(thread_id)
        view = self.compactor.maybe_compact(msgs, summarize=summarize)
        return view if view is not None else msgs

    # ---------------------------------------------------------------- 内部
    def _registry_state(self):
        r = self.registry
        if r is None:
            return {"present": False, "tools": 0, "names": []}
        names = []
        for attr in ("list", "names", "tool_names"):
            fn = getattr(r, attr, None)
            if fn is None:
                continue
            try:
                names = sorted(fn()) if callable(fn) else sorted(fn)
                break
            except Exception:
                continue
        return {"present": True, "tools": len(names), "names": names}

    def _gate_state(self):
        g = self.gate
        if g is None:
            return {"present": False, "allow": [], "deny": []}
        return {
            "present": True,
            "allow": sorted(getattr(g, "allow", []) or []),
            "deny": sorted(getattr(g, "deny", []) or []),
            "audit_path": getattr(g, "audit_path", None),
        }