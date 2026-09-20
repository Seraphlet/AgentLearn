"""Tracer —— 调用链追踪(Harness 模块⑤)

类比: 外卖配送轨迹。
  点外卖能看到: 商家接单 -> 骑手到店 -> 配送中 -> 已送达。
  Tracer 给 Agent 的每一步盖一个"开始 / 结束 / 耗时 / 成功没有"。

★ 和审计日志(audit)的区别 —— 两者都要, 不能互相替代:
    audit : 谁做了什么【决策】(安全视角, 用于追责/合规)
    trace : 每一步花了多久、成功没有、谁调用了谁(性能/排障视角)
  一句话: audit 回答"该不该做", trace 回答"为什么慢 / 为什么炸"。

★ 关键陷阱(踩过): 装饰器把返回值吃掉 -> 图直接报错。
  因为 LangGraph 节点必须返回 dict, 装饰器里少写一句 return 就变成返回 None。
  修法: try 里 return out, finally 里记日志 —— 两者互不干扰。
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import time

# ★ 用 contextvars 存"当前正在执行的 span", 嵌套调用就能自动接上父子关系
_current = contextvars.ContextVar("current_span", default=None)


class Tracer:
    def __init__(self, path=None):
        # path=None       -> 只在内存(测试用)
        # path="trace.jsonl" -> 落盘
        self.path = path
        self.spans = []
        self._seq = 0

    # ---------------------------------------------------------------- 落一条
    def _next_seq(self):
        self._seq += 1
        return self._seq

    def new_span_id(self):
        return f"s{self._next_seq()}"

    def record(self, name, *, ms, ok, err=None, parent_id=None, span_id=None):
        span = {
            "span_id": span_id or self.new_span_id(),
            "parent_id": parent_id,
            "name": name,
            "ms": round(float(ms), 3),
            "ok": bool(ok),
            "err": err,
            "ts": time.time(),
        }
        self.spans.append(span)
        if self.path:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(span, ensure_ascii=False) + "\n")
            except OSError:
                pass        # ★ 记日志失败不能影响主流程
        return span

    # ---------------------------------------------------------------- 装饰器
    def trace(self, name=None):
        """★ 记录耗时 + 成败, 并且必须把原返回值透传出去"""
        def deco(fn):
            label = name or getattr(fn, "__name__", "anonymous")

            @functools.wraps(fn)
            def wrapper(*a, **kw):
                parent = _current.get()
                parent_id = parent["span_id"] if parent else None
                holder = {"span_id": self.new_span_id(), "name": label}
                token = _current.set(holder)     # ← 子调用读这个, 建立父子关系
                t0 = time.perf_counter()
                ok, err = True, None
                try:
                    out = fn(*a, **kw)
                    return out                    # ★★★ 这一行绝不能漏
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {e}"
                    raise                         # ★ 记完还得继续抛, 不能吞
                finally:
                    _current.reset(token)
                    self.record(label, ms=(time.perf_counter() - t0) * 1000,
                                ok=ok, err=err, parent_id=parent_id,
                                span_id=holder["span_id"])

            return wrapper
        return deco

    # ---------------------------------------------------------------- 输出
    def summary(self):
        """把 span 打印成树"""
        children = {}
        for s in self.spans:
            children.setdefault(s["parent_id"], []).append(s)
        lines = []

        def walk(pid, depth):
            for s in children.get(pid, []):
                mark = "OK " if s["ok"] else "ERR"
                extra = f"  <- {s['err']}" if s["err"] else ""
                lines.append(f"{'  ' * depth}{mark} {s['name']}  {s['ms']}ms{extra}")
                walk(s["span_id"], depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def total_ms(self):
        return round(sum(s["ms"] for s in self.spans if s["parent_id"] is None), 3)

    def clear(self):
        self.spans.clear()
        self._seq = 0