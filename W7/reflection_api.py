"""边界②: 反思循环的窄接口。

这是 Day6 唯一"新写的设计"。内部的 8 个 state 字段全部私有。
调用方(supervisor 的 writer 节点)只看见 5 个字段。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from W7.reflection_graph import stop_reason

@dataclass
class ReflectionResult:
    draft: str                  # 终稿(自动取历史最优稿)
    score: int                  # 该稿评分
    rounds: int                 # 实际修改次数
    passed: bool                # 是否真达标(而非撞次数上限)
    notes: list = field(default_factory=list)   # 未解决的硬伤; 空 = 干净

    def write_wrih_reflection(topic:str,*,pass_score:int =8,max_revise:int=3,graph=None) ->ReflectionResult:
        """输入主题, 返回带自省闭环的终稿。内部实现对调用方不透明。"""
        if graph is None:
            raise RuntimeError("graph 必须注入(依赖注入：换实现不改调用方)")

        out = graph.invoke({
            "topic": topic, "draft": "", "score": 0, "issues": [],
        "revise_count": 0, "trajectory": [], "best_draft": "", "best_score": 0,
        })

        def stop_reason():
            if out["score"] >= pass_score:
                return "passed"
            if out["revise_count"] >= max_revise:
                return "max_revise"
            return "stalled"
        # ★ 关键: 不是"抄字段", 是【判定】。passed 由程序算, 不信任模型自评
        passed = stop_reason()=="passed"
        notes=[] if passed else  [f"未达标下线(原因: {stop_reason()})"]
        notes.extend(f"硬伤: {x}" for x in (out.get("fatal_issues") or []))
        return ReflectionResult(
        draft=out.get("best_draft") or out.get("draft", ""),
        score=int(out.get("score", 0)),
        rounds=int(out.get("revise_count", 0)),
        passed=passed,
        notes=notes,
    )

class FakeGraph:
    def __init__(self, out): self.out = out
    def invoke(self, _): return self.out

from reflection_api import write_with_reflection
# 场景1: 达标
r = write_with_reflection("x", graph=FakeGraph(
    {"score": 8, "revise_count": 1, "best_draft": "终稿", "draft": "初稿"}))
assert r.passed and r.draft == "终稿" and r.notes == []

# 场景2: 撞上限, 未达标  ← ★ 这条最重要
r = write_with_reflection("x", graph=FakeGraph(
    {"score": 7, "revise_count": 3, "best_draft": "终稿", "draft": "初稿"}))
assert not r.passed and r.notes, "未达标却没留 notes → supervisor 会以为成功"
print("OK")    