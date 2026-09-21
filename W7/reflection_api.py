"""边界②: 反思循环的窄接口。

这是 Day6 唯一"新写的设计"。内部的 8 个 state 字段全部私有。
调用方(supervisor 的 writer 节点)只看见 5 个字段。

跑法(自测, 不连网):
    cd W7 && python reflection_api.py
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ReflectionResult:
    draft: str                  # 终稿(自动取历史最优稿)
    score: int                  # 该稿评分
    rounds: int                 # 实际修改次数
    passed: bool                # 是否真达标(而非撞次数上限)
    notes: list = field(default_factory=list)   # 未解决的硬伤; 空 = 干净


def build_reflection_graph():
    """★ 生产实现: 反射子图在【这里】造, 调用方只拿到一个 graph 对象。

    两种跑法都要能 import 到(直接跑脚本 / 从仓库根当包跑), 所以给两条路。
    注意: 这一步会 import reflection_graph, 该模块在 import 期就建 LLM 客户端,
          所以别在模块顶层调它 —— 放在装配期调(见 writer_pipeline.main)。
    """
    try:
        from reflection_graph import build_graph          # cd W7 && python xxx.py
    except ImportError:
        from W7.reflection_graph import build_graph       # 仓库根 python -m W7.xxx
    return build_graph()


def write_with_reflection(topic: str, *, pass_score: int = 8,
                          max_revise: int = 3, graph=None) -> ReflectionResult:
    """输入主题, 返回带自省闭环的终稿。内部实现对调用方不透明。"""
    if graph is None:
        # 没注入就自己造一个 —— "不知道内部实现"的约束仍然只落在这一个函数里
        graph = build_reflection_graph()

    out = graph.invoke({
        "topic": topic, "draft": "", "score": 0, "issues": [],
        "revise_count": 0, "trajectory": [], "best_draft": "", "best_score": 0,
    })

    def stop_reason():
        if out.get("fatal_issues"):
            return "fatal"        # ★ 硬伤一票否决: 与分数无关, 有硬伤就是不达标
        if out["score"] >= pass_score:
            return "passed"
        if out["revise_count"] >= max_revise:
            return "max_revise"
        return "stalled"

    # ★ 关键: 不是"抄字段", 是【判定】。passed 由程序算, 不信任模型自评
    passed = stop_reason() == "passed"
    notes = [] if passed else [f"未达标下线(原因: {stop_reason()})"]
    notes.extend(f"硬伤: {x}" for x in (out.get("fatal_issues") or []))
    # ★ 返回的稿子是 best_draft, 那分数就必须配 best_score —— 不能配"最后一稿的分数"
    best = out.get("best_draft") or out.get("draft", "")
    best_score = out.get("best_score")
    return ReflectionResult(
        draft=best,
        score=int(best_score if best_score else out.get("score", 0)),
        rounds=int(out.get("revise_count", 0)),
        passed=passed,
        notes=notes,
    )


class FakeGraph:
    """测试替身: 不发一个 token 也能验接口"""
    def __init__(self, out): self.out = out
    def invoke(self, _): return self.out


if __name__ == "__main__":
    # 场景1: 达标
    r = write_with_reflection("x", graph=FakeGraph(
        {"score": 8, "best_score": 8, "revise_count": 1,
         "best_draft": "终稿", "draft": "初稿"}))
    assert r.passed and r.draft == "终稿" and r.notes == []

    # 场景2: 撞上限, 未达标  ← ★ 这条最重要
    r = write_with_reflection("x", graph=FakeGraph(
        {"score": 7, "best_score": 7, "revise_count": 3,
         "best_draft": "终稿", "draft": "初稿"}))
    assert not r.passed and r.notes, "未达标却没留 notes → supervisor 会以为成功"

    # 场景3: 分数配稿子 —— 返回的是 best_draft, 分数就必须配 best_score
    r = write_with_reflection("x", graph=FakeGraph(
        {"score": 8, "best_score": 9, "revise_count": 2,
         "best_draft": "好稿", "draft": "最后一稿"}) )
    assert r.score == 9 and r.draft == "好稿" and r.passed

    # 场景4: 硬伤一票否决 —— 9 分也不能算达标, 且 notes 里必须看得见
    r = write_with_reflection("x", graph=FakeGraph(
        {"score": 9, "best_score": 9, "revise_count": 1, "fatal_issues": ["编造数据"],
         "best_draft": "终稿", "draft": "初稿"}))
    assert not r.passed, "有硬伤还判达标 → supervisor 会带着硬伤收工"
    assert any("编造数据" in n for n in r.notes)
    print("reflection_api OK")
