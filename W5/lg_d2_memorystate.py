from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class MemoryState(TypedDict):
    messages: list[dict]
    compress_summary: str
    long_term_hit: str
    decision: str


def decide_path(state: MemoryState):
    """路由函数：读 state，返回节点名（省略 path_map 因为返回值=节点名）"""
    if len(state["messages"]) > 20:
        return "compress"  # ① 消息太多 → 压缩
    recent = [m.get("content", {}) for m in state["messages"]]
    if any("记住" in c for c in recent):
        return "long"  # ② 最近有"记住" → 存长期
    return "short"  # ③ 默认 → 短期窗口


def short_node(state: MemoryState) -> dict:
    return {"decision": "short"}  # 只写自己那格，别的字段保留


def long_node(state: MemoryState) -> dict:
    return {"decision": "long", "long_term_hit": "[模拟] 从向量库召回用户画像"}


def compress_node(state: MemoryState) -> dict:
    return {"decision": "compress", "compress_summary": "[模拟] 老消息已摘要"}


b = StateGraph(MemoryState)
for name, fn in [("short", short_node), ("long", long_node), ("compress", compress_node)]:
    b.add_node(name, fn)

b.add_conditional_edges(START, decide_path)
for name in ["short", "long", "compress"]:
    b.add_edge(name, END)

g = b.compile()

# 三条测试路（messages 统一用 dict，别混 str）

tests = {
    "走短期": {"messages": [{"role": "user", "content": "今天天气不错"}],
               "compressed_summary": "", "long_term_hit": "", "decision": ""},
    "走长期": {"messages": [{"role": "user", "content": "记住我住北京"},
                            {"role": "assistant", "content": "好的"}],
               "compressed_summary": "", "long_term_hit": "", "decision": ""},
    "走压缩": {"messages": [{"role": "user", "content": f"普通消息{i}"} for i in range(21)],
               "compressed_summary": "", "long_term_hit": "", "decision": ""},
}
for label, data in tests.items():
    print(f"\n=== {label} ===")
    for s in g.stream(data, stream_mode="updates"):   # 看实际触发了哪个节点
        print("  ", s)
    out = g.invoke(data)
    extra = out.get("long_term_hit") or out.get("compressed_summary") or ""
    print(f"  decision={out.get('decision')}  {extra}")
for label, data in tests.items():
    out = g.invoke(data)
    print(f"\n数据结构：{label} → out = {out}\n")
    extra = out.get("long_term_hit") or out.get("compressed_summary") or ""
    print(f"{label} → decision={out['decision']}  {extra}")
