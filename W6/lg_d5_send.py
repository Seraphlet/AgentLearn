"""W6-D5 实验2: Send 动态扇出 (Map-Reduce) —— 修正版
   修正 ①: distributor 改成普通节点, 只返回 dict
   修正 ②: Send 移到条件边的路由函数里返回
"""
import time
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# ══════════════ state ══════════════
class MapState(TypedDict):
    topics: list[str]                             # 待办清单（输入）
    results: Annotated[list[str], add]            # 并行结果：靠 add reducer 汇聚
    summary: str                                  # 普通字段：后写覆盖先写


def do_work(topic: str) -> str:
    """抽成独立函数，是为了对照组能直接调它（串行跑一遍）"""
    time.sleep(1)                                 # 模拟 1 秒 IO（等网络/等模型）
    return f"{topic} 完成"


# ══════════════ 节点（★ 返回值只能是 dict） ══════════════
def distributor(state: MapState) -> dict:
    print(f"[distributor] 扇出 {len(state['topics'])} 份活")
    return {}                                     # ★ 修正1：返回 dict，不返回 Send


def worker(state: dict) -> dict:
    # ★ state 是 Send 塞进来的【私有字典 {"topic": ...}】，不是完整 MapState
    #   所以这里拿不到 messages / results —— 需要什么，Send 时就得塞什么
    topic = state["topic"]
    r = do_work(topic)
    print(f"[worker] {topic} 回来")
    return {"results": [r]}                       # 返回 list，配合 add reducer


def join(state: MapState) -> dict:
    print(f"[join] 收到 {len(state['results'])} 份")
    return {"summary": " | ".join(sorted(state["results"]))}
    # ★ 不要再写 return {"results": state["results"]}
    #   results 带 add reducer，再写一遍 = 把已有列表又加一次 → 4 条变 8 条


# ══════════════ 路由函数（★ 只有这里能返回 Send） ══════════════
def route_to_workers(state: MapState) -> list[Send]:
    # 它收到的 state 和普通节点看到的一样（父图完整 state），所以读得到 topics
    return [Send("worker", {"topic": t}) for t in state["topics"]]


# ══════════════ 构图 ══════════════
g = StateGraph(MapState)
g.add_node("distributor", distributor)
g.add_node("worker", worker)
g.add_node("join", join)

g.add_edge(START, "distributor")
g.add_conditional_edges("distributor", route_to_workers, ["worker"])   # ★ 修正2
# ↑ 注意：这里【没有】 g.add_edge("distributor", "worker") 了
g.add_edge("worker", "join")          # 所有 worker 都回来后才跑 join
g.add_edge("join", END)
graph = g.compile()


# ══════════════ 跑 + 验证 ══════════════
topics = ["RAG", "Agent记忆", "多Agent", "MCP"]

t0 = time.time()
out = graph.invoke({"topics": topics})        # 实验组：走图（并行）
par = time.time() - t0

t0 = time.time()
for t in topics:
    do_work(t)                                # 对照组：同一批活串行做
ser = time.time() - t0

speedup = ser / par
print(f"\n[算账] 串行 {ser:.2f}s | 并行 {par:.2f}s | 加速比 {speedup:.2f}x")
print("[结果]", out["results"])
print("[汇总]", out["summary"])

# ── 断言 ──
assert len(out["results"]) == 4, f"结果条数不对(翻倍了?): {out['results']}"
assert sorted(out["results"]) == sorted(f"{t} 完成" for t in topics), out["results"]
assert speedup > 2.0, f"加速比只有 {speedup:.2f}x, 不像并行"
assert all(t in out["summary"] for t in topics), "join 没拿到全部结果"

print("✅ 实验2 四项断言全绿")