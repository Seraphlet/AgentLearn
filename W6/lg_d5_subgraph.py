"""W6-D5 实验1: 子图当零件 —— 一张编译好的图, 当普通节点塞进父图"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ══════════ 零件先单独造好、单独测 ══════════

class SubState(TypedDict):
    text: str
    result: str


def shout(state: SubState)->dict:
    return {"result": state["text"].upper()+"!"}

sub_b=StateGraph(SubState)
sub_b.add_node("shout",shout)
sub_b.add_edge(START,"shout")
sub_b.add_edge("shout",END)
subgraph=sub_b.compile()  # ★ 必须先 compile 才能当节点 (清单避坑表第1条)

# ══════════ 父图: 把零件装上去 ══════════

class ParentState(TypedDict):
    text: str
    result: str     # ★ 父图必须有这个"格子" —— 子图写的 result 才有地方落
    final: str

def entry(state: ParentState)->dict:
    return {"text":state["text"].strip()} # 父图先加工一下, 再交给子图

def parent_end(state: ParentState)->dict:
    return {"final":"父图收到："+state["result"]}

p=StateGraph(ParentState)
p.add_node("entry",entry)
p.add_node("inner",subgraph)
p.add_node("parent_end",parent_end)
p.add_edge(START,"entry")
p.add_edge("entry","inner")
p.add_edge("inner","parent_end")
p.add_edge("parent_end",END)
parent=p.compile()

# ══════════ 验证三件事 ══════════
# ① 零件能单独用(= 可复用 / 可单测)
solo = subgraph.invoke({"text": "hello"})
print("① 子图单独跑:", solo)
assert solo["result"] == "HELLO!", f"子图单独跑就错了: {solo}"

# ② 装进父图, 整条链路通
out = parent.invoke({"text": "  hello langgraph  "})
print("② 父图结果:", out["final"])
assert out["final"] == "父图收到：HELLO LANGGRAPH!", f"父图链路断了: {out}"

# ③ 命名空间: stream 打印出"这是谁家的节点"
raw = list(parent.stream({"text": "hi"}, stream_mode="updates", subgraphs=True))

# ❌ 旧: 循环论证 —— "inner" 是我自己起的父图节点名, 撞车也能绿
# assert any("inner" in str(ev) for ev in raw), "没看到 inner 前缀 → 没证明出层级"

# ✅ 新: 两层都查, 结构化取字段, 不做整句子串比对
nested = [(ns, upd) for ns, upd in raw if ns]        # 命名空间非空 = 真进到子图里
assert nested, "没有任何带非空命名空间的事件 → 子图只是被当普通节点了"

ns, upd = nested[0]
assert ns[0].startswith("inner:"), f"命名空间前缀不对: {ns!r}"           # 父节点名 + UUID
assert "shout" in upd, f"子图内部节点没暴露, 实际 update 键={list(upd)!r}"  # ★ 关键证据

# 反向判据: 子图内部的名字不该出现在父图层级
top_level = [upd for ns, upd in raw if not ns]
assert not any("shout" in upd for upd in top_level), "层级归属错了: shout 泄漏到父图层级"

# 另一条独立证据: 子图自己的图里确实登记了这个节点
assert "shout" in subgraph.get_graph().nodes, "子图里没有 shout 节点"
print("✅ 实验1 ③ 三项断言全绿(结构化, 非子串)")
 


