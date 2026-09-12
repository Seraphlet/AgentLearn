"""W6-D4 实验2: 时间旅行 —— 纯回放 + 改档分叉
   ★ 不用 LLM: 确定性图, 才能干净地验证"哪几个节点被重跑了"
"""

import io 
import contextlib
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class T(TypedDict):
    messages: Annotated[list,add_messages]
    way: str #普通字段(无 reudcer)：用来延时"改档分岔”


def step_a(state:T)->dict:
    print("→ A 执行")
    return {"messages":[("ai","A完成")]}

def step_b(state: T) ->dict:
     # 读 state 里的 way: 默认方式 / 改档后的另一种方式
     way=state.get("way") or "默认方式"
     print(f"→ B 执行 ({way})")
     return {"messages":[("ai",f"B完成({way})")]}

def step_c(state: T) ->dict:
    print("→ C 执行")
    return {"messages":[("ai","C完成")]}

builder=StateGraph(T)
builder.add_node("a",step_a)
builder.add_node("b",step_b)
builder.add_node("c",step_c)
builder.add_edge(START,"a")
builder.add_edge("a","b")
builder.add_edge("b","c")
builder.add_edge("c",END)
graph=builder.compile(checkpointer=MemorySaver())

cfg={"configurable":{"thread_id":"tt-1"}}

# ═══ 第 0 步: 完整跑一遍, 制造出"历史" ═══

log=io.StringIO()
with contextlib.redirect_stdout(log):
    r0=graph.invoke({"messages":[("user","完整跑一遍A B C ")]},cfg)
print("原始跑完",log.getvalue().replace("\n","|").strip())
print("原始消息数:", len(r0["messages"]))        # 预期 4 = Human + A + B + C

# ═══ 第 1 步: 看存档时间线(新→旧) ═══

snaps=list(graph.get_state_history(cfg))
print(f"\n=== 存档时间线({len(snaps)} 个, 新→旧) ===")
for i,s in enumerate(snaps):
    cid=s.config["configurable"]["checkpoint_id"]
    print(f"#{i} cid={cid[:8]}  next={s.next}  消息数={len(s.values['messages'])}")
# ↓ 先看这张表: next 非空的档才有"活没干完", 才有东西可回放

# ═══ 第 2 步: 挑"a 跑完、b 没跑"的存档点 ═══

target=next(s for s in snaps if s.next==("b",))
tcid=target.config["configurable"]["checkpoint_id"]
print(f"\n目标存档: cid={tcid[:8]}  next={target.next}  "
      f"消息数={len(target.values['messages'])}")   # 预期 2

# ═══ 第 3 步: 纯回放 —— 直接复用 snap.config 当读档钥匙 ═══

log2=io.StringIO()
with contextlib.redirect_stdout(log2):
    r_replay=graph.invoke(None,target.config)  # ★ None = 不传新输入
txt_replay=log2.getvalue()
print("回放日志:", txt_replay.replace("\n", " | ").strip())
print("回放后消息数:", len(r_replay["messages"]))    # 预期 4

# ═══ 第 4 步: 改档分叉 —— 改 way 再回放, 未来的 B 就不一样 ═══
new_cfg=graph.update_state(target.config,{"way":"另一种方式"})
print(f"\n分叉配置 cid={new_cfg['configurable']['checkpoint_id'][:8]}")
log3 = io.StringIO()
with contextlib.redirect_stdout(log3):
    r_fork = graph.invoke(None, new_cfg)
txt_fork = log3.getvalue()
print("分叉日志:", txt_fork.replace("\n", " | ").strip())
print("分叉后消息数:", len(r_fork["messages"]))

print("\n=== 分叉后每条消息(真身) ===")
for i, m in enumerate(r_fork["messages"]):
    print(f"[{i}] {type(m).__name__}: {m.content!r}")

# 定位 B 那条, 把括号的码点打出来 → 一眼看出全角还是半角
b_txt = next(str(m.content) for m in r_fork["messages"] if "B完成" in str(m.content))
print("\nB 的真身:", repr(b_txt))
print("括号码点:", [hex(ord(c)) for c in b_txt if c in "()（）"])
# 半角 '(' = 0x28   ')' = 0x29
# 全角 '（' = 0xff08 '）' = 0xff09



# ═══ 第 5 步: 断言核验(不靠眼睛!) ═══
print("\n" + "=" * 16 + " 断言核验 " + "=" * 16)

# ── 纯回放: 只重跑 B/C, 没重跑 A ──
assert "→ B 执行 (默认方式)" in txt_replay, "B 没重跑"
assert "→ C 执行" in txt_replay,            "C 没重跑"
assert "→ A 执行" not in txt_replay,        "★ A 竟然重跑了 → 这不是回放, 是从头跑"
assert txt_replay.index("→ B") < txt_replay.index("→ C"), "顺序乱了"
assert len(r_replay["messages"]) == len(target.values["messages"]) + 2, \
    f"应只新增 2 条, 实际 {len(r_replay['messages']) - len(target.values['messages'])}"

# ── 改档分叉: 未来跟着改了 ──
assert "→ B 执行 (另一种方式)" in txt_fork, "改档没生效"
assert "→ A 执行" not in txt_fork,           "A 不该重跑"
fork_text = " | ".join(str(m.content) for m in r_fork["messages"])
assert "B完成(另一种方式)" in fork_text, "B 那条没带上改后的值"   # 精确指认 

# ── 原分支还在(存档只追加, 不覆盖) ──
snaps_after = list(graph.get_state_history(cfg))
cids_after = [s.config["configurable"]["checkpoint_id"] for s in snaps_after]
assert tcid in cids_after, "原存档点不见了 → 存档被覆盖了(不该发生)"
assert len(snaps_after) > len(snaps), "分叉应新增存档, 而非原地覆盖"

print("✅ 时间旅行 8 项断言全绿(回放 5 / 分叉 3)")
print(f"   ★ 关键: 两次回放的日志里都没有 '→ A 执行' → A 没重跑, 这才是读档")