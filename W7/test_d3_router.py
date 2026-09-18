"""test_router.py —— 离线测条件边的三条出口。不调 LLM, 0 成本, 秒出结果"""
import os, sys

os.environ.setdefault("STALL", "1")
sys.argv = ["test_router.py", "normal"]      # 让模块以 normal 模式导入(不会跑 main)
import reflection_graph as rg                 # ← 这能安全 import, 全靠那个 __main__ 守门

def mk(score, count, scores):
    return {"score": score, "revise_count": count,
            "trajectory": [{"score": s} for s in scores]}

stall = rg.STALL_STOP
cases = [
    ("出口1 达标",            mk(8, 1, [7, 8]),   "end"),
    ("出口1 边界(恰好=阈值)", mk(8, 0, [8]),      "end"),
    ("出口2 次数用尽",        mk(7, 3, [7, 7, 7]), "end"),
    ("出口3 变差",            mk(7, 1, [8, 7]),   "end" if stall else "reviser"),
    ("★ 平局(验证 < 的那行)", mk(7, 1, [7, 7]),   "reviser"),   # 用 <= 会错成 end
    ("正常进步",              mk(7, 1, [6, 7]),   "reviser"),
    ("单轮不该触发停滞",      mk(7, 0, [7]),      "reviser"),
]

fail = 0
print(f"[STALL_STOP={stall}]")
for name, st, expect in cases:
    got = rg.route_after_critic(st)
    ok = got == expect
    print(f"{'✅' if ok else '❌'} {name}: 期望 {expect}, 实际 {got}  {'← 这行就是证据' if not ok else ''}")
    fail += (not ok)

print(f"\n{'全部通过' if not fail else f'{fail} 项失败'}")