"""sample_rounds.py —— 同一输入跑 N 次, 统计 rounds 分布, 用来定 MAX_ROUNDS
   把"8 是拍的"换成"8 是有依据的"(依据 = 最坏值 + 2 轮缓冲)"""
import os
import re
import statistics
import subprocess
import sys

N = 5
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "supervisor_demo.py")

rounds_list, ok, fail = [], 0, 0

for i in range(N):
    r = subprocess.run([sys.executable, TARGET],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    m = re.search(r"rounds=(\d+)", r.stdout)
    got = int(m.group(1)) if m else None
    if r.returncode == 0 and got:
        ok += 1
        rounds_list.append(got)
    else:
        fail += 1
    print(f"  第 {i+1} 次: rounds={got}  exit={r.returncode}")

print(f"\n成功 {ok}/{N}   fail {fail}/{N}")
if rounds_list:
    print(f"rounds: min={min(rounds_list)} max={max(rounds_list)} "
          f"mean={statistics.mean(rounds_list):.1f}  样本={sorted(rounds_list)}")
    print(f"\n建议 MAX_ROUNDS = {max(rounds_list) + 2}  (最坏 {max(rounds_list)} + 2 轮缓冲)")
else:
    print("全部失败 → 先看失败日志, 别拿失败样本算统计")