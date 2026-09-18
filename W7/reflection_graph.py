"""W7-Day3: 反思循环 Critique-Revise
模型会无限制的修改，需要加上Maxtoken
generator → critic(结构化评审) → reviser → 回到 critic
双重停止条件: 分数达标 或 达到 MAX_REVISE 或 停滞

跑法(Windows CMD):
    python w7\\reflection_graph.py            # 正常
    python w7\\reflection_graph.py forced     # PASS_SCORE=11, 强制走满 MAX_REVISE
    set STALL=0 && python w7\\reflection_graph.py   # 关掉停滞检测
"""
import os
import sys
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState


# ══════════ 0. 配置 ══════════
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
FORCED = (MODE == "forced")
PASS_SCORE = 11 if MODE == "forced" else 8   # ★ 8 是拍的(待校准); forced 用 11 保证永不达标
MAX_REVISE = 3                               # ★ 3 也是拍的(成本上限)
STALL_STOP = (not FORCED) and (os.getenv("STALL", "1") != "0") # 停滞检测开关
print(f"=== MODE={MODE} PASS_SCORE={PASS_SCORE} MAX_REVISE={MAX_REVISE} "
      f"STALL_STOP={STALL_STOP}"
      + ("  ← forced: 停滞出口已自动关闭, 本次只测『次数用尽』" if FORCED else ""))
llm = ChatOpenAI(model=MODEL_NAME, base_url=BASE_URL,
                 api_key=API_KEY, temperature=0)

# ══════════ 1. 结构化评审 + state ══════════
class Critique(BaseModel):
    """② 强制 critic 输出可判断的结果, 而不是一段自由文本"""
    score: int = Field(ge=1,le=10,description="1-10分，8分及以上达标")
    issues: list[str] = Field(default_factory=list,description="软问题，影响分数")
    # ★ 硬伤: 与分数无关, 一律拦下
    fatal_issues: list[str] = Field(default_factory=list,
        description="硬伤清单: 编造/无来源的具体数据、事实错误、违规内容。"
                    "只要非空, 无论多少分都不达标")
    passed: bool =Field(description="是否达标")

class RState(MessagesState):
    topic: str
    draft: str
    score: int
    issues: list[str]
    revise_count: int
    # ★ 每轮评审轨迹(operator.add = 追加, 不是覆盖)
    trajectory: Annotated[list, operator.add]
    # ★ 保留历史最优稿(改坏了不至于越改越烂)
    best_draft: str
    best_score: int

    # ══════════ 2. 三个节点 ══════════
def generator(state: RState) ->dict:
    """初稿"""
    r = llm.invoke(f"请写一篇约200字的中文短文，主题：{state['topic']}。只输出正文。")
    draft =str(r.content)
    print(f"[generator]初稿 {len(draft)}字")
    return {"draft": draft,"revise_count":0,"best_draft":draft,"best_score":0}


# ★ json_mode: 唯一在你环境可用的方式; prompt 里必须有 "JSON"
JUDGE_SYS = """你是严格但公正的中文写作评审。
评分维度: ① 观点清晰度 ② 论据支撑 ③ 结构层次 ④ 表达流畅
评分标准: 8 分及以上为达标(可以直接发布), 6-7 分为尚可, 5 分及以下为不合格。
issues 必须写「具体可改的问题」(如"第2段缺少例子"), 不要写"可以更好"这类空话。
issues 最多 3 条, 每条不超过 15 字。

输出格式: 只输出一个 JSON 对象, 形如
{"score": 7, "issues": ["第2段缺少具体例子", "结论过于仓促"], "passed": false}"""

def _parse_critique(raw) ->Critique:
    """手动校验: json_mode 不保证数值范围"""
    if not isinstance(raw,Critique):
        raw = Critique.model_validate(raw)
    if not (1<=int(raw.score)<=10):
        raise ValueError(f"score 越界:{raw.score}")
    return raw 

def critic(state:RState) ->dict:
    """评审:打分+列问题。同时更新轨迹与最优稿"""
    judge = llm.with_structured_output(Critique,method="json_mode")
    prompt = f"{JUDGE_SYS}\n\n文稿:\n{state['draft']}"

    c=None
    for attempt in range(3):
        try:
            c=_parse_critique(judge.invoke(prompt))
            break
        except Exception as e:
            print(f"  [warn] 第 {attempt+1} 次评审解析失败: {type(e).__name__}: {str(e)[:80]}")

    if c is None:
        # 解析失败时【不放过】, 而不是误判为达标
        c=Critique(score=1,issues=["评审解析失败，无法判断"],passed=False)
        print("  [warn] 三次都失败 → 保守判定为不达标(宁可多改一轮, 也不放过)")
    # ★ 单一事实来源: 达标与否【由程序用 score 判定】, 模型的 passed 只做交叉检查
    passed_by_code =c.score>=PASS_SCORE
    if passed_by_code !=c.passed:
        print(f"  [warn] 模型说 passed={c.passed}, 但 score={c.score} 对照阈值 "
            f"{PASS_SCORE} 应为 {passed_by_code} → 以程序的判定为准")
    score = int(c.score)
    best_score,best_draft = state.get("best_score",0),state.get("best_draft","")
    if score>best_score:
        best_score,best_draft = score,state["draft"] #刷新最优解
    entry = dict(round=len(state.get("trajectory",[])),score=score,issues=c.issues,draft_len=len(state["draft"]),draft_head=state["draft"][:60].replace("\n"," "))
    print(f"[critic] 第 {entry['round']} 轮: {score} 分 / 问题 {len(c.issues)} 条"
          f" / 程序判定 {'达标' if passed_by_code else '未达标'}")
    return {"score": score, "issues": c.issues, "trajectory": [entry],
            "best_score": best_score, "best_draft": best_draft,
            "messages": [("assistant", f"[评审] {score} 分 | 问题: {c.issues}")]}
def reviser(state: RState) -> dict:
    """按评审意见重写(只带原稿 + 意见, 不带全历史 → 省 token)"""
    r = llm.invoke(
        "请根据评审意见改进以下文稿, 保持主题不变、篇幅相近, 只输出改进后的正文。\n\n"
        f"原稿:\n{state['draft']}\n\n评审意见:\n{state['issues']}"
    )
    draft =str(r.content)
    n= state.get("revise_count",0)+1
    print(f"[reviser]第{n}次修改,新稿{len(draft)}字"f"(原 {len(state['draft'])}字)")
    return {"draft":draft,"revise_count":n}


# ══════════ 3. 条件边(循环 + 多重出口)⭐ ══════════
def route_after_critic(state: RState)->Literal["reviser","end"]:
    score=state["score"]
    count= state.get("revise_count",0)
    traj=state.get("trajectory",[])
    fatal =state.get("fatal_issues") or []

    if count>= MAX_REVISE:
        return "end"        # 出口1: 达标
    if fatal:
        return "reviser"     # ② 硬伤 > 分数
    if score >=PASS_SCORE:
        return "end"      
    if STALL_STOP and len(traj) >= 2 and traj[-1]["score"] < traj[-2]["score"]:
        return "end"        # 出口3: 改了没变好 → 别继续烧钱
    return "reviser"

def build_graph():
    b= StateGraph(RState)
    b.add_node("generator",generator)
    b.add_node("critic",critic)
    b.add_node("reviser",reviser)

    b.add_edge(START,"generator")
    b.add_edge("generator","critic")
    b.add_conditional_edges("critic",route_after_critic,{"reviser":"reviser","end":END})
    b.add_edge("reviser","critic")  # ★ 回边 = 反思循环的生命线

    return b.compile()

# ══════════ 4. 跑 + 断言 ══════════
INITIAL = {
    "topic": "为什么 Agent 需要记忆系统",
    "draft": "", "score": 0, "issues": [], "revise_count": 0,
    "trajectory": [], "best_draft": "", "best_score": 0,
}

def stop_reason(out):
    if out["score"]>=PASS_SCORE:
        return "达标"
    if out["revise_count"]>=MAX_REVISE:
        return "达到 MAX_REVISE 上限"
    return "停滞(改了没变好)"

def main():
    graph = build_graph() 
    print(f"=== 开始跑 (MODE={MODE}, PASS_SCORE={PASS_SCORE}, "
          f"MAX_REVISE={MAX_REVISE}, STALL_STOP={STALL_STOP}) ===\n")
    # 一次运行同时拿链和终态 —— 不做跨运行对比
    chain,final_state =[],None
    for mode,chunk in graph.stream(INITIAL,stream_mode=["updates","values"]):
        if mode =="updates":
            for node_name in chunk:
                chain.append(node_name)
        elif mode =="values":
            final_state =chunk
          #  print(final_state,end="\n")
    out = final_state
    traj =out["trajectory"]

    print(f"\n[节点链] {' → '.join(chain)}")

    # ── ④ 分数轨迹 ──
    print("\n=== ④ 分数轨迹 ===")
    for t in traj:
        print(f"  第 {t['round']} 轮: {t['score']} 分, {t['draft_len']} 字")

    # ── ④ v1 vs 终稿对比 ──
    v1 = traj[0]["draft_head"] if traj else ""
    print("\n=== ④ v1 vs 最优稿 ===")
    print(f"  v1 (前60字): {v1}")
    print(f"  最优 (前60字): {out['best_draft'][:60].replace(chr(10), ' ')}")
    print(f"  字数: v1={len(out['best_draft'])} 字" if len(traj) == 1 else "")

    print(f"\n[结果] 停止原因={stop_reason(out)}  最终评分={out['score']}  "
          f"最优评分={out['best_score']}  修改次数={out['revise_count']}")
    print(f"\n=== 终稿(采用最优稿) ===\n{out['best_draft']}")

if __name__=="__main__":
    main()