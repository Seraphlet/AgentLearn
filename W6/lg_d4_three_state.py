"""W6-D4 整合版: 审批三态 A/B/C 一次跑完
    A = 放行(执行原参数)   B = 拒绝(工具不执行)   C = 改参(update_state 改写 pending 参数 + 审计留痕)
"""

import os
import io
import contextlib
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


# ═══════════════════ 1. 工具 + 状态 ═══════════════════
@tool
def delete_file(path: str) -> str:
    """【危险】删除本地文件(演示版, 不真删)"""
    # ★ 这行 print 是"验真身"用的: 工具到底跑没跑、跑的是哪个路径, 全看它。
    #   断言也靠捕获这行来判定 —— 不要删。
    print(f">>> 工具真身执行: 删除 {path}")
    return f"文件已删除: {path}"


class S(TypedDict):
    messages: Annotated[list, add_messages]   # 有 reducer: 只能追加/按 id 替换
    audit_note: str        # ★ 无 reducer 的普通字段, 用来承载"人工审批留痕"


# ════════ 2. 两个 LLM: 决策的绑工具, 收口的不绑 ════════
_cfg = dict(model="deepseek-v4-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0)

llm_decide = ChatOpenAI(**_cfg).bind_tools([delete_file])

# ★ 关键设计: 收口的 LLM 【不绑工具】→ 物理上产不出 tool_calls → 图必然进 END。
#   这不是"劝模型别重试", 是让重试这条路在代码里不存在(赌恶不赌善)。
llm_summarize = ChatOpenAI(**_cfg)


# ═══════════════════ 3. 五个节点 ═══════════════════
def decide(state: S) -> dict:
    """只干一件事: 让模型决策。★必须 return, AIMessage 才有机会落进 state"""
    ai_msg = llm_decide.invoke(state["messages"])
    print(f"[decide] tool_calls={len(ai_msg.tool_calls)}")
    return {"messages": [ai_msg]}
    # ⚠️ 你上一版把 llm.invoke 和 interrupt 塞在同一个节点里 → interrupt 走异常通道,
    #    ai_msg 没机会 return → state 最后一条还是 HumanMessage → 改参时 AttributeError。
    #    拆成 decide / approve 两个节点就是为了修这个。


def approve(state: S) -> dict:
    """只干一件事: 问人。★这里【不能】放有副作用的代码"""
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return {}                              # 没有 tool_calls, 没什么可审的

    tc = last.tool_calls[0]
    print(f"[approve] 从 state 读到的待审参数: {tc['args']}")

    # ★ interrupt 之前的代码在 resume 时会【重放】, 所以本函数会打印两次。
    #   结论: interrupt 之前只能放幂等逻辑(读 state / print)。
    #   把"发邮件""扣款"放这里 = resume 时重复执行。
    answer = interrupt({
        "question": f"模型想删除 {tc['args']['path']}, 批准吗?",
        "options": ["approve", "reject", "edit"],
    })

    if answer == "reject":
        # ★ 返回不带 tool_calls 的 AIMessage → 路由判定为 END → 工具一次都不执行
        return {"messages": [AIMessage(content="已取消删除，本次未执行任何操作。")]}

    # approve / edit 都放行: state 里那条 AIMessage 原样流向 tools。
    # edit 时它的参数已被外部 update_state 改写过了。
    return {}


def audit(state: S) -> dict:
    """★治叙事★ 把"人改过参数"这件事, 写成模型看得见的一条消息"""
    note = state.get("audit_note", "")
    if not note:
        return {}
    print(f"[audit] 注入留痕: {note}")
    return {
        "messages": [HumanMessage(content=(
            f"【审批人操作记录】{note}。"
            "这是审批人主动、有意做出的决定，已获批准执行；"
            "这不是错误，无需道歉、无需重试其他路径，请直接据实向用户总结结果。"
        ))],
        "audit_note": "",          # 用完清空, 避免污染后续轮次
    }
    # ⚠️ 已知取舍: 这里用 HumanMessage "冒充"人在说话, 在 API 眼里和真人发言无法区分。
    #    单机 demo 可以; 多用户产品要换成可区分的通道(自定义消息类型/独立字段)。


def summarize(state: S) -> dict:
    """★治重试★ 不绑工具的 LLM 收口 → 只能说话, 不可能再调工具"""
    ai_msg = llm_summarize.invoke(state["messages"])
    print("[summarize] 收口完成, tool_calls=0 (结构保证, 非提示词保证)")
    return {"messages": [ai_msg]}


# ═══════════════════ 4. 路由 + 构图 ═══════════════════
def route_after_decide(state: S):
    last = state["messages"][-1]
    return "approve" if (isinstance(last, AIMessage) and last.tool_calls) else END


def route_after_approve(state: S):
    last = state["messages"][-1]
    return "tools" if (isinstance(last, AIMessage) and last.tool_calls) else END


builder = StateGraph(S)
builder.add_node("decide", decide)
builder.add_node("approve", approve)
builder.add_node("tools", ToolNode([delete_file]))
builder.add_node("audit", audit)
builder.add_node("summarize", summarize)

builder.add_edge(START, "decide")
builder.add_conditional_edges("decide",  route_after_decide,  {"approve": "approve", END: END})
builder.add_conditional_edges("approve", route_after_approve, {"tools": "tools",     END: END})
builder.add_edge("tools", "audit")        # 工具跑完 → 先留痕(治叙事)
builder.add_edge("audit", "summarize")    # 再收口(不绑工具) → 必然 END
builder.add_edge("summarize", END)

# 一张图 + 一个 checkpointer; 会话隔离靠 invoke 时的 thread_id
graph = builder.compile(checkpointer=MemorySaver())

# 拓扑(先看地图再看代码):
#   START → decide ─(有tool_calls)→ approve ─(放行)→ tools → audit → summarize → END
#                  └(无)──────────→ END      └(拒绝)→ END


# ═══════════════ 5. 三个场景 ═══════════════
def run_a():
    """A 放行: 执行模型原本给的参数 /tmp/a.txt"""
    print("\n" + "=" * 18 + " A 放行 " + "=" * 18)
    cfg = {"configurable": {"thread_id": "state-a"}}
    graph.invoke({"messages": [("user", "帮我删除 /tmp/a.txt")]}, cfg)  # 停在 approve 的 interrupt
    out = graph.invoke(Command(resume="approve"), cfg)                  # 放行 → tools 执行
    print("A 最终:", out["messages"][-1].content)
    return out, graph.get_state(cfg)


def run_b():
    """B 拒绝: 取消, 工具一次都不执行"""
    print("\n" + "=" * 18 + " B 拒绝 " + "=" * 18)
    cfg = {"configurable": {"thread_id": "state-b"}}
    graph.invoke({"messages": [("user", "帮我删除 /tmp/b.txt")]}, cfg)
    out = graph.invoke(Command(resume="reject"), cfg)                   # 拒绝 → END
    print("B 最终:", out["messages"][-1].content)
    return out, graph.get_state(cfg)


def run_c():
    """C 改参: 人在图外把 pending 参数从 a.txt 改成 old.txt, 再放行"""
    print("\n" + "=" * 18 + " C 改参 " + "=" * 18)
    cfg = {"configurable": {"thread_id": "state-c"}}
    graph.invoke({"messages": [("user", "帮我删除 /tmp/a.txt")]}, cfg)  # 停在 approve

    # ── ① 读档, 确认最后一条是 AIMessage(这是能改参的前提) ──
    st = graph.get_state(cfg)
    last = st.values["messages"][-1]
    print("最后一条类型:", type(last).__name__)          # 期望 AIMessage
    assert isinstance(last, AIMessage) and last.tool_calls, \
        "最后一条不是带 tool_calls 的 AIMessage → 改参无从下手"

    # ── ② 复制一份再改(不能直接改原对象: checkpoint 是不可变快照) ──
    new_tc = [{**tc, "args": {"path": "/tmp/old.txt"}} for tc in last.tool_calls]

    # ★ id=last.id 是生死线: add_messages 按 id 做 upsert
    #   带 id → 原地替换; 不带 → 自动生成新 UUID → 变成"追加", 会有两条 AI 消息
    fixed = AIMessage(id=last.id, content=last.content, tool_calls=new_tc)

    # ── ③ 改参 + 留痕 在同一次 update_state 里写下去(避免两次写入不同步) ──
    graph.update_state(cfg, {
        "messages": [fixed],
        "audit_note": "审批人把删除目标从 /tmp/a.txt 改成了 /tmp/old.txt",
    })
    print("已把删除目标改为 /tmp/old.txt")

    # ★ 顺序铁律: 先 update_state 改好, 再 Command(resume) 恢复。反了改参会被覆盖。
    out = graph.invoke(Command(resume="edit"), cfg)
    print("C 最终:", out["messages"][-1].content)
    return out, graph.get_state(cfg)


# ═══════════════ 6. 断言核验(把"我看过"变成"可重复验证") ═══════════════
def check(txt, out_a, st_a, out_b, st_b, out_c, st_c):
    print("\n" + "=" * 16 + " 断言核验 " + "=" * 16)

    # ── A: 放行 → 原参数被执行, 且图真收敛 ──
    assert ">>> 工具真身执行: 删除 /tmp/a.txt" in txt, "A: 原参数 a.txt 没被执行"
    assert st_a.next == (), f"A: 图没收敛, 还停在 {st_a.next}"

    # ── B: 拒绝 → 走拒绝分支, 且工具一次都没执行 ──
    assert "已取消" in out_b["messages"][-1].content, "B: 没走拒绝分支"
    assert ">>> 工具真身执行: 删除 /tmp/b.txt" not in txt, "B: 工具竟然执行了"
    assert st_b.next == (), f"B: 图没收敛, 还停在 {st_b.next}"

    # ── C: 改参 → 执行的是 old.txt, 且审计留痕生效 ──
    assert ">>> 工具真身执行: 删除 /tmp/old.txt" in txt, "C: 改后的 old.txt 没被执行"
    assert "[audit] 注入留痕" in txt,        "C: 审计留痕没注入 → 模型可能又编'我误删了'"
    assert "[summarize] 收口完成" in txt,    "C: 收口节点没跑到"
    assert st_c.next == (), f"C: 图没收敛, 还停在 {st_c.next}"
    assert getattr(st_c.values["messages"][-1], "tool_calls", []) == [], \
        "C: 最后一条不该带 tool_calls"

    # ★ "重试被掐死"的判据: A/B/C 各进一次 decide, 共 3 次。若变 4 次 = 又空转重试了
    n_decide = txt.count("[decide]")
    assert n_decide == 3, f"decide 应跑 3 次(每场景 1 次), 实际 {n_decide} 次 → 有重试"

    # ── 软检查: 模型的话术会飘, 不该让脚本崩, 但要提醒你回看 ──
    if "审批" not in out_c["messages"][-1].content:
        print("⚠️ 软检查未过: C 的总结没提到'审批', 叙事可能又飘了 —— 回看上面 C 最终")

    print("✅ D4 整合版 8 项断言全绿(A 2 项 / B 3 项 / C 4 项)")


def main():
    # 把三个场景的终端输出全部捕获下来 → 断言靠它判定, 不靠眼睛数
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        out_a, st_a = run_a()
        out_b, st_b = run_b()
        out_c, st_c = run_c()

    txt = log.getvalue()
    print(txt)                    # 原始日志照打, 方便你回看模型说了什么
    check(txt, out_a, st_a, out_b, st_b, out_c, st_c)


if __name__ == "__main__":
    main()