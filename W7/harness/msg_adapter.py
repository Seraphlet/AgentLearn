"""msg_adapter —— 消息格式适配层(Day6 集成层)

为什么需要它(实测出来的, 不是设计洁癖):
  Day2/Day3 用的是 LangChain 的【message 对象】(有 .content/.type 属性, 没有 .get)
  Day5 的 SessionStore / ContextCompactor 读的是【dict】(m.get("role"))
  直接对接 -> AttributeError: 'HumanMessage' object has no attribute 'get'

还有第二个更阴的坑: 角色别名不一样
  LangChain: "human" / "ai"
  我们内部:   "user" / "assistant"
  即使 .get 能过, 角色也会全变成 "?" -> 成对性检查静默失效(不报错!)

设计原则: 边界显式转换, 内部只用一种表示(dict)。
"""
from __future__ import annotations

#   ★ 角色归一化: 按"词干前缀"匹配, 不用穷举别名表。
#   教训: 第一版我手写 {"humanmessage": "user", "aichatmessage": "assistant", ...},
#         结果漏了 "aimessage" -> AIMessage 被映射成 'aimessage' 而不是 'assistant'。
#         穷举别名表必然漏 —— 改成前缀匹配, 这一类漏就结构性消失了。

ROLE_STEMS =[
    ("human","user"),
    ("system","system"),
    ("function","tool"),
    ("assistant","assistant"),
    ("user","user"),
    ("ai","assistant"),  #放最后： "aichatemessage"/"aimessage"都吃
]

ROLE_ALIASES = {                   # 精确别名(优先于前缀匹配)
    "human": "user", "user": "user",
    "ai": "assistant", "assistant": "assistant",
    "system": "system",
    "tool": "tool", "function": "tool",
}

TO_LC_TYPE = {"user": "human", "assistant": "ai", "system": "system", "tool": "tool"}

def _stem_match(s:str):
    for stem,role in ROLE_STEMS:
        if s.startswith(stem):
            return role
    return None

def _attr(obj,name,default=None):
    """先试 dict访问，再试属性访问。两条路都走不到才用默认值"""
    if isinstance(obj,dict):
        return obj.get(name,default)
    return getattr(obj,name,default)

def norm_role(raw) ->str:
    """把任何写法(role / type / 类名)统一成 user/assistant/system/tool"""
    if raw is None:
        return "?"
    s=str(raw)
    if "." in s or "<" in s:
        s=s.rsplit(".",1)[-1].strip("'>")   # ★ 是 str 的方法, 不是 numpy.char.rsplit 的函数
    s = s.lower()
    if s in ROLE_ALIASES:
        return ROLE_ALIASES[s]
    hit = _stem_match(s)
    if hit:
        return hit
    return s or "?"


def to_dict(msg) -> dict:
    """LangChain message 对象 / dict -> 内部 dict。幂等: dict 进来还是 dict。"""
    if isinstance(msg,str):
        return {"role":"user","content":msg}

    raw_role =_attr(msg,"role")
    if raw_role is None:
        raw_role =_attr(msg,"type")
    if raw_role is None:
        raw_role = type(msg).__name__

    d = {"role": norm_role(raw_role), "content": _attr(msg, "content", "") or ""}

    name= _attr(msg,"name")
    if name:
        d["name"]=name

    tcs = _attr(msg,"tool_calls")   # LangChain: 对象列表

    if tcs:
        d["tool_calls"] = [
            {"id": _attr(tc, "id"), "name": _attr(tc, "name"),
             "args": _attr(tc, "args") or {}}
            for tc in tcs
        ]

    tcid = _attr(msg, "tool_call_id")
    if tcid:
        d["tool_call_id"] = tcid
    return d


def to_lc(d: dict, *, HumanMessage=None, AIMessage=None,
          SystemMessage=None, ToolMessage=None):
    """内部 dict -> LangChain 对象。类从外部注入(避免 import 依赖)。"""
    role = norm_role(d.get("role"))
    content = d.get("content", "")
    kw = {"name": d["name"]} if d.get("name") else {}

    # ★ 注入的类是 None 就当场说清楚"忘了传", 而不是 'NoneType' object is not callable
    cls = {"user": HumanMessage, "system": SystemMessage,
           "tool": ToolMessage}.get(role, AIMessage)
    if cls is None:
        raise TypeError(f"to_lc: role={role!r} 需要对应的消息类, 但调用方没有注入它")

    if role == "user":
        return HumanMessage(content=content, **kw)
    if role == "system":
        return SystemMessage(content=content, **kw)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=d.get("tool_call_id", ""), **kw)
    if d.get("tool_calls"):
        return AIMessage(content=content or "", tool_calls=[
            {"name": t["name"], "args": t["args"], "id": t["id"], "type": "tool_call"}
            for t in d["tool_calls"]], **kw)
    return AIMessage(content=content, **kw)

def normalize(msgs) -> list:
    """整列转换"""
    return [to_dict(m) for m in msgs]


def drop_orphan_tool_results(msgs: list) -> tuple[list, int]:
    """★ 最后一道防线: 删掉找不到发起者的 tool 结果。返回 (新列表, 删掉几条)"""
    called = set()
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                called.add(tc["id"])
    out, dropped = [], 0
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id") not in called:
            dropped += 1
            continue
        out.append(m)
    return out, dropped


if __name__ == "__main__":
    # 冒烟自测(不连网): python msg_adapter.py
    from langchain_core.messages import (HumanMessage as H, AIMessage as A,
                                         SystemMessage as S)
    assert normalize([H(content="hi"), A(content="yo")]) == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    # 类名 / 类 repr 两种写法都要认出来(这一支以前会被 numpy 的 rsplit 炸掉)
    assert norm_role(repr(H)) == "user"
    assert norm_role("HumanMessage") == "user"
    assert norm_role(A(content="x").type) == "assistant"
    assert norm_role(None) == "?"
    # 内部 dict -> LC 对象, 角色别名必须转回来
    assert to_lc({"role": "user", "content": "x"}, HumanMessage=H).type == "human"
    assert to_lc({"role": "assistant", "content": "x"}, AIMessage=A).type == "ai"
    assert to_lc({"role": "system", "content": "x"}, SystemMessage=S).type == "system"
    # 成对性最后一道防线: 孤儿 tool 结果必须被丢掉
    kept, dropped = drop_orphan_tool_results([
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "name": "t", "args": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "tool", "tool_call_id": "9", "content": "孤儿"},
    ])
    assert dropped == 1 and len(kept) == 2
    print("msg_adapter OK")
