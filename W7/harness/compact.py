"""ContextCompactor —— 上下文压缩(Harness 模块④)

类比: 行李箱超重。
  要带 40 件衣服上飞机, 航空公司只让带 10 件。两种做法:
    (a) 直接扔掉 30 件                          -> 你会忘事(不可逆丢失)
    (b) 写张清单"带了 5 件外套 3 件衬衫…", 实物少带 -> 摘要
  压缩就是 (b)。

★ 关键陷阱: assistant.tool_calls 和它对应的 tool 结果是一对, 不能拆开!
  类比: 你不能只寄出"问路的话"却不寄"对方的回答" —— 很多 API 会直接 400。
  真实踩过的坑: 删旧消息时把 assistant 的 tool_calls 删了、留下 tool 结果,
  模型收到"孤儿 tool 结果" -> 报错。
  修法: 切分点如果正好落在 tool 消息上, 就往前退到发起它的 assistant 消息。

★ 压缩是"投影"不是"销毁": 原始消息仍在 SessionStore 里(存全量)。
  好处: 改策略(max 20 -> 10)不用回填历史; 压缩算错也不会不可逆。
"""

from __future__ import annotations

from regex import R

def _role(m):
    return m.get("role") or m.get("type") or "?"

def _is_tool_result(m):
    return _role(m) =="tool"

def _safe_cut(messages,cut):
    """★ 把切分点往前挪, 保证不把 assistant.tool_calls 和 tool 结果拆开"""
    cut = max(0,min(cut,len(messages)))
    while cut>0 and cut <len(messages) and _is_tool_result(messages[cut]):
        cut -=1
    return cut


class ContextCompactor:
    def __init__(self,max_messages=20,keep_recent=6,keep_system=True):
        # max_messages: 超过这个条数才压
        # keep_recent : 压缩后至少保留最近这么多条(原文)
        # keep_system : system 提示词永远保留
        # 可以考虑新增消息条数和文字总数触发压缩，如果每条文字量级很大,就不能按照条数压缩，要按照tooken压缩
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.keep_system = keep_system
        self.compactions = 0
        self.last_summary = None
    # ---------------------------------------------------------------- 判定
    def needs_compaction(self,messages):
        return len(messages)> self.max_messages

    # ---------------------------------------------------------------- 主入口
    def maybe_compact(self,messages,summarize=None):
        """
        返回压缩后的【新列表】; 不需要压缩(或切不动)时返回 None。
        ★ 返回 None 而不是原列表: 让调用方能区分"压过了"和"没动"。
        """
        if not self.needs_compaction(messages):
            return None 
        head,body =[],list(messages)
        if self.keep_system and body and _role(body[0]) =="system":
            head,body=[body[0]],body[1:]

        cut =max(0,len(body) - self.keep_recent)
        cut=_safe_cut(body,cut) # ★ 成对性修正
        if cut ==0:
            return None  # 切不动(keep_recent 太大), 宁可不动


        old, recent = body[:cut], body[cut:]
        summary=self._summarize(old,summarize)
        self.compactions+=1
        self.last_summary=summarize
        # ★ 摘要 + 只保留最近 K 条, 一起返回(避坑: 别只插摘要不删历史)
        return head + [{"role": "system", "content": f"[历史摘要] {summary}"}] + recent

    def _summarize(self,old,summarize):
        if summarize is not None:
            return summarize(old) # ★ 依赖注入: 有 LLM 用 LLM
        roles ={}
        for m in old:
            r=_role(m)
            roles[r] =roles.get(r,0)+1
        detail = ", ".join(f"{k}x{v}" for k, v in sorted(roles.items()))
        return f"已折叠 {len(old)} 条早期消息 ({detail})"   # 没 LLM 也能跑