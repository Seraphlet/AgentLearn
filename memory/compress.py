"""
上下文压缩——>一条滚动摘要，永远只有一条
"""

def _estimate_tokens(text:str) ->int:#token估算
    """
    估算tokens，达到上限压缩一次
    """
    return max(1,int(len(text)*0.6))

def _fmt(messages) ->str:#历史消息拼接
    """
    消息列表，摘要原材料
    """
    return "\n".join(f"{m['role']}:{m['content']}" for m in messages)


def compress(messages:list[dict],llm,keep:int=4) ->list[dict]:#压缩逻辑函数
    """
    滚动压缩版本
    """
    sys_msgs=[m for m in messages if m.get("role")=="system" and not m.get("is_summary")]
    old_sums=[m for m in messages if m.get("is_summary")]
    others=[m for m in messages if m.get("role")!="system" and not m.get("is_summary")]
    if len((others))<=keep:
        return messages
    old,recent=others[:-keep],others[-keep:]

    # 滚动关键：旧摘要 + 新滚出窗口的原文 → 合并成一条新摘要
    text=_fmt(old)
    if old_sums:
        text=f"已有摘要:{old_sums[-1]['content']}\n新增对话：\n{text}"
    summary=llm.summarize(text)
    summary_msg={"role":"system","content":f"历史摘要：{summary}","is_summary":True}
    return sys_msgs + [summary_msg] + recent

def maybe_compress(messages,llm,max_tokens:int=3000,keep=4):#自动压缩检测
    """
    自动触发，超阈值才压缩，没超原样返回
    """
    if (_estimate_tokens(_fmt(messages))>max_tokens and len(messages)> keep+1):
        return compress(messages,llm,keep=keep)
    return messages









