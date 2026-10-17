import json

def sse(obj: dict,event: str | None=None) -> str:
    """统一 SSE 编码。
    为什么用 JSON：json.dumps 会把内容里的换行/引号/特殊字符全部转义，
    保证 data 永远是"一个物理行"，不会把帧拆坏。
    """

    prefix = f"event: {event}\n" if event else ""
    data=json.dumps(obj,ensure_ascii=False)
    return f"{prefix}data: {data}\n\n"

SSE_HEADERS={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # 关掉 nginx 等反代的缓冲，否则它会攒着发
}

