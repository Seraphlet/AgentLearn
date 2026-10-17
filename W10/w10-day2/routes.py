import asyncio
from re import S

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from graph_builder import extract_text, text_of     # text_of 就是 Day2 的 _text_of，改成公开名
from models import ChatRequest, ChatResponse
from sse import SSE_HEADERS, sse

router = APIRouter()


def get_graph(request: Request):
    """依赖注入：路由不关心graph从哪来"""
    graph = getattr(request.app.state,"graph",None)
    if graph is None:
        raise HTTPException(status_code=503,detail="graph 尚未就绪")
    return graph

@router.get("/health")
async def health():
    return {"status":"health","service":"agent-api"}

@router.post("/chat",response_model=ChatResponse,summary="非流式对话")
async def chat(req: ChatRequest,graph=Depends(get_graph)):
    config = {"configurable":{"thread_id":req.session_id}}

    try:
        result =await graph.ainvoke(
            {"messages":[("user",req.message)]},
            config,
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"{type(e).__name__}:{e}")
    msgs=result.get("messages",[])
    return ChatResponse(reply=extract_text(msgs),session_id=req.session_id,turns=len(msgs))

@router.get("/stream/demo",summary="SSE 管道自检 （不需要llm）")
async def stream_demo():
    async def gen():
        tokens=  ["你好", "，", "这是", "SSE", "演示", "\n", "第二行：含换行符的内容"]
        for i, tok in enumerate(tokens):
            yield sse({"i": i,"t": tok},event="token")
            await asyncio.sleep(0.3)
        yield sse({"done": True,"n": len(tokens)},event="done")
    return StreamingResponse(gen(),media_type="text/event-stream",headers=SSE_HEADERS)

# ========== ② 真图流式 ==========
@router.post("/chat/stream",summary="流式对话")
async def chat_stream(req:ChatRequest,request:Request,graph=Depends(get_graph)):
    config = {"configurable": {"thread_id": req.session_id}}

    async def gen():
        n=0
        try:
            async for chunk,meta in graph.astream(
                {"messages": [("user",req.message)]},
                config,
                stream_mode="messages"
            ):
                node=meta.get("langgraph_node","")
                text=text_of(chunk)
                # 过滤 1：空的 chunk（工具调用阶段，content 常为空）
                if not text:
                    continue
                # 过滤 2：只放行"要对用户说话"的节点（见第 4 节纠错 ③）
                if node and not node.startswith(("agent","respond","chat")):
                    continue
                n=n+1
                yield sse({"i": n,"t": text,"node": node},event="token")
                if await request.is_disconnected():   # 用户关了页面就别白干了
                    break
            yield sse({"done": True, "n": n}, event="done")
        except asyncio.CancelledError:
            raise   # 客户端断开，正常退出
        except Exception as e:
            # ③ 错误也走 SSE，别让客户端"静默断流"
            yield sse({"error": f"{type(e).__name__}: {e}"}, event="error")
            yield sse({"done": True, "n": n, "ok": False}, event="done")
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


