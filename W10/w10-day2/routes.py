from fastapi import APIRouter, Depends, HTTPException, Request

from graph_builder import extrace_text
from models import ChatRequest, ChatResponse

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
            {"messages":["user",req.message]},
            config,
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"{type(e).__name__}:{e}")
    msgs=result.get("messages",[])
    return ChatResponse(reply=extrace_text(msgs),mession_id=req.session_id,turns=len(msgs))
