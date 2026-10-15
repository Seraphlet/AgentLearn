# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
app = FastAPI(title="My Agent API", version="0.1.0")
# ===== ② 请求模型：这就是"验票闸机" =====
class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, description="要回显的内容")
    times: int = Field(default=1, ge=1, le=5, description="重复次数")

# ===== ① 三个接口 =====
@app.get("/ping")
def ping():
    """最简单的存活探测"""
    return {"status": "ok"}


@app.get("/health")
def health():
    """生产用健康检查（后面 Docker 会用到）"""
    return {"status": "healthy", "service": "agent-api"}


@app.post("/echo")
def echo(req: EchoRequest):
    """演示请求模型 + 校验"""
    if req.times > 3:
        raise HTTPException(status_code=400, detail="times 太大")
    return {"you_said": req.message * req.times, "times": req.times}