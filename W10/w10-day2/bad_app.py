# bad_app.py —— 故意错的版本，用来体验串台
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
CURRENT = {"thread_id": "default"}        # ❌ 全局变量：所有人共用一格
STORE: dict[str, list[str]] = {}


class Req(BaseModel):
    message: str
    session_id: str


@app.post("/bad")
async def bad(req: Req):
    CURRENT["thread_id"] = req.session_id      # ① 写下
    await asyncio.sleep(0.2)                   # ② 让出控制权 ← 别人在这里插队了
    tid = CURRENT["thread_id"]                 # ③ 读回，已经被别人改过
    STORE.setdefault(tid, []).append(req.message)
    return {"你以为是": req.session_id, "实际用了": tid}