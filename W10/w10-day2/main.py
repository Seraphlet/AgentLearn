from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver    # ← 注意 aio 路径
from fastapi.middleware.cors import CORSMiddleware
from graph_builder import build_graph
from routes import router


DB_PATH = "sessions.db"

@asynccontextmanager
async def lifespan(app:FastAPI):
    # 启动：建连接 + 编译图（一次）
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        app.state.graph = build_graph(checkpoint=saver)
        yield
app = FastAPI(title="My Agent API", version="0.2.0", lifespan=lifespan)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",   # ← 两个都写，见坑①
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],            # 自定义头（X-API-Token）也靠这条放行
    expose_headers=["*"],
)
