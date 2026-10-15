from pydantic import BaseModel, Field
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息")
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",              # 原计划提醒过：thread_id 别带非法字符
        description="会话标识，前端生成；只允许字母数字 _ -",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    turns: int = Field(description="当前会话累计消息数，用来肉眼验证持久化")