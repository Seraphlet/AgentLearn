from email import message
from re import S

from pydantic import BaseModel, Field
class MessageItem(BaseModel):
    role:str
    content:str
    truncated: bool=False

class HistoryResponse(BaseModel):
    session_id:str
    total: int = Field(description="该会话总消息数")
    returned: int
    next_cursor:int | None =Field(default=None,description="还有更早的消息时给偏移量")
    messages: list[MessageItem]
class SessionItem(BaseModel):
    session_id: str
    turns: int
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