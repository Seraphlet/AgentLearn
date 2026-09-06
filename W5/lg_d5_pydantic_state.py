from typing import TypedDict
from pydantic import BaseModel, Field

# —— TypedDict：只在静态检查时有类型提示，运行时形同虚设 ——

class TD(TypedDict):
    count: int

td: TD ={"count": "不是整数"}  # 运行时完全不报错

# —— Pydantic：运行时真校验 ——
class PD(BaseModel):
    count: int =Field(ge=0,description="必须是非负数")

try:
    PD(count=-5) #ValidationError: 类型错误
except:
    print("拦截了类型错误")

print(PD(count=5).model_dump())

# 验证：reducer 字段追加 vs 普通字段覆盖
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class DemoState(BaseModel):
    messages: Annotated[list,add_messages]=[]# 有 reducer：追加
    counter: int=0 #无reducer:覆盖

def n1(state: DemoState) -> dict:
    return {"messages": [{"role": "user", "content": "第一条"}],"counter": 1}

def n2(state: DemoState) -> dict:
    return {"messages": [{"role": "assistant", "content": "第二条"}], "counter": 2}

b = StateGraph(DemoState)
b.add_node("n1",n1)
b.add_node("n2",n2)
b.add_edge(START,"n1")
b.add_edge("n1","n2")
b.add_edge("n2",END)
g=b.compile()
final=g.invoke({})
print("messages 长度:", len(final["messages"]))   # 2 → reducer 追加了两条
print("counter:", final["counter"])  

# 计数器：每节点返回 {"visits": 1} 就 +1

def increment(current: int,new: int)-> int:
    return current +new

# 日志：字符串列表拼接

def add_log(current: list[str],new: list[str])-> list[str]:
    return current + new

class CountState(BaseModel):
    visit: Annotated[int,increment] =0
    log: Annotated[list[str],add_log]=[]
    