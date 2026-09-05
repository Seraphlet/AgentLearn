from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class QState(TypedDict):
    question: str
    answer: str


def classify_router(state: QState) -> str:
    """路由函数：只读state,返回字符串（此处圣洛path_map，因为返回值=节点名）"""
    q = state["question"].lower()
    if any(k in q for k in ["+", "-", "*", "/", "等于", "几"]):
        return "math_node"
    if any(k in q for k in ["搜索", "什么是"]):
        return "tool_node"
    return "chat_node"


def math_node(state: QState) -> dict:
    return {"answer": "[数学] " + state["question"]}


def tool_node(state: QState) -> dict:
    return {"answer": "[工具] " + state["question"]}


def chat_node(state: QState) -> dict:
    return {"answer": "[闲聊] " + state["question"]}


b = StateGraph(QState)
for name, fn in [("math_node", math_node), ("tool_node", tool_node), ("chat_node", chat_node)]:
    b.add_node(name, fn)

# START 直接挂条件边（入口路由）：省掉"先到分类节点再路由"那步
b.add_conditional_edges(START,classify_router)
for name in ["math_node","tool_node","chat_node"]:
    b.add_edge(name,END)

g = b.compile()
for q in  ["1+1 等于几", "搜索 LangGraph", "今天天气不错"]:
    print(g.invoke({"question": q}))

# 预期三行分别走 math/tool/chat 三个分支
