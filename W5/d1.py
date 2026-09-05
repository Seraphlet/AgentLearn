from typing import TypedDict
from langgraph.graph import StateGraph,START,END
# ── State：定义"工单上有哪些格子"
class State(TypedDict):
    question:str # 入口时填好（invoke 传入）
    answer:str  # 由节点往这里填
    turns:int #新增
# ── node：普通函数，输入整张工单，返回"我要更新的格子" ────

def think(state:State)->dict:
    return {"answer":f"[已思考]{state['question']}","turns":state["turns"]+1} #读旧值写新值

def respond(state:State)->dict:
    return {"answer":state["answer"]+"->回复完成"}

# ── 搭图：加工序 → 连传送带

builder=StateGraph(State)
builder.add_node("think",think) # 给工序起名字 + 绑函数
builder.add_node("respond",respond)
builder.add_edge(START,"think")# 投料口 → 第一道工序
builder.add_edge("think","respond")#think干完 → respond
builder.add_edge("respond",END)# respond 干完 → 出口

# ── compile：图纸变生产线（图因此变成了一个 Runnable）─────

graph=builder.compile()
# ── invoke：放一张工单，跑到底

out=graph.invoke({"question": "什么是 LangGraph?","turns":0})
print(out)# {'question': '...', 'answer': '[已思考] ... → 回复完成'}


# 在 lg_basics.py 末尾追加

print("\n--- 逐节点观察流转 ---")
for chunk in graph.stream({"question":"你好","turns":0}):
    for node_name,update in chunk.items(): # chunk = {节点名: 它更新了什么}
        print(f"[{node_name}] 填了 -> {update}")
