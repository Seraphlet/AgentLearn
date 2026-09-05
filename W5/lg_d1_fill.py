
from typing import TypedDict
from langgraph.graph import StateGraph,START,END

class AddState(TypedDict):
    a:int
    b:int
    result:int

def add_node(state:AddState)->dict:
    return {"result":state["a"]+state["b"]}

b=StateGraph(AddState)
b.add_node("add",add_node)
b.add_edge(START,"add")
b.add_edge("add",END)
g=b.compile()
print("加法器 invoke:",g.invoke({"a":3,"b":5}))
for step in g.stream({"a":3,"b":5}):
    print("stream",step)
class ConcatState(TypedDict):
    parts:list[str]
    joined:str

def upper_node(state:ConcatState)-> dict:
    return {"parts":[p.upper()for p in state["parts"]]}

def join_node(state:ConcatState)-> dict:
    return {"joined": " | ".join(state["parts"])}

def length_node(state: ConcatState) -> dict:
    return {"joined": state["joined"] + f" (len={len(state['joined'])})"}

b2 = StateGraph(ConcatState)
for n, fn in [("upper", upper_node), ("join", join_node), ("length", length_node)]:
    b2.add_node(n, fn)

b2.add_edge(START, "upper")
b2.add_edge("upper", "join")
b2.add_edge("join", "length")
b2.add_edge("length", END)

g2 = b2.compile()
print("\n拼接 invoke:", g2.invoke({"parts": ["hello", "world"]}))
for step in g2.stream({"parts": ["hello", "world"]}):   # 亲眼看每节点改哪格
    print("  stream:", step)


