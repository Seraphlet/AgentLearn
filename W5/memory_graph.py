"""W5-D6 项目日：MemoryManager(类) → MemoryManagerGraph(图)"""
import os
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ── 1. State（关键：recent 是普通字段，可整体覆盖；messages 带 reducer 只追加）──

class MemoryState(BaseModel):
    messages: Annotated[list[Any],add_messages] = Field(default_factory=list) # 完整历史
    recent: list[Any] = Field(default_factory=list)  # ⚠️ 压缩产物窗口（无 reducer → 可覆盖）
    summary: str =Field(default="",description="压缩后保留的历史摘要")
    retrieved: list[str] = Field(default_factory=list,description="长期召回内容")
    memory_path: str =Field(default="",description="本次走的路径:short/long/compress")

# ── 2. 长期记忆存储（简化版 dict，进程内共享；W3 你已做过 Chroma 落盘，可替换）──

memory_store = {}
@tool
def save_long_term(key:str,content:str) -> str:
    """把重要信息存入长期记忆。当用户说'记住...'或提到个人偏好时调用。"""
    memory_store[key] = content
    return f"已记住: {key} = {content}"

@tool
def recall_long_term(query: str)->str:
    """从长期记忆检索信息。当用户询问之前提过的事实/偏好时调用。"""
    hits=[f"{k}:{v}"for k,v in memory_store.items()if query in k or query in v]
    return "\n".join(hits) if hits else "(长期记忆中没有相关内容)"

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ── 3. 路由节点（原来 if/else 决策 → 独立节点，只输出 memory_path）──
def fmt(msgs)->str:
    return "\n".join(m.content if hasattr(m,"content") and m.content else str(m) for m in msgs)

def route_node(state: MemoryState) ->dict:     
    prompt=f"""判断这段对话最需要哪种记忆处理，只输出一个词：
- long: 用户明确要求记住信息(说'记住...')，或询问之前提过的事实/偏好
- short: 普通对话，无需特殊处理
对话历史:
{fmt(state.messages[-4:])}"""
    decision =llm.invoke(prompt).content.strip().lower()
    if "long" in decision:
        return {"memory_path":"long"}
    if len(state.messages) > 10:              # 规则优先：超阈值直接压缩
            return {"memory_path": "compress"}
    return {"memory_path": "short"}
        

# ── 4. 三条记忆路径节点 ──
def short_node(state: MemoryState) -> dict:
    """普通对话：消息已在 messages 自然累加，recent 同步为当前窗口"""
    return {"recent": state.messages[-6:], "memory_path": "short"}

def long_node(state: MemoryState) -> dict:
    """长期记忆路径：真正的'存什么/取什么'由 agent 调工具完成，这里只标记路径"""
    return {"memory_path": "long"}

def compress_node(state: MemoryState) -> dict:
    """压缩：旧消息 → LLM 摘要；⚠️ 裁剪写入 recent（普通字段），不碰 messages"""
    if len(state.messages)<=4:
        return {"recent": state.messages, "summary": state.summary, "memory_path": "compress"}
    old, recent = state.messages[:-4], state.messages[-4:]
    sp = f"把以下对话压缩成一句摘要，保留关键事实(人名/偏好/结论):\n{fmt(old)}\n(已有摘要:{state.summary or '无'})"
    new_summary = llm.invoke(sp).content.strip()
    return {"recent": recent, "summary": new_summary, "memory_path": "compress"}

# ── 5. agent 节点（复用 D3 ReAct，读 recent 窗口 + summary/retrieved 注入）──
def agent_node(state: MemoryState) -> dict:
    # 防死循环：消息过多强制收口（D3 老规矩）
    window = state.recent if state.recent else state.messages
    if len(window) > 3:
        window = window + [{"role": "user", "content": "请基于以上内容直接给出最终答案，不要再调用工具。"}]
    ctx = []
    if state.summary:
        ctx.append(("system", f"历史摘要:{state.summary}"))
    if state.retrieved:
        ctx.append(("system", f"长期记忆:\n{'，'.join(state.retrieved)}"))
    ctx.extend(window)
    return {"messages": [llm.bind_tools([save_long_term, recall_long_term]).invoke(ctx)]}

builder = StateGraph(MemoryState)
builder.add_node("route", route_node)
builder.add_node("short", short_node)
builder.add_node("long", long_node)
builder.add_node("compress", compress_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([save_long_term, recall_long_term]))

builder.add_edge(START, "route")
builder.add_conditional_edges("route",lambda s:s.memory_path,{"short": "short", "long": "long", "compress": "compress"})
for n in ["short", "long", "compress"]:
    builder.add_edge(n,"agent")    # 三路径都汇入 agent

builder.add_conditional_edges("agent", tools_condition) # 有 tool_calls → tools，否则 END
builder.add_edge("tools", "agent") # 工具执行完回 agent

memory_graph = builder.compile()
print("✅ 图编译成功")

# ── 7. 测试三路径 + 记住/召回端到端 ──
if __name__ == "__main__":
    # 测试 A：compress（塞 12 条 > 阈值）
    r1 = memory_graph.invoke({"messages": [("user", f"第{i}句闲聊内容{i}") for i in range(12)]})
    print("\n[A compress] 路径:", r1["memory_path"])
    print("r1 keys:", r1.keys())
    print("路径:", r1["memory_path"]) 
    print("  压缩后 recent 消息数:", len(r1["recent"]), "| 摘要:", r1.get("summary", "")[:40])

    # 测试 B：long 记住（先存）
    r2 = memory_graph.invoke({"messages": [("user", "记住我喜欢喝美式咖啡")]})
    print("\n[B 记住] 路径:", r2["memory_path"], "| 记忆库:", memory_store)

    # 测试 C：long 召回（后问——同一进程内 dict 共享）
    r3 = memory_graph.invoke({"messages": [("user", "我喜欢喝什么咖啡？")]})
    print("[C 召回] 路径:", r3["memory_path"])
    for ev in memory_graph.stream(
        {"messages": [("user", "我喜欢喝什么咖啡？")]}, stream_mode="updates"
        ):
        for node, update in ev.items():
            msgs = update.get("messages", [])
            if msgs:
                last = msgs[-1]
                text = getattr(last, "content", "") or str(last)   # 取文本内容
                print(f"  {node}: {text[:60]}")                     # 对字符串切片，安全
            else:
                print(f"  {node}: {update}")

    # 测试 D：short 普通对话
    r4 = memory_graph.invoke({"messages": [("user", "你好，简单介绍下你自己")]})
    print("\n[D short] 路径:", r4["memory_path"], "| 回复:", r4["messages"][-1].content[:30])