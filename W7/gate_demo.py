"""W7-Day4 交付物④: 危险工具触发审批, 拒绝时【确实未执行】

两种模式:
    python gate_demo.py --offline          # 不调 LLM, 假 agent 节点 → 0 成本, 秒出
    python gate_demo.py                    # 真 LLM (需 $env:DEEPSEEK_API_KEY)

四个场景:
    --case safe        安全工具, 不过闸门审批直接执行
    --case reject      危险工具 + 人工拒绝  → 期望: 文件还在
    --case approve     危险工具 + 人工批准  → 期望: 文件被删
    --case whitelist   危险工具 + 白名单放行 → 期望: 文件被删  (★ 正向对照)

设计要点(面试可讲):
  1) 危险工具真的动文件系统 —— 验证才有物理证据, 不是靠返回值自说自话
  2) Gate 判定只做一次: 路由用的判定结果经 precheck 传给 call_tool, 不重复问
  3) Gate.check() 是纯函数(不记账/不改状态) —— 所以 HITL 恢复时节点被重放也安全
"""


from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from tabnanny import check

from huggingface_hub.constants import SANDBOX_SERVER_BUCKET
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import sandbox
from xlwings import ret

from W6.w6d6_c_e2e import approve
from harness.registry import ToolRegistry
from harness.gate import PermissionGate, Decision
from harness.executor import call_tool

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "_sandbox")
AUDIT = os.path.join(HERE, "demo_audit.jsonl")
VICTIM = "victim.txt"

# ---------------------------------------------------------------- 工具 + 闸门

def build_harness():
    """每次调用都从干净的沙箱开始"""
    if os.path.exists(SANDBOX):
        shutil.rmtree(SANDBOX)
    os.makedirs(SANDBOX,exist_ok=True)
    with open(os.path.join(SANDBOX,VICTIM),"w",encoding="utf-8") as f:
        f.write("我很重要")
    if os.path.exists(AUDIT):
        os.remove(AUDIT)

    reg=ToolRegistry()

    @reg.register(tag=["fs","read"])
    def list_files() ->str:
        """列出沙箱目录里的文件(只需)"""
        return ",".join(sorted(os.listdir(SANDBOX))) or "(空)"

    @reg.register(level="dangerous",tags=["fs","write"])
    def delete_file(filename:str) ->str:
        """删除沙箱里的文件(不可逆)"""
        os.remove(os.path.join(SANDBOX,os.path.basename(filename))) # ★ 真动文件系统
        return f"已删除 {filename}"
    return reg,PermissionGate(reg,audit_path=AUDIT)


# ------------------------------------------------------------------ 图节点
def  make_offline_agent(tool_name,tool_args):
    """假 agent: 不调LLM，直接吐一个写死的 tool_call"""
    def agent(state):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": tool_name, "args": tool_args, "id": "call_1"}])]}
    return agent

def make_llm_agent(llm, lc_tools):
    """真 agent: 让模型自己决定调什么"""
    def agent(state):
            return {"messages": [llm.bind_tools(lc_tools).invoke(state["messages"])]}
    return agent

def make_tool_node(registry,gate,counter):
    def tool_node(state):
        last = state["messages"][-1]
        out = []
        for tc in (getattr(last,"tool_calls",None) or []):
            name,args =tc["name"],tc["args"]

            counter["check"]+=1
            res=gate.check(name,actor="agent")

            if res.decision is Decision.NEED_APPROVAL:
                approve=interrupt({
                    "action": "approve_tool",
                    "tool": name,
                    "args": args,
                    "reason": res.reason,
                })
                approve=(lambda *_:approve)
            else:
                approve=None

            outcome=call_tool(registry,gate,name,args,actor="agent",approve=approve,precheck=res) #复用判定

            text = outcome.get("result") if outcome["ok"] else outcome["error"]
        return {"messages": out}
    return tool_node

def route_after_agent(state):
    last= state["messages"][-1]
    return "tools" if getattr(last,"tool_calls",None) else END

def build_graph(registry,gate,*,agent_fn,counter):
    b=StateGraph(MessagesState)
    b.add_node("agent",agent_fn)
    b.add_node("tools",make_tool_node(registry,gate,counter))
    b.addedge(START,"agent")
    b.add_conditional_edges("agent",route_after_agent,{"tools":"tools",END:END})
    b.add_edge("tools",END) 
    return b.compile(checkpointer=MemorySaver())


def run_case(case, *, offline=True):
    reg, gate = build_harness()
    counter = {"check": 0}

    if case == "safe":
        tool_name, tool_args, resume = "list_files", {}, None
    elif case == "reject":
        tool_name, tool_args, resume = "delete_file", {"filename": VICTIM}, False
    elif case == "approve":
        tool_name, tool_args, resume = "delete_file", {"filename": VICTIM}, True
    elif case == "whitelist":
        tool_name, tool_args, resume = "delete_file", {"filename": VICTIM}, None
        gate.allow = {"delete_file"}
    else:
        raise SystemExit(f"未知场景: {case}")

    if offline:
        agent_fn = make_offline_agent(tool_name, tool_args)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com",
                         temperature=0)
        agent_fn = make_llm_agent(llm, reg.to_langchain_tools())   # 见第五节补丁

    graph = build_graph(reg, gate, agent_fn=agent_fn, counter=counter)
    config = {"configurable": {"thread_id": f"demo-{case}"}}

    first = graph.invoke({"messages": [HumanMessage(
        content=f"请调用工具 {tool_name}")]}, config)
    interrupted = "__interrupt__" in first
    pauses = len(first.get("__interrupt__") or ())

    if interrupted and resume is not None:
        graph.invoke(Command(resume=resume), config)

    



