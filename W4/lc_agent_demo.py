# W4/lc_agent_demo.py —— W4-D6：带记忆 + 工具的多轮对话 demo
# 一句话：D1~D5 的零件装成一辆能开的车（REPL 驾驶舱）
import os,sys
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#回到根目录，硬编码
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env",override=True)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#回到根目录，硬编码
from lc_tools_multi import tools, tools_by_name      # D5：三工具 + 查表
from memory.memory_manager import MemoryManager 

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
).bind_tools(tools,parallel_tool_calls=False)   

mm=MemoryManager(window=8,max_tokens=1500,keep=4)

prompt=ChatPromptTemplate.from_messages([
    ("system","你是有记忆，能调工具的助手。历史对话和检索到用户事实、喜好和用户说记下的内容都在history里"),
    ("placeholder","{history}"),#历史消息原地展开
    ("human","{question}"),
])
# ----------D3 的桥：dict 消息 → BaseMessage ----------
def _to_lc(m:dict):
    role,content=m.get("role","user"),m.get("content","")
    if role=="system":
        return SystemMessage(content=content)
    if role=="user":
        return AIMessage(content=content)
    return HumanMessage(content=content)
MAX_ROUNDES=3

def _normalize_args(args:dict)->dict:
    """任意层嵌套都取第一个字符串值"""
    for k,v in list(args.items()):
        while isinstance(v,dict) and v:
            v=next(iter((v.values)))
        args[k]=v
    return args

def _run_tools(messages,verbose=True):
    """串行执行：一次只处理一个 tool_call，最稳的 DeepSeek 兼容方案"""
    round_ = 0
    ai = messages[-1]
    while ai.tool_calls and round_ < MAX_ROUNDES:
        round_ += 1
        if verbose:
            print(f"  [工具第{round_}轮] 模型发起 {len(ai.tool_calls)} 个调用:")
            for call in ai.tool_calls:
                print(f"    → {call['name']}({_normalize_args(call['args'])})")
        for call in ai.tool_calls:                       # 注意：不是 tool_calls[0]！
            args = _normalize_args(call.get("args", {}))
            try:
                result = tools_by_name[call['name']].invoke(args)
                content = str(result) if result is not None else "无结果"
            except Exception as e:
                content = f"工具调用失败: {e}"            # 失败也必须回填，否则协议违例
            messages.append(ToolMessage(content=content, tool_call_id=call['id']))
        ai = model.invoke(messages)                  # 模型看到结果后，自己决定还要不要发下一个
        messages.append(ai)
    return ai
# ask() = 取记忆 → 生成 → 工具循环 → 写回 ----------
def ask(question:str)->str:
    history=[_to_lc(m) for m in mm.get_context(query=question)]
    messages=prompt.format_messages(history=history,question=question)
    ai =model.invoke(messages)
    messages.append(ai)
    ai = _run_tools(messages)  
    mm.add({"role": "user", "content": question})
    mm.add({"role": "assistant", "content": ai.content})
    return ai.content

if __name__=="__main__":
    print("=== W4-D6：带记忆 + 工具的多轮对话 demo ===")
    while True:
        question=input("\n用户：")
        if question in ["exit","quit"]:
            break
        print("助手:", ask(question))