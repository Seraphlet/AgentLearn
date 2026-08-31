import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def get_weather(city:str)->str:
    """查询指定城市的天气。参数city:城市名。"""
    return f"{city}今天晴🌤,26℃，东♂风三级"

tools = [get_weather]

tools_by_name = {t.name: t for t in tools}   # 名字 → 工具对象 的查表

#  bind_tools：把工具"绑"给模型

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
).bind_tools(tools)

# 手写工具调用循环（ W2 的 fc_loop）
def _flatten_args(args: dict) -> dict:
    """解嵌套：{'city': {'city': '北京'}} → {'city': '北京'}"""
    for k, v in list(args.items()):
        if isinstance(v, dict) and k in v and len(v) == 1:
            args[k] = v[k]        # 只解开"外层键名=内层键名"的一层
    return args

def run(query: str, verbose: bool = True):
    messages=[HumanMessage(query)]
    ai=model.invoke(messages)
    messages.append(ai)

    round=0
    while ai.tool_calls:
        round += 1

        if verbose:
            print(f"\n[第{round}轮] 模型发起 {len(ai.tool_calls)} 个工具调用:")
            for call in ai.tool_calls:
                print(f"  工具: {call['name']}  参数: {call['args']}  解析后: {_flatten_args(call["args"])}")
        for call in ai.tool_calls:
            args=_flatten_args(call["args"])
            result=tools_by_name[call["name"]].invoke(args)
            messages.append(ToolMessage(content=str(result),tool_call_id=call["id"]))

        ai = model.invoke(messages)             # 带着结果再问模型
        messages.append(ai)
    return ai.content 
if __name__ == "__main__":
    print("\n=== 模型最终回答 ===")
    print(run("北京天气怎么样?"))