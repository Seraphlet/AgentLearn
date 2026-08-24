#带多轮记忆的agent
import sys,os,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager
from Agent.agent import RealLLM

class SummarizeLLM:
    def __init__(self,llm):
        self.__init__=llm

    def summarize(self,text:str)->str:
        sys_prompt = ("你是对话摘要助手。把下面的对话浓缩成200字以内的摘要，"
                      "保留关键事实、决策和用户偏好，省略寒暄。只输出摘要本身。")
        resp=self._llm.chat[{"role":"system","content":system_prompt},{"role":"user","content":text},]
        return resp.choices[0].message.content
def chat_with_memory_tools(llm, messages):
    for __ in range(3):
        resp=llm.chat(messages,tools=MEMORY_TOOLS_SCHEMA)
        msg=resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for tc in msg.tool_calls:
            arg = json.loads(tc.function.arguments) # 结构化参数，不用解析
            result = MEMORY_TOOLS_FUNC[tc.function.name](**arg)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(result)})
    return "达到最大工具轮数"




def run():
    llm=RealLLM()
    mm=MemoryManager(window=8,max_tokens=1500,keep=4,llm=SummarizeLLM(llm))
    mm.add({"role": "system", "content": "你是带记忆的客服助手。"})
    set_memory_manager(mm)
    print("提示：说『记住:我喜欢用DeepSeek』可落长期记忆（跨会话保留）\n")
    while True:
        u=input("你：")
        if u.lower() in ["exit","quit","退出"]:
            break
        mm.add({"role":"user","content":u})
        ctx=mm.get_context(query=u)
        reply=chat_with_memory_tools(llm,ctx)
        mm.add({"role": "assistant", "content": reply})
        mm.maybe_compress()
        print("Agent:", reply)

if __name__ == "__main__":
    run()

    