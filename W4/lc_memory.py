import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# 修复说明：
# 为什么 try 会找不到包？
# 因为这里使用的是相对导入：
#   from ..memory.memory_manager import MemoryManager
# 当你直接运行脚本时（例如：python lc_memory.py），Python 会把这个文件当成顶层脚本，__package__ 为空，
# 也就无法解析 "..memory" 这种上级包路径，所以会抛出 ModuleNotFoundError / ImportError。
# 正确做法有两种：
# 1）如果 "memory" 是项目根目录下的包，则先把项目根目录加入 sys.path，再用绝对导入；
# 2）如果这个文件位于包内，则应该用 "python -m 包名.模块名" 方式启动，而不是直接执行脚本文件。
# 下面是正确的注释示例，放在这里不直接覆盖原逻辑：
#
# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
# from memory.memory_manager import MemoryManager
#
# 也可以直接写成绝对导入（取决于你的项目目录结构）：
# from W4.memory.memory_manager import MemoryManager
#
# 真实的修复模板应当替换这段 try 逻辑：
# try:
#     from memory.memory_manager import MemoryManager
#     _mm = MemoryManager(window=6, max_tokens=500)
#     def get_history(x=None):
#         q = (x or {}).get("question", "")
#         return [_to_lc(m) for m in _mm.get_content(query=q)]
# except Exception as e:
#     print(f"[降级]未找到memory包({e}),使用最小窗口版")
#     class _MinilMM:
#         def __init__(self, window=6):
#             self.buf, self.window = [], window
#         def get_context(self, query=None):
#             return self.buf[-self.window:]
#         def add_user(self, t):
#             self.buf.append({"role": "user", "content": t})
#         def add_ai(self, t):
#             self.buf.append({"role": "assistant", "content": t})
#     _mm = _MinilMM()
#     def get_history(x=None):
#         return [_to_lc(m) for m in _mm.get_context()]
#
# 说明：
# - 这里的关键点是：相对导入必须在包上下文中才成立；
# - 直接运行脚本时，绝对导入 + sys.path 补丁最稳妥；
# - 另外，get_history 需要兼容 x=None，否则链调用时可能出现参数不匹配。

#桥的包装器：dict 消息 → LangChain BaseMessage
def   _to_lc(m:dict):
    role,content=m.get("role","user"),m.get("content","")
    if role=="system":
        return SystemMessage(content=content)
    if role=="assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)

#接入 W3 自研 MemoryManager（失败则降级最小窗口版）

try:
    from memory.memory_manager import MemoryManager
    _mm=MemoryManager(window=6,max_tokens=500)
    def get_history(x):
        #真实版：query=本轮问题——>长期召回+滚动窗口
        return [_to_lc(m) for m in _mm.get_context(query=x.get("question"))]
except Exception as e:
    print(f"[降级]未找到memory包({e}),使用最小窗口版")
    class _MinilMM:
        def __init__(self,window=6):
            self.buf,self.window=[],window
        def get_context(self,query=None):
            return self.buf[-self.window:]
        def add_user(self,t):
            return self.buf.append({"role":"user","content":t})
        def add_ai(self,t):
            self.buf.append({"role":"assistant","content":t})
    _mm= _MinilMM()
    def get_history(x):
        return [_to_lc(m) for m in _mm.get_context()]

#  链：链首选历史，再 prompt|model|parser 

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有记忆的助手，记得之前聊过的内容。"),
    ("placeholder", "{history}"),            # 关键：展开历史 messages
    ("human", "{question}"),
])

load_mem=RunnableLambda(lambda x:{**x,"history":get_history(x)})
chain= load_mem | prompt | model | StrOutputParser()

def ask(question:str)->str:
    ans = chain.invoke({"question":question})
    if hasattr(_mm,"add_user"):
        _mm.add_user(question); _mm.add_ai(ans)
    else:
        _mm.add({"role":"user","content":question})
        _mm.add({"role":"assistant","content":ans})
    return ans

if __name__ == "__main__":
    # 验收：第 2、3 问能答出「篮球」「小明」=> 记忆生效 [3]
    for q in ["我叫小明，喜欢打篮球", "我刚说的爱好是什么？", "我名字叫什么？"]:
        print(f"Q: {q}\nA: {ask(q)}\n")

    # 步骤B：确认 load_mem 在链首 + 看历史在累积
    print("=== 链结构图（load_mem 应排最前）===")
    chain.get_graph().print_ascii()
    print("=== 当前记忆历史 ===")
    for m in _mm.get_context():
        print(f"  {m.get('role')}: {m.get('content')}")