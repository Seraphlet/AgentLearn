#安装依赖 pip install langchain langchain-openai langchain-community python-dotenv
# pip install -U langchain langchain-openai langchain-core
# 最小LCEL链
#管道：prompt | model | parser

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 配料机：{question} → [system, user] 消息

prompt=ChatPromptTemplate([("system","你是一个简介的助手"),("user","{question}"),])

# 加工机：ChatOpenAI 接 DeepSeek

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
                )

#包装机：AIMessage → 纯字符串

parser = StrOutputParser()

# 5. 传送带：一条链
chain = prompt | model | parser

#调用

print("=== invoke ===")
result = chain.invoke({"question": "什么是 LCEL？"})

print(type(result).__name__)          # 'TextAccessor'

# 现在（问本质）：
print(isinstance(result, str))        # True ← 这才是验收

print("\n=== stream（流式，一个字一个字冒）===")
for chunk in chain.stream({"question": "数到5"}):
    print(chunk, end="", flush=True)

print("\n\n=== batch（一次处理多个）===")
results=chain.batch(
    [
        {"question":"1+1=?"},
        {"question":"2+2=?"}
    ]
)
print(results)
print("\n==链结构图")
chain.get_graph().print_ascii() # 应看到 prompt → model → parser 的流向