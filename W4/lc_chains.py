import os 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import(
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
load_dotenv()
model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

parser=StrOutputParser()

answer_prompt=ChatPromptTemplate.from_messages([
    ("system","你是一个简洁回答的助手"),
    ("user","{question}")
])

kw_prompt=ChatPromptTemplate.from_messages([
    ("system","你是关键词提取器，请输出3个关键词，用逗号分隔"),
    ("user","{question}")
])
answer_chain = answer_prompt | model | parser
kw_chain = kw_prompt | model | parser

parallel_chain=RunnableParallel(
    answer=answer_chain,
    keywords=kw_chain,
)
print("=== 步骤A1：RunnableParallel 并行 ===")
res = parallel_chain.invoke({"question": "什么是LCEL？"})
print(res)

print("类型:", {k: type(v).__name__ for k, v in res.items()})

print("\n=== 步骤A2：RunnablePassthrough.assign 追加字段 ===")

enrich=RunnablePassthrough.assign(
    长度=lambda x: len(x["question"]),
    前5个字=lambda x: x["question"][:5],
)
out=enrich.invoke({"question":"什么是LCEL？"})
print(out)

# RunnableBranch 分支
weather_prompt=ChatPromptTemplate.from_messages([
    ("system","你是天气助手，用一句话回答天气问题"),
    ("user","{question}"),
])

calc_prompt=ChatPromptTemplate.from_messages([
    ("system","你是计算小助手，只输出计算结果"),
    ("user","{question}"),
])

default_prompt=ChatPromptTemplate.from_messages([
    ("system","你只智能助手,简要回答"),
    ("user","{question}"),
])

weather_chain = weather_prompt | model | parser
calc_chain = calc_prompt | model | parser
default_chain = default_prompt | model | parser

branch=RunnableBranch(
     (lambda x: "天气" in x["question"], weather_chain),   # 条件1 → 天气链
    (lambda x: "计算" in x["question"], calc_chain),      # 条件2 → 计算链
    default_chain,                                   
)

print("\n=== 步骤C：RunnableBranch 分支 ===")
for q in ["北京天气怎么样？", "计算 12*8", "你好呀"]:
    ans = branch.invoke({"question": q})
    print(f"问: {q}\n答: {ans}\n")


# 步骤 D：get_graph().print_ascii() 可视化

print("=== 步骤D：并行链结构图 ===")
parallel_chain.get_graph().print_ascii()

print("\n=== 步骤D：分支链结构图 ===")
branch.get_graph().print_ascii()

# 步骤 E：RunnableLambda 接自研函数

def TextCleansing(text:str)->str:
    """模拟自研函数，无实际逻辑处理"""
    return " ".join(text.split())[:50]

chain_with_func=answer_prompt | model | parser | RunnableLambda(TextCleansing)

print("\n=== 步骤E：RunnableLambda 加工站 ===")

out_e=chain_with_func.invoke({"question": "用一句话介绍 LangChain"})
print(out_e)
print("类型:", type(out_e).__name__)