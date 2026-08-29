import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#回到根目录，硬编码
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from memory.memory_manager import MemoryManager

mm=MemoryManager(window=8,max_tokens=1500,keep=4)
mm.add({"role":"user","content":"用户喜欢Deekseek，预算2000元"},persist=True)
mm.add({"role": "user", "content": "用户叫小明，喜欢蓝色"}, persist=True)

def make_retriever(mm,k=3):
    """把mm.recall包成LangChain标准Retriever：query->list[Document]"""
    def recall_docs(query:str)->list[Document]:
        facts=mm.recall(query,k=k)
        return [Document(page_content=f,metadata={"source":"lang_term"}) for f in facts]
    return RunnableLambda(recall_docs)

retriever=make_retriever(mm,k=3)

docs=retriever.invoke("用户喜欢什么模型？")
for d in docs:
    print("召回",d.page_content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

prompt=ChatPromptTemplate.from_template(
    "根据已知信息回答。\n已知：{context}\n：问题{question}\n回答："
)

def _ftm(docs):
    return "\n".join(d.page_content for d in docs)

chain =(
    {"context": retriever | _ftm,"question":RunnablePassthrough()}
    | prompt
    |model
    |StrOutputParser()
)

q="用户喜欢什么模型，预算多少？"
print("问：",q)
print("答：",chain.invoke(q))