# test_retriever.py
from W4.retriever import MemoryRetriever

class FakeMemory:
    """模拟 MemoryManager.recall —— 不联网也能测"""
    def recall(self, query, k=3):
        return ["用户叫小明", "喜欢蓝色", "预算2000"]

r = MemoryRetriever(mm=FakeMemory(), k=3)

# 测试1：invoke 返回 list[Document]
docs = r.invoke("用户喜欢什么")
assert isinstance(docs, list) and len(docs) == 3
print(f"✓ invoke 返回 {len(docs)} 条 Document")

# 测试2：page_content 和 metadata 都在
print(f"✓ 内容: {docs[1].page_content} | 来源: {docs[1].metadata['source']}")

# 测试3：能直接进 LCEL 链
from langchain_core.runnables import RunnableLambda
chain = r | RunnableLambda(lambda ds: "；".join(d.page_content for d in ds))
print(f"✓ 链输出: {chain.invoke('用户喜欢什么')}")