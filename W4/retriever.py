# BaseRetriver 子类

from typing import Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field,ConfigDict

class MemoryRetriever(BaseRetriever):
    """把W3的Memory Manager.recall包装成官方的Retriever
       BaseRetriever是个padantic模型，所以字段要声明类型+Field
    """

    model_config=ConfigDict(arbitrary_types_allowed=True) #允许非pydantic对象
    mm: Any =Field(description="带recall(query,k)方法的记忆管理器")
    k: int=Field(default=3,description="召回条数")

    def _get_relevant_documents(self, query:str,**kwargs) -> list[Document]:
         """唯一必须实现的抽象方法：query → Document 列表。

        **kwargs 兼容新旧版本签名（旧版无 run_manager，新版有）。
        """
         facts=self.mm.recall(query,k=self.k) # W3 的召回
         return[
              Document(page_content=f,metadata={"source":"长期记忆"})
              for f in facts    
         ]
    


