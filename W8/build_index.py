"""
W8-Day2  ③嵌入 + ④入库(落盘)
================================================================
Day1 管"把文档切成块"; 今天管"把块变成能被按意思搜出来的库"。

运行:
    python w8\build_index.py --smoke        # ★ 先跑这个: 离线假嵌入, 0 成本, 只验链路
    python w8\build_index.py               # 本地 BGE(需 sentence-transformers, 首跑要下模型)
    python w8\build_index.py --embed openai # 云端 text-embedding-3-small(需 OPENAI_API_KEY)
    python w8\build_index.py --reset        # 推倒重建(删掉 chroma_db 重新嵌入)

为什么要有 --smoke:
    把"链路通不通"(确定性) 和"语义好不好"(智能性) 拆开测。
    假嵌入是"同样的文本永远得到同样的向量", 它能证明落盘/加载/坐标都对,
    但绝不能拿它的检索结果评价质量。
    顺序永远是: 先测确定性, 再测智能性。
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import langchain_openai

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # 让 w8 目录里的模块能被 import

import day1_load_split as d1    

PERSIST_DIR = str(HERE / "chroma_db")
COLLECTION = "my_docs"
MIN_CHARS = 20        # ★ 过滤 Day1 那个 1 字符 "。" 垃圾块(它进库就是纯噪声)
BATCH = 64    

# ===================== 嵌入三选一 =====================

class HashEmbeddings(Embeddings):
    """离线假嵌入 —— 只看"有没有关系", 不看"有多像"。

    做法: 把文本切成 2 字组合, 哈希到固定维度, 再 L2 归一化。
      - 同样文本 → 同样向量      (确定性 ✅)
      - 共享汉字的文本 → 有非零相似度 (能搜到东西 ✅)
      - 同义不同字("咖啡" vs "美式") → 搜不到  (所以不能评价质量 ❌)
    用途只有一个: 冒烟测试链路。
    """
    def __init__(self,dim: int=256):
        self.dim=dim

    def _vec(self,text:str) -> list[float]:
        v=[0.0] *self.dim
        for i in range(len(text)-1):
            gram=text[i:i+2]
            h=int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8],16)
            v[h % self.dim]+=1.0
            norm =sum(x * x for x in v ) **0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

def get_embeddings(kind: str) ->Embeddings:
    """kind: smoke | bge | openai"""
    if kind =="smoke":
        return HashEmbeddings(dim=256)

    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if kind == "bge":
        # 中文本地首选。BGE-M3 更强但很大(2GB+), CPU 首跑很慢。
        # 想快点先用小的: "BAAI/bge-small-zh-v1.5" (约 100MB)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},# ★ 归一化 → 内积≈余弦
        )
    raise ValueError(f"未知 embed 类型: {kind}")


# ===================== 工具 =====================
def clean_meta(m: dict) -> dict:
    """Chroma 只收 str/int/float/bool。list/dict/None 会被拒 —— 入库前洗一遍。"""
    return {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}

def index_exists() -> bool:
    """库落盘了吗 —— 判断依据是真文件, 不是'我记得我建过'。"""
    p=Path(PERSIST_DIR)
    return p.is_dir() and (p / "chroma.sqlite3").exists()

def count_of(vs:Chroma) -> int:
    try:
        return vs._collection.count()
    except Exception:
        try:
            return len(vs.get().get("ids",[]))
        except Exception:
            return -1

def load_chunks() -> list:
    """复用 Day1 的加载 + 分块, 并顺手过滤碎块。"""
    d1.ensure_sample_docs()                       # demo 模式会自动造样例文档
    docs_dir = HERE / "docs"
    docs = []
    for f in sorted(docs_dir.iterdir()):
        if f.suffix.lower() in (".txt", ".md"):
            # load_text_file accepts a Path and returns one Document.
            docs.append(d1.load_text_file(f))
    if not docs:
        raise SystemExit(f"docs 目录里没有 .txt/.md: {docs_dir}")
    return d1.split_documents(docs)

# ===================== 主流程 =====================
def build(embed:str,reset:bool):
    if reset and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        print(f"[reset] 已删除 {PERSIST_DIR}")

    emb=get_embeddings(embed)
    t0=time.time()

    #——分支 A:库已存在，直接加载，不重复嵌入——
    if index_exists() and not reset:
        vs=Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=PERSIST_DIR)
        n = count_of(vs)
        print(f"[加载] 检测到已有索引 → 直接加载, 不重复嵌入")
        print(f"       collection={COLLECTION}  条数={n}  耗时={time.time()-t0:.2f}s")
        return vs, 0, []

    # ---- 分支 B: 首次 → 嵌入 + 落盘 ----
    chunks=load_chunks()
    raw_n=len(chunks)
    kept=[c for c in chunks if len(c.page_content.strip()) >= MIN_CHARS]
    dropped = [len(c.page_content) for c in chunks if len(c.page_content.strip()) < MIN_CHARS]
    print(f"[分块] 共 {raw_n} 块  过滤碎块(<{MIN_CHARS}字) {raw_n - len(kept)} 块 "
          f"(长度={dropped})  入库 {len(kept)} 块")
    texts=[c.page_content for c in kept]
    metas =[clean_meta(dict(c.metadata)) for c in kept]


    # ★ hnsw:space 只在【创建集合】时生效。对已有集合改它 = 静默无效。
    vs=Chroma(
        collection_name=COLLECTION,
        embedding_function=emb,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    for i in range(0,len(texts),BATCH):
        vs.add_texts(texts=texts[i:i + BATCH], metadatas=metas[i:i + BATCH])
        print(f"      已嵌入 {min(i + BATCH, len(texts))}/{len(texts)}")


    if hasattr(vs, "persist"):                 # 老版本要手动; 新版自动
        try:
            vs.persist()
        except Exception:
            pass

    print(f"[建库] collection={COLLECTION}  条数={count_of(vs)}  耗时={time.time()-t0:.2f}s")
    print(f"       落盘目录: {PERSIST_DIR}")
    print(f"       文件: {[p.name for p in Path(PERSIST_DIR).iterdir()]}")
    return vs, len(kept), texts

# ===================== 自检(今天的重点) =====================
def self_check(vs:Chroma,n_added:int,texts:list[str],k:int=3) ->None:
    """四问: 落盘了吗 / 条数对吗 / 再加载还是这个数吗 / 真能搜出来吗"""
    print("\n" + "=" * 60)
    print("自检")
    print("=" * 60)
    # ① 落盘文件存在吗
    files = sorted(p.name for p in Path(PERSIST_DIR).iterdir())
    print(f"① 落盘文件      : {files}")
    assert "chroma.sqlite3" in files, "没落盘 → persist_directory 有问题"

    # ② 条数对不对(和"进了多少块"对齐)
    n = count_of(vs)
    print(f"② 库里条数      : {n}   (本次入 {n_added})")
    if n_added:
        assert n == n_added, f"条数不符: 库里 {n} vs 期望 {n_added}"

    # ③ 重新打开还是这个数吗(证明能复用, 不重复嵌入)
    vs2 = Chroma(collection_name=COLLECTION, embedding_function=vs.embeddings,
                 persist_directory=PERSIST_DIR)
    n2 = count_of(vs2)
    print(f"③ 重新打开条数  : {n2}   → {'一致 ✅' if n2 == n else '不一致 ❌'}")
    assert n2 == n, "重新加载后条数变了 → 落盘/加载有问题"

    # ④ 真能搜出来吗(拿一块自己的前 30 字去搜, 看它自己排第几)
    if texts:
        probe = texts[0][:30]
        hits = vs.similarity_search_with_score(probe, k=k)
        print(f"④ 探针查询      : {probe!r}")
        for rank, (doc, score) in enumerate(hits, 1):
            tag = "← 就是它" if doc.page_content[:30] == probe else ""
            print(f"    top{rank} score={score:.4f} {doc.page_content[:28]!r} {tag}")
        assert any(d.page_content[:30] == probe for d, _ in hits), \
            "自己的前 30 字都搜不回自己 → 检索链路有问题"
        print("    → 自己的块能搜回自己 ✅")

    print("\n注: Chroma 返回的是【距离】, 越小越像。设了 hnsw:space=cosine 就是余弦距离。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="bge", choices=["smoke", "bge", "openai"])
    ap.add_argument("--smoke", action="store_true", help="等价于 --embed smoke")
    ap.add_argument("--reset", action="store_true", help="删掉旧库重建")
    args = ap.parse_args()

    kind = "smoke" if args.smoke else args.embed
    print(f"嵌入后端: {kind}")
    vs, n_added, texts = build(embed=kind, reset=args.reset)
    self_check(vs, n_added, texts)


if __name__ == "__main__":
    main()
