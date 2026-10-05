"""
W8-Day4  混合检索: Dense(向量) + Sparse(BM25) + RRF 融合
================================================================
三条硬规则(今天的重点, 也是三条防线):
  1) 两路必须【同源】  : 语料一律从 Chroma 拉全量, BM25 用同一批 id 建索引。
                          否则 RRF 拿两份不同的 id 空间融合 = 融合出幻觉。
  2) 分词器必须是同一函数: query 和 doc 用同一个 tokenizer。切法不一致 → token 对不上 → BM25 静默失效。
  3) RRF 只用【名次】   : 所以接口收的是 id 的有序列表, 不是 (id, score)。

运行:
    python w8\hybrid.py --explain -q "ERR_4517 报错怎么办"     # ★ 看两路各自排第几
    python w8\hybrid.py --mode dense  -q "..."
    python w8\hybrid.py --mode sparse -q "..." --tok bigram
    python w8\hybrid.py --mode hybrid -q "..."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from langchain_chroma import Chroma
import build_index as bi
import rag_qa as rq     

RRF_K=60  # RRF 常数(原论文常用值)
N_PER_ROUTE = 20               # ★ 每路候选数, 不是最终 k
DEFAULT_K = 5                  # 融合后交给下游的条数

# ============ 分词器: 必须对 query 和 doc 用同一个 ============
_IDENT=re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*\d[A-Za-z0-9_\-\.]*|[A-Za-z0-9_\-\.]{3,}")

def tok_jieba(text:str) -> list[str]:
    """词级分词。

    ★ 那句正则保护是【必要的防呆】, 不是顺手写的:
      不加它, jieba 会把 "ZQ-7781" 切成 ['ZQ', '-', '7781'] ——
      把一个 IDF 极高的强 token 拆成三个弱 token, 定位能力被稀释。
      想验证这一点: 把 _IDENT.findall 那行注释掉, 重跑实验对比。
    """
    import jieba
    parts=_IDENT.split(text)
    keep=[m.group(0).lower() for m in _IDENT.finditer(text)]
    toks:list[str]=[]
    for p in parts:
        toks+=[t for t in jieba.lcut(p) if t.strip()]
    return toks + keep

def tok_bigram(text:str) -> list[str]:
    """字符 bigram: 中文稀疏检索的另一种常见做法。

    对未登录词(专名/新型号)更Robustness —— 因为不依赖词典。
    代价: 维度膨胀(中文约 2 万+), 且单字信息被稀释。
    """
    s=re.sub(r"\s+","",text)
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else ([s] if s else [])

TOKENIZERS = {"jieba": tok_jieba, "bigram": tok_bigram}

# ============ Sparse: BM25 ============
class SparseIndex:
    """BM25 倒排索引。k1/b 是 BM25 的两个超参(⚠️ 按 rank_bm25 默认值)。
    """
    def __init__(self,ids,texts,tokenizer,k1:float=1.5,b:float=0.75):
        from rank_bm25 import BM250kapi
        self.ids=list(ids)
        self.tokenizer=tokenizer
        self.tokens=[tokenizer(t) for t in texts]
        self.bm25=BM250kapi(self.tokens,k1=k1,b=b)
        self.k1,self.b=k1,b

    def search(self,query:str,k:int):
        """返回 [(id, bm25_score)] 降序。★ 分数无上界, 别当相似度用。"""
        scores=self.bm25.get_scores(self.tokenizer(query))
        order=sorted(range(len(scores)),key=lambda i:-scores[i])[:k]
        return [(self.ids[i],float(scores[i])) for i in order]

def explain_token(self,query:str):
    """演示用: 打印 query 切成了什么 + 每个 token 的 IDF。

        这一段是【算的】不是【拍的】—— 你自己看数字就知道 jieba 有没有切碎标识符。
    """
    import math
    df = defaultdict(int)
    for toks in self.tokens:
        for t in set(toks):
            df[t]+=1
    N=len(self.tokens)
    print(f"\n [分词诊断] tokenizer={self.tokenizer.__name__} 语料={N} 条")
    print(f"  {'token':<16}{'df':>6}{'idf':>8}")
    for t in self.tokenizer(query):
        d=df.get(t,0)
        idf= math.log((N-d+0.5)/(d+0.5)+1)
        print(f"  {t!r:<16}{d:>6}{idf:>8.3f}")

# ============ RRF 融合(只吃名次, 不吃分数) ============
def rrf_fuse(ranked_idlists,k:int=RRF_K,weights=None,top_n=None):
    """输入: [[id_rank1, id_rank2, ...], ...]   ← 注意里面【没有 score】
    输出: [(id, rrf_score)] 降序

    故意把签名设计成"只收 id 列表": 这样你不可能不小心把 BM25 的分数混进来。
      接口形状本身就是一道防线。
    """
    score=defaultdict(float)
    for i,ids in enumerate(ranked_idlists):
        v=1.0 if weights is None else weights[i]
        for rank,did in enumerate(ids,1):
            score[did]+=v/(k+rank)
    order=sorted(score.items(),key=lambda x:(-x[1],str(x[0])))
    return order [:top_n] if top_n else order
# ============ 整合 ============
class HybridRetriever:
    def __init__(self,kind:str="bge-small",tok:str="jieba",n_per_route:int=N_PER_ROUTE,rrf_k:int=RRF_K):
        info =bi.read_info()
        if not info:
            raise SystemExit("没找到索引信息 → 先跑 python w8\\build_index.py --reset")
        bi.guard_backend(kind)
        self.emb=bi.get_embeddings(kind)
        self.vs =Chroma(collection_name=bi.COLLECTION,embedding_function=self.emb,persist_directory=bi.PERSIST_DIR)
        # ★ 规则1: 从 Chroma 拉全量 —— 两路共享同一批 id
        data=self.vs.get(include=["documents","metadatas"])
        self.ids=data["ids"]
        self.docs=data["documents"]
        self.metas=data["metadatas"]
        self.id2content=dict(zip(self.ids,self.docs))
        self.content2id={d:i for i,d in zip(self.ids,self.docs)}
        self.n_per_route,self.rrf,self.tokname=n_per_route,rrf_k,tok
        self.sparse=SparseIndex(self.ids,self.docs,TOKENIZERS[tok])
        print(f"[加载] {info.get('model_name')} dim={info.get('dim')} "
              f"条数={len(self.ids)}  分词器={tok}  RRF k={rrf_k}")
    # ---- 路1: Dense ----
    def dense(self,query:str,k:int):
        """用 Day3 同一个 to_similarity 换算方向, 保证"越大越像"一致。"""
        out=[]
        for doc,dist in self.vs.similarity_search_with_score(query,k=k):
            did=getattr(doc,"id",None)
            if did is None:
                did = self.content2id.get(doc.page_content)
            if did is None:
                raise RuntimeError(
                    "dense 结果拿不到 id → 无法和 BM25 对齐。\n"
                    "  你的 langchain_chroma 版本可能没给 Document 附 id。\n"
                    "  先把内容反查表补上, 或升级 langchain-chroma。"
                )
            out.append((did,rq.to_similarity(dist)))
        return out
    # ---- 路2: Sparse ----
    def sparse_search(self, query: str, k: int):
        return self.sparse.search(query, k)
    # ---- 融合 ----
    def hybrid(self,query:str,k:int,weights=None):
        d=self.dense(query,self.n_per_route)
        s=self.sparse_search(query,self.n_per_route)
        fused = rrf_fuse([[i for i, _ in d], [i for i, _ in s]],
                         k=self.rrf_k, weights=weights, top_n=k)
        return fused, d, s
    def preview(self, did, n=50):
        return self.id2content[did].replace("\n", " ")[:n]

def show(route_name, rows, preview, k=5):
    print(f"\n  [{route_name}]")
    for rank, (did, sc) in enumerate(rows[:k], 1):
        print(f"    {rank}. score={sc:>8.4f}  {preview(did)}...")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="bge-small")
    ap.add_argument("--tok", default="jieba", choices=list(TOKENIZERS))
    ap.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--n", type=int, default=N_PER_ROUTE, help="每路候选数")
    ap.add_argument("--explain", action="store_true", help="打印两路各自的名次 + 分词诊断")
    ap.add_argument("--diag", action="store_true", help="打印 query 的 token + IDF")
    args = ap.parse_args()

    hr = HybridRetriever(kind=args.embed, tok=args.tok, n_per_route=args.n)
    q = args.question
    print(f"\nquery = {q!r}")

    if args.diag:
        hr.sparse.explain_token(q)

    if args.mode == "dense":
        d = hr.dense(q, args.k)
        show(f"dense (相似度, 越大越像)", d, hr.preview)
        return
    if args.mode == "sparse":
        s = hr.sparse_search(q, args.k)
        show(f"sparse BM25 (分数无上界)", s, hr.preview)
        return

    fused, d, s = hr.hybrid(q, args.k)
    if args.explain:
        # ★ 这两行是今天的核心证据: 看"对的块"在两路各排第几
        dre = {i: r for r, (i, _) in enumerate(d, 1)}
        sre = {i: r for r, (i, _) in enumerate(s, 1)}
        print(f"\n  --- 两路名次对照 (前 {min(args.k, len(fused))} 条融合结果) ---")
        print(f"  {'rank':<6}{'dense':>8}{'sparse':>8}   RRFF分  内容")
        for r, (did, fs) in enumerate(fused[:args.k], 1):
            dn = dre.get(did, "—")
            sn = sre.get(did, "—")
            print(f"  {r:<6}{str(dn):>8}{str(sn):>8}   {fs:.4f}  {hr.preview(did, 34)}")
        show("dense 原始 top", d, hr.preview, args.k)
        show("sparse 原始 top", s, hr.preview, args.k)
    show(f"★ 融合结果 (RRF, k={hr.rrf_k})", fused, hr.preview, args.k)
    print(f"\n注: RRF 分数上限 ≈ 1/({hr.rrf_k}+1) = {1/(hr.rrf_k+1):.4f}")
    print(f"    → 这个数【不能】拿去和 Day3 的 min_sim 比, 见文档警告。")


if __name__ == "__main__":
    main()    

    

