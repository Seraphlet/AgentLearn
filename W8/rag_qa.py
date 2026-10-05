"""
W8-Day3  RAG 第 ④⑤ 段: 检索 + 生成(带引用 + 拒答)
================================================================
交付物: ① rag_qa.py  ② retrieve()  ③ 带引用的回答  ④ 拒答

运行:
    python w8\rag_qa.py                          # 交互式
    python w8\rag_qa.py -q "手册里XX是什么"        # 单问
    python w8\rag_qa.py --k-only -q "XX"          # ★ 只看检索, 不调模型(交付物②: 独立可测)
    python w8\rag_qa.py --topk 6 --min-sim 0.45
    set OLLAMA=1 && python w8\rag_qa.py           # 全本地(Ollama), 文档不出机器
"""
from __future__ import annotations

import argparse
from operator import imod
import os
import re
import sys
from pathlib import Path
from unittest.mock import DEFAULT

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from langchain_chroma import Chroma
import build_index as bi

#————————————————————————————参数
DEFAULT_K=4
DEFAULT_MIN_SIM = 0.35      # ★★ 这是【拍的】, 必须用 calibrate_threshold.py 换成实测值
GENERATION_BACKEND = "deepseek"   # deepseek | ollama

# ================== 段④ 检索(交付物②, 独立可测) ==================
def to_similarity(distance: float) -> float:
    """Chroma 返回的是【距离】, 越小越像。换算成【相似度】, 越大越像。

    ★ hnsw:space=cosine 时: cosine_distance = 1 - cos_sim
      所以 sim = 1 - distance, 范围大致 [0, 2], 正常落在 [0, 1]。
    ★ 如果建库时不是 cosine 空间, 这个换算不成立 —— 靠 self_test 里的
      "自己搜自己 distance≈0" 来验证方向没反。
    """
    return 1.0 - float(distance)

def retrieve(query: str,k:int=DEFAULT_K,vs: Chroma | None=None):
    """段④: query → [(Document, similarity)]

    ★ 这个函数【不碰 LLM】, 所以能脱离生成层单独测。
      文档要求"独立可测", 靠的就是这条边界。
    """
    raw= vs.similarity_search_with_score(query,k=k)
    return [(doc,to_similarity(dist)) for doc,dist in raw]

def filter_by_score(hits,min_sim:float):
    return [(d,s) for d,s in hits if s>=min_sim]

# ================== 段⑤ 生成(交付物③④) ==================
SYSTEM_PROMPT = """你是知识库问答助手。严格遵守以下规则:

1) 只能依据【资料】回答。不得使用你自己的知识, 不得推测、不得补充。
2) 每个事实后面必须标注来源编号, 格式如 [1] 或 [1][2]。
3) 如果【资料】不足以回答问题, 只回答这一句: 资料未提及。不要解释, 不要猜测。
4) 回答简洁直接, 不要复述这些规则, 不要出现"根据资料"这类套话以外的内容。
"""

def build_context(hits) -> str:
    """把 chunks 编号成 [1][2][3] —— 编号是引用溯源的载体"""
    parts =[]
    for i, (doc,sim) in enumerate(hits,1):
        src = Path(str(doc.metadata.get("source","?"))).name
        off= doc.metadata.get("start_index","_")
        parts.append(f"[{i}] 来源: {src} 偏移: {off} 相似度: {sim:.3f}\n{doc.page_content}")
    return "\n\n".join(parts)

def build_messages(query:str,hits):
    return [
        ("system",SYSTEM_PROMPT),
        ("human", f"【资料】\n{build_context(hits)}\n\n【问题】{query}"),
    ]

CITE_RE = re.compile(r"\[(\d+)\]")

def check_citations(text:str,n_hits:int) -> dict:
    """引用校验: 模型会编出不存在的编号(如只有 3 条却引用 [5])。

    ★ 只能查出【编号越界】, 查不出【编号张冠李戴】。
      后者只能人工看 —— 这是验证缺口, 不是已解决。
    """
    nums= [int(x) for x in CITE_RE.findall(text)]
    invalid= sorted({x for x in nums if x < 1 or x > n_hits})
    return {
        "cited":sorted(set(nums)),
        "invalid":invalid,
        "has_cite":bool(nums),
    }


def get_llm():
    if GENERATION_BACKEND == "ollama" or os.getenv("OLLAMA"):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                          temperature=0)
    from langchain_openai import ChatOpenAI
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "\n没找到 DEEPSEEK_API_KEY。\n"
            "  用云端:  set DEEPSEEK_API_KEY=sk-xxxx\n"
            "  用本地:  set OLLAMA=1  (需先装 Ollama 并 ollama pull qwen2.5:7b)\n"
            "  只看检索: 加 --k-only, 不调模型\n"
        )
    return ChatOpenAI(model="deepseek-v4-flash", api_key=key,
                      base_url="https://api.deepseek.com/v1",
                      temperature=0)      # ★ 0 提高确定性, 但 LLM 仍非完全确定

def answer(query: str, k: int =DEFAULT_K,min_sim: float = DEFAULT_MIN_SIM,vs: Chroma |None=None,llm =None) -> dict:
    """两道闸的完整问答。

    闸1(前置, 便宜): 最像的那块都没过阈值 → 直接拒答, 【不调模型】
    闸2(后置, 兜底): 让模型看资料够不够 → 它自己说"资料未提及"
    """
    raw = retrieve(query,k=k,vs=vs)
    best = raw[0][1] if raw else -1.0
    # ---- 闸1: 分数不过线 → 拒答, 一次 LLM 都不调 ----
    if not raw or best < min_sim:
        return{
            "status": "refused_by_score",
            "text": "资料未提及。",
            "best": best, "hits": raw, "cites": None,
            "note": f"最高相似度 {best:.3f} < 阈值 {min_sim}",
        }
    hits = filter_by_score(raw, min_sim)
    msgs = build_messages(query, hits)
    resp= llm.invoke(msgs)
    text=resp.content if hasattr(resp,'content') else str(resp)
    cites= check_citations(text,len(hits))

    refused = "资料未提及" in text
    return {
        "status": "refused_by_model" if refused else "ok",
        "text": text, "best": best, "hits": hits, "cites": cites,
        "note": (f"模型判定资料不足" if refused else
                 (f"无效引用 {cites['invalid']}" if cites["invalid"] else "")),
    }

# ================== 展示 ==================

def show_hits(hits,title="检查结果"):
    print(f"\n--- {title} ---")
    for i ,(doc,sim) in enumerate(hits,1):
        src = Path(str(doc.metadata.get("source","?"))).name
        snippet = doc.page_content.replace("\n","")[:60]
        print(f" [{i}] sim={sim:.3f} {src} {snippet} ...")

def show_result(r):
    show_hits(r["hits"])
    print(f"\n--- 回答 ---\n{r['text']}")
    print(f"\n状态: {r['status']}   最高相似度: {r['best']:.3f}"
          + (f"   {r['note']}" if r["note"] else ""))
    if r.get("cites"):
        c= r["cites"]
        print(f"引用编号: {c['cited'] if c['cited'] else '（没有标引用）'}")
        if c['invalid']:
            print(f" 无效引用（资料里没有这些编号）: {c['invalid']}  ← 模型编的")

def load_vs(kind: str) -> Chroma:
    info = bi.read_info()
    if not info:
        raise SystemExit("没找到索引信息文件 → 先跑 python w8\\build_index.py --reset")
    bi.guard_backend(kind)          # ★ 模型对不上就不让启动(Day2 的防线, 今天继续用)
    emb = bi.get_embeddings(kind)
    vs = Chroma(collection_name=bi.COLLECTION,
                embedding_function=emb,
                persist_directory=bi.PERSIST_DIR)
    print(f"[加载] 模型={info.get('model_name')} dim={info.get('dim')} "
          f"条数={info.get('count')}")
    return vs

# ================== 自检 ==================
def self_test(vs: Chroma):
    """检索层自检 —— 不花钱, 每次启动都跑"""
    print("\n" + "=" * 60 + "\n检索层自检\n" + "=" * 60)
    snap = vs.get(limit=1)
    if not snap["documents"]:
        raise SystemExit("库是空的")
    text = snap["documents"][0]
    probe = text[:30]

    hits = retrieve(probe, k=3, vs=vs)
    for i, (d, s) in enumerate(hits, 1):
        hit = " ← 就是它" if d.page_content[:30] == probe else ""
        print(f"  top{i} sim={s:.4f}{hit}")

    assert hits[0][1] > 0.95, (
        f"自己搜自己相似度只有 {hits[0][1]:.3f} (期望≈1.0)\n"
        "  → 要么 hnsw:space 不是 cosine, 要么 to_similarity 方向反了"
    )
    print("  → 自己搜自己 ≈ 1.0 ✅  (方向没反 + cosine 空间成立)")

# ================== 主流程 ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="bge-small")
    ap.add_argument("--topk", type=int, default=DEFAULT_K)
    ap.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    ap.add_argument("-q", "--question")
    ap.add_argument("--k-only", action="store_true",
                    help="只跑检索不调模型(= 交付物②的独立可测接口)")
    args = ap.parse_args()

    vs = load_vs(args.embed)
    self_test(vs)

    k, min_sim = args.topk, args.min_sim
    llm = None if args.k_only else get_llm()

    def once(q):
        if args.k_only:
            hits = retrieve(q, k=k, vs=vs)
            show_hits(hits, f"检索结果 (top-{k}, 不调模型)")
            ok = [(d, s) for d, s in hits if s >= min_sim]
            print(f"\n超过阈值 {min_sim} 的: {len(ok)}/{len(hits)} 块")
            print("→ 若过线为 0, 正式流程会走【拒答】")
            return
        show_result(answer(q, k=k, min_sim=min_sim, vs=vs, llm=llm))

    if args.question:
        once(args.question)
        return

    print("\n输入问题; :k N 改 topk, :sim X 改阈值, :q 退出\n")
    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", "quit", "exit"):
            break
        if q.startswith(":k "):
            k = int(q[3:]); print(f"topk = {k}"); continue
        if q.startswith(":sim "):
            min_sim = float(q[5:]); print(f"min_sim = {min_sim}"); continue
        once(q)

        
if __name__ == "__main__":
    main()
