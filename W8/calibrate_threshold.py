"""
把【拍】的阈值换成【算】的。
做法: 拿"该答的"和"该拒的"各若干问题, 各跑一遍检索, 看两组分数的分布,
      找一个能分开两堆的切点。
运行: python w8\calibrate_threshold.py
"""
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from langchain_chroma import Chroma
import build_index as bi
import rag_qa as rq

# ★★ 请照着 w8\docs\ 里的文档, 手抄 4~5 个"文档里能答出来"的问题
GOOD_Q = [
    "把这里换成手册里真有的问题1",
    "手册里真有的问题2",
    "手册里真有的问题3",
    "手册里真有的问题4",
]

# 这些是"文档肯定答不出"的 —— 拒答的对照组
BAD_Q = [
    "红烧肉怎么炖才软烂",
    "今天北京天气怎么样",
    "Python 的 GIL 是什么",
    "如何申请信用卡",
    "周杰伦最新专辑叫什么",
]


def main():
    vs = Chroma(collection_name=bi.COLLECTION,
                embedding_function=bi.get_embeddings("bge-small"),
                persist_directory=bi.PERSIST_DIR)

    def best_scores(qs):
        out = []
        for q in qs:
            top = rq.retrieve(q, k=4, vs=vs)
            out.append(top[0][1] if top else -1.0)
        return out

    g = best_scores(GOOD_Q)
    b = best_scores(BAD_Q)

    print("【该答的】问题 → top1 相似度")
    for q, s in zip(GOOD_Q, g):
        print(f"  {s:.4f}  {q}")
    print("\n【该拒的】问题 → top1 相似度")
    for q, s in zip(BAD_Q, b):
        print(f"  {s:.4f}  {q}")

    print(f"\n该答组: min={min(g):.4f}  median={st.median(g):.4f}  max={max(g):.4f}")
    print(f"该拒组: min={min(b):.4f}  median={st.median(b):.4f}  max={max(b):.4f}")

    gap_lo, gap_hi = max(b), min(g)
    print(f"\n该拒组的最高分 = {gap_lo:.4f}")
    print(f"该答组的最低分 = {gap_hi:.4f}")

    if gap_lo < gap_hi:
        mid = (gap_lo + gap_hi) / 2
        print(f"\n✅ 两组可分。切点取中位 → min_sim ≈ {mid:.3f}")
        print(f"   理由: 该拒组全在 {gap_lo:.3f} 以下, 该答组全在 {gap_hi:.3f} 以上。")
        print(f"   ★ 把 rag_qa.py 的 DEFAULT_MIN_SIM 改成 {mid:.2f}")
    else:
        print(f"\n❌ 两组【不可分】: 该拒组最高分 {gap_lo:.3f} 反而 >= "
              f"该答组最低分 {gap_hi:.3f}")
        print("   含义: 单靠绝对阈值分不开。可能原因:")
        print("     · 该答组里有问题文档里其实答不了(数据错)")
        print("     · 该拒组里有问题和文档内容意外沾边(选了不合适的对照)")
        print("     · 阈值这一招不够 → 需要 rerank 或混合检索(W8 后段)")
        print("   ★ 别硬凑一个数字让它'看起来能分' —— 那是自欺。")


if __name__ == "__main__":
    main()