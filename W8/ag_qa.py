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