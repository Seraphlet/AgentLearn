"""
W8-Day5 参数实验: chunk_size / chunk_overlap 对照实验
================================================================
把 pipeline 上【拍的】数字换成【算的】数字。

三条设计决定(今天最该记住的):
  ① gold 用【原文必经字符串】, 不用 chunk id —— 否则没法跨配置比较
  ② 必须同时报上下文成本 —— 固定 k 时大 chunk 天然拿到更多上下文
  ③ 每档配置用独立目录, 不碰主索引 w8\chroma_db

运行:
    python w8\exp_chunk_size.py                          # 必做: 256/512/1024, ovl=15%
    python w8\exp_chunk_size.py --overlap-ratio 0 0.15 0.3   # 二维网格
    python w8\exp_chunk_size.py --fair-budget            # 固定上下文预算(公平)
    python w8\exp_chunk_size.py --suggest                # 找 gold 锚点候选
    python w8\exp_chunk_size.py --detail q03             # 某条 query 各档排第几
"""

