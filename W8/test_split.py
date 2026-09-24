"""W8-Day1 验收: 加载 + 分块的 12 条断言
跑法(在 AgentLearn 根目录):
    python w8\test_split.py
全部离线, 不调 LLM, 不花钱。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from langchain_core.documents import Document

import day1_load_split as d1

T = []
def case(fn):
    T.append(fn)
    return fn


CN_TEXT = "".join(f"这是第{i}句关于检索增强生成的中文说明。" for i in range(1, 41))
ENDS = ("。", "！", "？", "；", "，")


# ───────────────────────── 加载段 ─────────────────────────
@case
def t_load_text_file_keeps_metadata():
    """验收: 加载出来必须带 source [避坑表]"""
    p = HERE / "docs" / "样例手册.md"
    d1.ensure_sample_docs()
    doc = d1.load_text_file(p)
    assert isinstance(doc, Document)
    assert doc.metadata["source"] == "样例手册.md", doc.metadata
    assert len(doc.page_content) > 100, "读进来是空的"


@case
def t_load_utf8_no_mojibake():
    """验收: 中文不乱码(Windows 的经典坑)"""
    d1.ensure_sample_docs()
    doc = d1.load_text_file(HERE / "docs" / "样例手册.md")
    assert "记忆系统" in doc.page_content, f"中文没读对: {doc.page_content[:80]}"
    assert "\ufffd" not in doc.page_content, "出现替换字符 = 编码错了"


@case
def t_scan_detector_finds_blank_pages():
    """验收: 扫描件探测能把'没文字层的页'找出来 [避坑表]"""
    pages = [
        Document(page_content="有字" * 30, metadata={"page": 0}),
        Document(page_content="   ", metadata={"page": 1}),
        Document(page_content="", metadata={"page": 2}),
        Document(page_content="有字" * 30, metadata={"page": 3}),
    ]
    assert d1.check_text_layer(pages) == [1, 2]
    assert d1.check_text_layer(pages[:1]) == [], "正常页不该被误报"


# ───────────────────────── 分块段 ─────────────────────────
@case
def t_short_text_not_split():
    """不超预算就不该切"""
    docs = d1.split_one_text("很短的一句话。", size=300, overlap=50)
    assert len(docs) == 1, f"短文本被切了 {len(docs)} 块"


@case
def t_long_text_split_under_budget():
    """验收: 长文本被切开, 且不超 size(有分隔符的前提下)"""
    docs = d1.split_one_text(CN_TEXT, size=200, overlap=0)
    assert len(docs) > 1, "长文本没被切"
    over = [len(d.page_content) for d in docs if len(d.page_content) > 200]
    assert not over, f"有块超预算: {over}"


@case
def t_chinese_separators_actually_matter():
    """★★ 对照实验: 加中文标点 vs 不加 —— 唯一变量就是 separators"""
    good = d1.split_one_text(CN_TEXT, size=200, overlap=0)
    bad = d1.split_one_text(CN_TEXT, size=200, overlap=0,
                            separators=["\n\n", "\n", " ", ""])

    g = sum(1 for d in good if d.page_content.endswith(ENDS))
    b = sum(1 for d in bad if d.page_content.endswith(ENDS))
    print(f"      加中文标点: {g}/{len(good)} 块以标点收尾 "
          f"({g / len(good):.0%})")
    print(f"      不加     : {b}/{len(bad)} 块以标点收尾 ({b / len(bad):.0%})")

    assert g / len(good) >= 0.8, f"加了中文标点还是切得碎: {g}/{len(good)}"
    assert b / len(bad) <= 0.3, "对照组竟然也切在标点上, 这条对照没牙"


@case
def t_no_separator_fallback_overflows():
    """★ 真实的坑: separators 里不留 "" 兜底, 遇到无分隔符内容会整块超预算"""
    text = "A" * 800      # 800 个字符, 中间一个分隔符都没有
    no_fallback = ["\n\n", "\n", "。", "，"]          # 故意不留 ""

    a = d1.split_one_text(text, size=100, overlap=0, separators=no_fallback)
    b = d1.split_one_text(text, size=100, overlap=0, separators=no_fallback + [""])

    la = [len(d.page_content) for d in a]
    lb = [len(d.page_content) for d in b]
    print(f"      没留 ''.join兜底: {len(a)} 块, 最长 {max(la)}")
    print(f"      留了兜底   : {len(b)} 块, 最长 {max(lb)}")

    assert max(la) > 100, "构造的用例没触发坑 —— 这条测试就白测了"
    assert max(lb) <= 100, "留了兜底还超预算"


@case
def t_overlap_ge_size_raises():
    """避坑表: overlap >= chunk_size -> 报错或无限切 [避坑表]"""
    try:
        d1.make_splitter(size=100, overlap=150)
        raise AssertionError("overlap > size 竟然没报错")
    except AssertionError:
        raise
    except Exception as e:
        print(f"      overlap>size  -> {type(e).__name__}: {str(e)[:60]}")

    # == 这个边界我不确定, 所以只报告不断言
    try:
        d1.make_splitter(size=100, overlap=100)
        print("      overlap==size -> 没报错(本版本只拦 >, 不拦 ==)")
    except Exception as e:
        print(f"      overlap==size -> {type(e).__name__}: {str(e)[:60]}")


@case
def t_overlap_is_real_and_expansion_measurable():
    """★ 算账: overlap 是真的吗? 膨胀多少?"""
    docs = d1.split_one_text(CN_TEXT, size=200, overlap=30)
    chunks = [d.page_content for d in docs]
    ov = d1.measure_actual_overlap(chunks)
    exp = d1.expansion_ratio(len(CN_TEXT), chunks)
    pred = d1.formula_expansion(200, 30)

    print(f"      设定 overlap=30  实测相邻重叠={ov}")
    print(f"      膨胀率: 实测 {exp:.1%}  公式预测 {pred:.1%}")

    assert ov and min(ov) > 0, "重叠根本没生效"
    assert exp > 0, "膨胀率为 0 = 没有重叠"
    assert exp < 1.5, f"膨胀太夸张了: {exp:.1%}"


@case
def t_metadata_follows_every_chunk():
    """★★ 验收: metadata 必须跟着每一块走(第⑥段引用全靠它) [避坑表]"""
    src = Document(page_content=CN_TEXT, metadata={"source": "手册.md", "page": 3})
    docs = d1.split_documents([src], size=200, overlap=20)
    assert len(docs) > 1
    for d in docs:
        assert d.metadata.get("source") == "手册.md", f"丢了 source: {d.metadata}"
        assert d.metadata.get("page") == 3, f"丢了 page: {d.metadata}"


@case
def t_start_index_monotonic():
    """add_start_index: 每块能说出自己从原文第几个字符开始(有就验, 没有就报告)"""
    docs = d1.split_one_text(CN_TEXT, size=200, overlap=20)
    idx = [d.metadata.get("start_index") for d in docs]
    if all(i is None for i in idx):
        print("      [报告] 本版本没写 start_index —— 属版本差异, 不影响验收")
        return
    assert all(i is not None for i in idx), f"部分块有 start_index, 部分没有: {idx}"
    assert idx == sorted(idx), f"start_index 不是递增的: {idx}"
    print(f"      start_index = {idx}")


@case
def t_markdown_headers_become_metadata():
    """★ 验收: Markdown 一级切 -> 章节标题进 metadata"""
    secs = d1.split_by_headers(d1.SAMPLE_MD)
    assert len(secs) >= 4, f"只切出 {len(secs)} 个章节, 太少了"
    hs = [s.metadata.get("h2") for s in secs if s.metadata.get("h2")]
    assert "第一章 记忆系统" in hs, f"h2 没进 metadata: {hs}"
    assert all(s.page_content.strip() for s in secs), "切出了空章节"


@case
def t_two_stage_keeps_headers():
    """★★ 二级切完, 一级的 metadata 必须还在 —— 这是两级切的全部价值"""
    final = d1.split_markdown_two_stage(d1.SAMPLE_MD, size=200, overlap=0)
    assert len(final) >= 4
    with_h = [d for d in final if d.metadata.get("h2")]
    assert with_h, f"二级切完 h2 全丢了: {[d.metadata for d in final]}"
    sources = {d.metadata.get("h1") for d in final}
    print(f"      块数 {len(final)}, 带 h2 的 {len(with_h)}, h1 取值 {sources}")


@case
def t_empty_and_whitespace_dont_crash():
    """验收: 空文档不炸"""
    a = d1.split_one_text("", size=100, overlap=0)
    b = d1.split_one_text("   \n  ", size=100, overlap=0)
    print(f"      空串 -> {len(a)} 块;  纯空白 -> {len(b)} 块")
    assert isinstance(a, list) and isinstance(b, list)


# ───────────────────────── runner ─────────────────────────
def main() -> int:
    print(f"跑 {len(T)} 个用例  (Python {sys.version.split()[0]})\n")
    ok, fails = 0, []
    for fn in T:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}")
            print(f"       {type(e).__name__}: {e}")
            fails.append(fn.__name__)
    print(f"\n  {ok}/{len(T)} 通过" + (f"   失败: {fails}" if fails else ""))
    return 0 if ok == len(T) else 1


if __name__ == "__main__":
    sys.exit(main())