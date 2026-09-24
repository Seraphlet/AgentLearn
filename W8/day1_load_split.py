"""W8-Day1 · RAG 第①②段: 文档加载 + 分块

跑法(Windows PowerShell, 在 AgentLearn 根目录):
    python w8\day1_load_split.py              # demo: 自动造样例文档, 0 准备, 一定能跑
    python w8\day1_load_split.py real         # real: 读 w8\docs\ 里的真实文件
    python w8\day1_load_split.py compare      # 对比不同 chunk_size 的代价

★ 两个我故意加的开关:
    demo   —— 不依赖你手上有任何文档, 先证明代码是活的
    compare —— 把 chunk_size 从"拍的"变成"算的"
"""
from __future__ import annotations

from importlib import metadata
from pydoc import Doc
import re
import sys
from pathlib import Path

from chromadb import Metadatas
# ── 兼容两代 langchain 的 import 路径, 不靠猜 ──────────────────────

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )
except ImportError:  # 老版本
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )

try: 
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ══════════════════ 0. 配置(★ 这两个数今天是"拍的", compare 模式会帮你还账) ═════
HERE =Path(__file__).resolve().parent
DOCS_DIR =HERE/"docs"
CHUNK_SIZE =300  # ★ 上限, 不是固定值
CHUNK_OVERLAP=50   # ★ 约 size 的 15% [避坑表: overlap 取 size 的 10-15%]
OVER_RATIO=0.15

# ★ 中文必须加中文标点, 否则会被按字符硬切 [避坑表]
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# ══════════════════ 1. 加载段 ══════════════════
def load_text_file(path:Path) ->Document:
    """读文本/Markdown。
    ★ 显式 encoding="utf-8": Windows 下默认 GBK, 中文会乱码 [避坑表]
    """
    text=path.read_text(encoding="utf-8")
    return Document(page_content=text,metadata={"source":path.name})

def load_text_file_official(path:Path) -> Document:
    """同样的功能, 用 LangChain 官方 loader —— 对比用, 两版都行。
    需要: pip install langchain-community
    """
    from langchain_community.document_loaders import TextLoader
    docs = TextLoader(str(path),encoding="utf-8").load()
    return docs[0]

def load_pdf_file(path:Path) ->list[Document]:
    """PDF: ★ 一页 = 一个 Document, metadata 里带 page。
    需要: pip install pypdf langchain-community
    """
    try: 
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        raise SystemExit("缺依赖：pip install pypdf langchain-community") from e
    return PyPDFLoader(str(path)).load()

def check_text_layer(pages:list[Document],min_chars:int=20) ->list[int]:
    """★ 扫描件探测 —— 这是"验证"不是"假设" [避坑表: PDF 是扫描件 -> 加载出来是空的]
    判据: 返回"疑似没有文字层"的页码列表(空列表 = 正常)
    """
    return [i for i, d in enumerate(pages)
            if len(d.page_content.strip()) < min_chars]

# ══════════════════ 2. 分块段 ══════════════════
def make_splitter(size:int=CHUNK_SIZE,overlap: int=CHUNK_OVERLAP,separators: list[str] | None=None):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators if separators is not None else SEPARATORS,
        keep_separator=True,        # 标点留在块里, 读起来才不是断句
        length_function=len,        # ★ 数"字符"不是数 token(下面有说明)
        add_start_index=True,       # ★ 记下这块在原文档的位置 -> 第⑥段引用要用2
    )

def split_documents(docs: list[Document],*,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP,separators=None)-> list[Document]:
    return make_splitter(size,overlap,separators).split_documents(docs)

def split_one_text(text:str,*,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP,metadata:dict | None=None,separators=None) ->list[Document]:
    return make_splitter(size,overlap,separators).create_documents([text],metadatas=[metadata or {}])

def split_by_headers(md_text:str) -> list[Document]:
    """Markdown 一级切: 按标题切成"章节"。
    ★ 价值: h1/h2/h3 会进 metadata —— 检索出来的块自带"它在哪一章"
    """
    md_sp=MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,   # 标题留在正文里, 块自带上下文  
    )
    return md_sp.split_text(md_text)

def split_markdown_two_stage(md_text:str,*,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP) ->list[Document]:
    """★ 真实项目的推荐做法: 两级切
      一级: 按标题切 -> 保证"语义边界"和"章节归属"
      二级: 超长的章节再用递归按长度切 -> 保证不超预算
      关键: 二级切完, 一级的 metadata(h1/h2/h3) 必须还在
    """
    sections=split_by_headers(md_text)
    return split_documents(sections,size=size,overlap=overlap)

# ══════════════════ 3. 度量(今天最值钱的部分: 把拍脑袋变成算的) ═════════
def measure_actual_overlap(chunks:list[str]) -> list[int]:
    """实测相邻块的真实重叠字符数。
    ★★ 重要认知: 你设的 chunk_overlap 是【下限】不是精确值 ——
       切分点会被"吸附"到分隔符上, 所以实际重叠往往 >= 设定值。
    """
    out=[]
    for a,b in zip(chunks,chunks[1:]):
        n=0
        for k in range(min(len(a),len(b)),0,-1):
            if a[-k:]==b[:k]:
                n=k
                break
        out.append(n)
    return out

def expansion_ratio(original_len:int,chunks:list[str])->float:
    """冗余膨胀率 = (所有块字符数之和 / 原文字符数) - 1
    ★ 这个数直接乘到 Day2 的 embedding 账单上
    ★ 公式近似: overlap / (chunk_size - overlap)
    """
    if not original_len:
        return 0.0
    return sum(len(c) for c in chunks)/original_len -1

def formula_expansion(size:int,overlap:int) -> float:
    """理论膨胀率 = overlap / (size - overlap)"""
    if size <=overlap:
        return float("inf")
    return overlap / (size-overlap)

# ══════════════════ 4. 报告 ══════════════════
def report_split(label:str,original:str,docs:list[Document]) ->None:
    lens =[len(d.page_content) for d in docs]
    chunks = [d.page_content for d in docs]
    print(f"\n=== {label} ===")
    if not docs:
        print("  0块(原文为空？)")
        return
    print(f"  原文 {len(original)} 字符 -> {len(docs)} 块")
    print(f"  块长: 最短 {min(lens)}  最长 {max(lens)}  平均 {sum(lens) // len(lens)}")
    print(f"  size 上限 {CHUNK_SIZE}  ->  超预算的块: "
          f"{[l for l in lens if l > CHUNK_SIZE] or '无'}")
    ov= measure_actual_overlap(chunks)
    if ov:
        print(f"实测相邻重叠：{ov}")
        print(f"  (设定值 {CHUNK_OVERLAP} -> 实测最小 {min(ov)} —— 设定是下限)")
    exp=expansion_ratio(len(original),chunks)
    print(f"  膨胀率 实测 {exp:.1%}   公式预测 {formula_expansion(CHUNK_SIZE, CHUNK_OVERLAP):.1%}")

    if docs[0].metadata:
        print(f"  metadata[0] = {docs[0].metadata}")

    print("  --- 前 2 块预览 ---")
    for i, d in enumerate(docs[:2]):
        head = d.page_content[:60].replace("\n", " ")
        print(f"  #{i} ({len(d.page_content)}字) {head}...")

def compare_sizes(text:str,size=(100,300,800)) ->None:
    """★ 把 chunk_size 从"拍的"变成"算的"
    你会看到: 块越小 -> 块数越多 -> 嵌入成本越高(Day2 会算这笔账)
    """
    print(f"\n=== chunk_size 选型对照(原文 {len(text)} 字符) ===")
    print(f"  {'size':>6} {'overlap':>8} {'块数':>6} {'平均块长':>9} {'膨胀率':>8} {'预计嵌入字符数':>15}")
    for s in size:
        ov =int(s * OVER_RATIO)
        docs=split_one_text(text,size=s,overlap=ov)
        lens=[len(d.page_content) for d in docs]
        total = sum(lens)
        print(f"  {s:>6} {ov:>8} {len(docs):>6} {total // max(len(docs), 1):>9} "
              f"{expansion_ratio(len(text), [d.page_content for d in docs]):>7.1%} "
              f"{total:>15}")
    print("  ★ 最后一列 = Day2 要付钱的那一列。块越小, 这一列涨得越快。")


# ══════════════════ 5. 样例文档(demo 模式自己造) ══════════════════
SAMPLE_MD = """# Agent 学习手册

## 第一章 记忆系统

Agent 若只依赖当前对话上下文，就像每次重启的助手：用户偏好、历史决策、任务进展统统遗忘。
记忆系统提供跨会话状态，让 Agent 能积累事实、经验与技能。

### 1.1 短期记忆

短期工作记忆保存当前对话、任务状态和中间结果。它保证多轮对话的连贯性。
上下文窗口一满，短期记忆就会失效，所以需要压缩或摘要。

### 1.2 长期记忆

长期记忆沉淀用户偏好、事实、技能与教训。它支持跨会话复用。
长期记忆带来存储、检索和更新开销，也可能写入错误、过时或敏感信息。

## 第二章 检索增强生成

RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写。
它让模型在回答前先检索外部知识库，从而减少幻觉、支持私有数据。

### 2.1 为什么要切块

整本书塞不进上下文窗口，而且检索精度会崩。所以要把文档切成 chunk。
切得太碎会丢上下文，切得太大召回噪声多。

### 2.2 重叠的作用

块与块之间保留一部分重叠，防止答案正好跨在边界上被切断。
重叠的代价是重复存储和更高的嵌入成本。
"""

SAMPLE_TXT="。".join(
    [f"这是第{i}句关于检索增强生成的中文说明，用来演示中文分块效果"
     for i in range(1, 41)]) + "。"

def ensure_sample_docs() -> list[Path]:
    """demo 模式顺手把样例写到 docs/, 这样 real 模式也有东西可读(0 准备)"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = DOCS_DIR / "样例手册.md"
    p2 = DOCS_DIR / "中文长文.txt"
    if not p1.exists():
        p1.write_text(SAMPLE_MD, encoding="utf-8")
    if not p2.exists():
        p2.write_text(SAMPLE_TXT, encoding="utf-8")
    return [p1, p2]
# ══════════════════ 6. 三种模式 ══════════════════

def run_demo() -> None:
    print("=" * 70)
    print(f"[demo] chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}  "
          f"separators={SEPARATORS}")
    print("=" * 70)

    # ── ① 加载 ──
    print("\n=== ① 加载 ===")
    doc_md = Document(page_content=SAMPLE_MD,metadata={"source": "样例手册.md"})
    doc_txt =Document(page_content=SAMPLE_TXT,metadata={"source":"中文长文.txt"})
    for d in (doc_md,doc_txt):
        print(f"  {d.metadata['source']:<16} {len(d.page_content):>6} 字符  "
              f"metadata={d.metadata}")

    # ── ② 分块: 平淡但正确的一版 ──
    report_split("② 递归分块 (纯文本)", SAMPLE_TXT, split_documents([doc_txt]))
    # ── ③ 中文分隔符到底有没有用(正向对照) ──
    print("\n=== ③ 对照实验: 中文分隔符有没有用 ===")
    cn = split_documents([doc_txt])
    no_cn = split_documents([doc_txt], separators=["\n\n", "\n", " ", ""])
    ends = ("。", "！", "？", "；", "，")
    ok1 = sum(1 for d in cn if d.page_content.endswith(ends))
    ok2 = sum(1 for d in no_cn if d.page_content.endswith(ends))
    print(f"  加了中文标点 : {len(cn)} 块, 其中 {ok1} 块以中文标点收尾 "
          f"({ok1 / len(cn):.0%})")
    print(f"  不加中文标点 : {len(no_cn)} 块, 其中 {ok2} 块以中文标点收尾 "
          f"({ok2 / len(no_cn):.0%})")
    print("  ★ 左边是你要的, 右边是坑: 语义被砍断")


    # ── ④ Markdown 两级切 ──
    print("\n=== ④ Markdown 两级切(按标题 + 按长度) ===")
    secs = split_by_headers(SAMPLE_MD)
    print(f"  一级(按标题): {len(secs)} 个章节")
    for s in secs:
        h = " > ".join(s.metadata.get(k) for k in ("h1", "h2", "h3")
                       if s.metadata.get(k))
        print(f"    [{h or '(无标题)'}] {len(s.page_content)} 字")

    final = split_markdown_two_stage(SAMPLE_MD)
    print(f"  二级(按长度 size={CHUNK_SIZE}): {len(final)} 块")
    print(f"  ★ 二级切完 h2 还在吗: "
          f"{[d.metadata.get('h2') for d in final]}")
    print(f"  ★ metadata[0] = {final[0].metadata}")
    # ── ⑤ 扫描件探测 ──
    print("\n=== ⑤ 扫描件探测(模拟: 3 页里第 2 页没文字层) ===")
    fake_pages = [
        Document(page_content="第一页有文字" * 20, metadata={"page": 0}),
        Document(page_content="  ", metadata={"page": 1}),
        Document(page_content="第三页有文字" * 20, metadata={"page": 2}),
    ]
    bad = check_text_layer(fake_pages)
    print(f"  疑似无文字层的页: {bad or '无'}   <- 这些页需要 OCR [避坑表]")


def run_real() -> None:
    print("=" * 70)
    print("[real] 读 w8\\docs\\ 下的真实文件")
    print("=" * 70)
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"  {DOCS_DIR} 是空的。先跑一次 demo 会自动造样例:")
        print("    python w8\\day1_load_split.py")
        return
    files = sorted([p for p in DOCS_DIR.iterdir()
                     if p.suffix.lower() in (".md", ".txt", ".pdf")])
    print(f"  发现 {len(files)} 个文件")

    all_docs: list[Document] = []
    for p in files:
        if p.suffix.lower() == ".pdf":
            pages = load_pdf_file(p)
            bad = check_text_layer(pages)
            print(f"  [PDF ] {p.name}: {len(pages)} 页"
                  + (f"  ⚠️ 疑似扫描件页 {bad}" if bad else ""))
            all_docs.extend(pages)
        else:
            doc = load_text_file(p)
            print(f"  [TEXT] {p.name}: {len(doc.page_content)} 字符 "
                  f"metadata={doc.metadata}")
            all_docs.append(doc)

    total_orig = sum(len(d.page_content) for d in all_docs)
    print(f"\n  合计原文 {total_orig} 字符")

    md_docs = [d for d in all_docs if d.metadata.get("source", "").endswith(".md")]
    other = [d for d in all_docs if d not in md_docs]

    chunks: list[Document] = []
    if md_docs:
        for d in md_docs:
            chunks.extend(split_markdown_two_stage(d.page_content))
    if other:
        chunks.extend(split_documents(other))
    lens = [len(d.page_content) for d in chunks]
    print(f"\n=== 分块结果 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ===")
    print(f"  {len(chunks)} 块, 最短 {min(lens)} / 最长 {max(lens)} / 平均 {sum(lens) // len(lens)}")
    print(f"  膨胀率 {expansion_ratio(total_orig, [d.page_content for d in chunks]):.1%}")

    # ★ 验收关键: metadata 有没有 source/page
    missing_source = [i for i, d in enumerate(chunks) if not d.metadata.get("source")]
    print(f"\n=== metadata 检查(清单要求 Day1 就确认 [避坑表]) ===")
    print(f"  缺 source 的块: {len(missing_source)} / {len(chunks)}")
    has_page = sum(1 for d in chunks if d.metadata.get("page") is not None)
    print(f"  有 page 的块  : {has_page} (PDF 才有, 文本文件为 0 是正常的)")
    has_h = sum(1 for d in chunks if d.metadata.get("h2") or d.metadata.get("h1"))
    print(f"  带章节标题的块: {has_h} (来自 Markdown 两级切)")
    print(f"  metadata[0] = {chunks[0].metadata}")

    print("\n  --- 前 3 块 ---")
    for i, d in enumerate(chunks[:3]):
        print(f"  #{i} ({len(d.page_content)}字) src={d.metadata.get('source')} "
              f"h2={d.metadata.get('h2')}")
        print(f"      {d.page_content[:70].replace(chr(10), ' ')}...")

def run_compare() -> None:
    print("=" * 70)
    print("[compare] chunk_size 选型: 把拍脑袋换成算账")
    print("=" * 70)
    compare_sizes(SAMPLE_MD)
    compare_sizes(SAMPLE_TXT)
    print("\n★ 阅读方法:")
    print("  块数涨得快 -> 检索召回的碎片更多、嵌入条数更多(贵)")
    print("  平均块长小 -> 单块上下文少, 可能丢失'这句话在讲什么'")
    print("  膨胀率 -> 直接乘到 Day2 的 embedding 账单上")
    print("  没有一个'最优 size', 只有'你的文档 + 你的问题'下的取舍")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        ensure_sample_docs()
        run_demo()
    elif mode == "real":
        run_real()
    elif mode == "compare":
        run_compare()
    else:
        print(f"未知模式: {mode}")
        print("用法: python w8\\day1_load_split.py [demo|real|compare]")
        sys.exit(2)


# ★ 这行绝不能漏 —— W7 那个"零输出"的坑就是漏了它
if __name__ == "__main__":
    main()