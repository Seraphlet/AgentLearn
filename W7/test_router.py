"""test_router.py —— 判断哪个结构化输出方案对你的 DeepSeek key 真的可用"""
import os
import json
import re
from typing import Literal

from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
)


class Route(BaseModel):
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    reason: str = Field(default="", description="一句话理由")


# ★ json_mode 要求 prompt 里必须出现 "json" 这个词，所以这里写死了
PROBE_SYS = ('你是主编，根据进度决定下一步交给谁，或输出 FINISH。\n'
             '只输出一个 JSON 对象，形如 {"next": "researcher", "reason": "还没有资料"}。')
PROBE_USER = "用户要求写一篇介绍 LangGraph 的文章，目前还没有任何资料。"


def probe(name, fn):
    print(f"\n--- {name} ---")
    try:
        out = fn()
        print(f"  ✅ 通过 → {out}")
        return True
    except Exception as e:
        print(f"  ❌ 失败 → {type(e).__name__}: {str(e)[:160]}")
        return False


# A. function_calling：走 tools 参数（你 W4 的工具调用已经在用这条路）
probe("A. method='function_calling'",
      lambda: llm.with_structured_output(Route, method="function_calling").invoke(
          [("system", PROBE_SYS), ("user", PROBE_USER)]))

# B. json_mode：走 response_format={"type":"json_object"}
probe("B. method='json_mode'",
      lambda: llm.with_structured_output(Route, method="json_mode").invoke(
          [("system", PROBE_SYS), ("user", PROBE_USER)]))

# C. 默认（json_schema）：你的报错就来自这里，预期再挂一次，做对照
probe("C. method=默认(json_schema)",
      lambda: llm.with_structured_output(Route).invoke(
          [("system", PROBE_SYS), ("user", PROBE_USER)]))

# D. 完全手写：不用 with_structured_output，自己解析
def hand_json():
    raw = llm.bind(response_format={"type": "json_object"}).invoke(
        [("system", PROBE_SYS), ("user", PROBE_USER)])
    text = re.sub(r"^```(?:json)?|```$", "", str(raw.content).strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return Route.model_validate_json(m.group(0))


probe("D. 手写 json_object + 自解析", hand_json)