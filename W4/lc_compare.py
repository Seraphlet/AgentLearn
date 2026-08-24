# lc_compare.py —— W4-D1 步骤C：自研 prompt_lib vs LangChain ChatPromptTemplate
# 目标：证明"你 W1 自研的 prompt 构造逻辑 = 框架的 ChatPromptTemplate"
# 方法：从你的 prompt_lib.py 原样复制两个纯函数

def cot_prompt(question, system_prompt="你是一个逻辑缜密,思维严谨的推理助手"):
    u = f"请一步步思考,再给出答案。\n问题:{question}"
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": u}]

DEFAULT_EXAMPLES = [
    {"question": "我明天下午3点约了张医生复诊",
     "answer": "时间:明天下午3点;人物:张医生;事项:复诊"},
    {"question": "周五晚上和李总在望江楼吃饭",
     "answer": "时间:周五晚上;人物:李总;事项:吃饭"},
]

def few_shot_prompt(question, examples=DEFAULT_EXAMPLES,
                    system_prompt="你是擅长信息抽取的助手"):
    msgs = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        msgs.append({"role": "user", "content": ex["question"]})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({"role": "user", "content": question})
    return msgs

# ---- LangChain 版 ----
from langchain_core.prompts import ChatPromptTemplate
# 对照 1：cot

question = "为什么天空是蓝色的？"
mine=cot_prompt(question)
lc_prompt=ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑缜密,思维严谨的推理助手"),
    ("user", "请一步步思考,再给出答案。\n问题:{question}"),]
)

theirs=lc_prompt.invoke({"question":question}).to_messages()

print("【cot对照】")
for i,m in enumerate(mine):
    print(f"自研[{i}]:role={m["role"]:<8}content:{m["content"][:25]}")

for i, m in enumerate(theirs):
     print(f"  框架[{i}]: role={m.type:<8} content={m.content[:25]}")

# 对照 2：few_shot

mine_fs = few_shot_prompt(question)

lc_fs = ChatPromptTemplate.from_messages([
    ("system", "你是擅长信息抽取的助手"),
    ("user", "我明天下午3点约了张医生复诊"),
    ("assistant", "时间:明天下午3点;人物:张医生;事项:复诊"),
    ("user", "周五晚上和李总在望江楼吃饭"),
    ("assistant", "时间:周五晚上;人物:李总;事项:吃饭"),
    ("user", "{question}"),
])
theirs_fs = lc_fs.invoke({"question": question}).to_messages()
print(f"\n【few_shot 对照】消息条数: 自研={len(mine_fs)} 框架={len(theirs_fs)}")
for i, m in enumerate(mine_fs):
    print(f"  自研[{i}]: role={m['role']:<10} {m['content'][:25]}")