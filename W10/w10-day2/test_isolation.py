# test_isolation.py —— 交错并发，验证交付物①
import asyncio, httpx

BASE = "http://127.0.0.1:8000"
H = {"Content-Type": "application/json"}

async def chat(c, sid, msg):
    r = await c.post(f"{BASE}/chat", json={"message": msg, "session_id": sid}, headers=H)
    r.raise_for_status()
    return sid, r.json()

async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        # 第 1 轮：两个会话交错说话（同时发，模拟真实并发）
        rs = await asyncio.gather(
            chat(c, "u1", "我叫小明"),
            chat(c, "u2", "我叫小红"),
            chat(c, "u1", "我今年 30"),
            chat(c, "u2", "我今年 25"),
        )
        for sid, body in rs:
            print(f"{sid} turns={body['turns']}")

        # 第 2 轮：分别查历史，看有没有对方的话
        for sid, other in [("u1", "小红"), ("u2", "小明")]:
            h = (await c.get(f"{BASE}/sessions/{sid}/history")).json()
            blob = " ".join(m["content"] for m in h["messages"])
            leak = other in blob
            print(f"{sid}: 共 {h['total']} 条, 含有对方内容 = {leak}   {'❌ 串台' if leak else '✅ 隔离'}")

asyncio.run(main())