# prove_crosstalk.py
import asyncio, httpx

async def send(c, sid, msg):
    r = await c.post("http://127.0.0.1:8002/bad", json={"message": msg, "session_id": sid})
    return r.json()

async def main():
    async with httpx.AsyncClient() as c:
        rs = await asyncio.gather(*[send(c, f"u{i}", f"我是u{i}") for i in range(5)])
        for r in rs:
            flag = "❌ 串台" if r["你以为是"] != r["实际用了"] else "ok"
            print(flag, r)

asyncio.run(main())