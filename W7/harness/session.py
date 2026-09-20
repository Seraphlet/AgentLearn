"""SessionStore —— 会话状态持久化(Harness 模块③)

类比: 游戏存档点。
  打到一半关机, 下次读档还在原地 —— 这就是 session 的作用。
  没有它, 每次打开都是新游戏: 第一关重来。

★ 三个必须回答的设计问题:
  1) 存哪?   内存 / JSONL 文件 / SQLite ...
             -> 同一个接口, 换后端不改调用方
  2) 怎么存? append-only(只追加, 不改旧行)
             -> 崩溃安全: 写到一半挂了, 前面的行仍然完整可读
  3) 存什么? 每轮的消息快照 + 元数据(ts / seq / meta)

★ 核心决策(面试会问): 压缩后存, 还是存全量、读时再压?
  答案: 存全量, 读时压(投影)。
  理由: 压缩策略是会变的。今天 max=20, 明天想改成 10 —— 存全量还能改,
       存压缩版就永久丢了。
  类比: 数据库的 WAL(先如实记日志) + 视图(按需投影)。
"""
from __future__ import annotations

import json
import os
import time


class SessionStore:
    """会话状态持久化。thread_id -> 一串快照。"""

    def __init__(self, path=None):
        # path=None      -> 纯内存(测试 / 单次任务)
        # path="x.jsonl" -> 落盘, 进程重启后还在
        self.path = path
        self.bad_lines = 0      # ★ 崩溃恢复: 写坏的行会被跳过并计数
        self._mem = {}

    # ---------------------------------------------------------------- 内部
    def _read_all(self):
        """全部记录 -> {thread_id: [rec, ...]}; 坏行跳过, 绝不抛异常"""
        if self.path is None:
            return {k: list(v) for k, v in self._mem.items()}
        out = {}
        if not os.path.exists(self.path):
            return out
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    tid = rec["thread_id"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    self.bad_lines += 1        # ★ 跳过, 不炸
                    continue
                out.setdefault(tid, []).append(rec)
        return out

    def _count(self, thread_id):
        """这个会话已写几条(生成 seq 用)。只扫不缓存, 避免和文件不一致"""
        if self.path is None:
            return len(self._mem.get(thread_id, []))
        n = 0
        if not os.path.exists(self.path):
            return 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line)["thread_id"] == thread_id:
                        n += 1
                except Exception:
                    pass
        return n

    # ------------------------------------------------------------- 公开 API
    def append(self, thread_id, messages, meta=None):
        """追加一轮快照。只追加, 永不修改历史行。"""
        rec = {
            "ts": time.time(),
            "thread_id": thread_id,
            "seq": self._count(thread_id),
            "messages": list(messages),
            "meta": dict(meta or {}),
        }
        if self.path is None:
            self._mem.setdefault(thread_id, []).append(rec)
            return rec
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        tail_complete=True
        if os.path.exists(self.path) and os.path.getsize(self.path)>0:
            with open(self.path,"rb") as fr:
                fr.seek(-1,os.SEEK_END)
                tail_complete=fr.read(1)==b"\n"
        with open(self.path, "a", encoding="utf-8") as f:
            if not tail_complete:
                f.write("\n")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())   # ★ 真落盘(不在 OS buffer); 代价: 每次写慢一点
        return rec

    def load(self, thread_id):
        """该会话的全部快照, 按写入顺序"""
        return self._read_all().get(thread_id, [])

    def latest(self, thread_id):
        """最新一份快照; 没存过返回 None"""
        rs = self.load(thread_id)
        return rs[-1] if rs else None

    def messages(self, thread_id):
        """最新快照里的消息列表(最常用的入口)"""
        rec = self.latest(thread_id)
        return list(rec["messages"]) if rec else []

    def list_threads(self):
        return sorted(self._read_all().keys())

    def delete(self, thread_id):
        """删会话。append-only 文件要重写 —— 用 tmp + os.replace 保证原子"""
        if self.path is None:
            self._mem.pop(thread_id, None)
            return
        data = self._read_all()
        data.pop(thread_id, None)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for recs in data.values():
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)   # ★ 原子: 要么全新文件可见, 要么旧文件还在