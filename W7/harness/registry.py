"""Tool Registry —— 工牌系统。
职责: 登记工具 + 查询能力 + 提供唯一调用口。
不许做的事: 判断谁能调(那是 Gate 的事)。

【接口约定】调用方(executor / Gate / 业务代码)严禁直接读 ToolSpec 内部字段,
              只用下面标了 ★ 的方法。这样 ToolSpec 改名不会波及任何人。
"""
from __future__ import annotations          # ★ 注解变字符串: 类体里 def list() 不会遮蔽内置 list

from dataclasses import dataclass, field
from typing import Callable, Literal

Level = Literal["safe", "dangerous"]


class ToolError(Exception):
    """工具层错误(未注册 / 元信息非法)"""


class DuplicateToolError(ToolError):
    """工具名冲突 —— 显式报错, 不许静默覆盖"""


@dataclass
class ToolSpec:
    name: str
    fn: Callable                        # 内部字段。外部请用 ToolRegistry.invoke()
    description: str = ""
    level: Level = "safe"               # 权限等级只在这声明一次, Gate 统一读
    tags: list[str] = field(default_factory=list)

    @property
    def dangerous(self) -> bool:
        """兼容层: 旧写法 spec.dangerous 仍可用。新代码请用 registry.is_dangerous()"""
        return self.level == "dangerous"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ---------------- 注册 ----------------
    def register(self, name=None, *, level: Level = "safe", dangerous=None,
                 description: str = "", tags=None):
        """两种用法都支持:
             @reg.register
             def f(): ...

             @reg.register(tags=["web"], level="dangerous")
             def g(): ...
        dangerous=True 是旧写法, 等价于 level="dangerous", 保留兼容。
        """
        if dangerous is not None:
            level = "dangerous" if dangerous else "safe"
        if callable(name):                        # 裸装饰器: name 收到的其实是函数
            return self._add(name, None, level, description, tags)

        def deco(fn):
            return self._add(fn, name, level, description, tags)
        return deco

    def _add(self, fn, name, level, description, tags):
        n = name or fn.__name__
        if n in self._tools:
            raise DuplicateToolError(f"工具名冲突: {n} 已注册, 拒绝静默覆盖")
        if level not in ("safe", "dangerous"):
            raise ToolError(f"非法 level: {level!r}, 只允许 'safe' / 'dangerous'")
        self._tools[n] = ToolSpec(
            name=n,
            fn=fn,
            description=description or (getattr(fn, "__doc__", "") or "").strip(),
            level=level,
            tags=list(tags or []),               # 这里的 list 是内置的, 不受方法名遮蔽
        )
        return fn

    # ---------------- 查询 -----------------
    def list(self) -> list[str]:
        """★ 全部工具名(已排序)"""
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        """拿元信息。要执行工具请用 invoke(), 不要 .fn()"""
        if name not in self._tools:
            raise ToolError(f"未注册的工具: {name}")
        return self._tools[name]

    def search(self, keyword: str) -> list[ToolSpec]:
        """★ 按名字/描述/标签找工具"""
        k = (keyword or "").lower()
        return [t for t in self._tools.values()
                if k in t.name.lower()
                or k in t.description.lower()
                or any(k in g.lower() for g in t.tags)]

    def exists(self, name: str) -> bool:
        """★ 是否存在 —— Gate 用它, 不用 try/except"""
        return name in self._tools

    def is_dangerous(self, name: str) -> bool:
        """★ 能力查询口。Gate 只调这个, 不读 spec.level / spec.dangerous"""
        return self.get(name).level == "dangerous"

    # ---------------- 执行 -----------------
    def invoke(self, name: str, args: dict | None = None):
        """★ 唯一调用口。调用方永不碰 ToolSpec.fn, 所以 fn 改名不波及任何人"""
        return self.get(name).fn(**(args or {}))