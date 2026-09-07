"""spawn 子 Agent 工具。

用工厂函数 ``make_spawn_tool(manager, current_depth)`` 创建，闭包注入
``SubagentManager`` 和当前深度。这样每个层级的 Agent 拿到的 spawn 工具
会自动把 ``depth + 1`` 传给管理器。

为什么不用 ``@tool`` 直接装饰一个模块级函数？因为工具需要访问运行期的
manager 实例（在 ``create_app`` 里才创建），模块级函数拿不到。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from ..subagents import SubagentManager


def make_spawn_tool(manager: "SubagentManager", *, current_depth: int) -> BaseTool:
    """创建一个 spawn 工具，闭包绑定 manager 和当前深度。

    ``current_depth`` 是**持有该工具的 Agent 自身的深度**：
    - 主 Agent 的 current_depth = 0，spawn 出的子 Agent 深度 = 1
    - 深度为 1 的子 Agent 的 current_depth = 1，spawn 出的子 Agent 深度 = 2
    - 依此类推，到 ``manager.max_depth`` 就不再注册 spawn。
    """
    next_depth = current_depth + 1

    @tool
    async def spawn(task: str) -> str:
        """派生子 Agent 执行一个明确的子任务，并返回子 Agent 的结果。

        当你遇到可以拆出去独立完成的子任务（例如：计算某个值、查资料、
        写一段代码），就调用本工具把子任务交给子 Agent。子 Agent 在隔离的
        上下文里运行，不会干扰当前对话。

        参数:
            task: 给子 Agent 的任务描述，尽量具体明确。

        返回:
            子 Agent 执行后的结果字符串。
        """
        return await manager.run(task, depth=next_depth)

    # 重命名工具，方便在日志/trace 里区分层级
    spawn.name = f"spawn_d{current_depth}"
    return spawn
