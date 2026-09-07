"""后台子 Agent 管理器。

主 Agent 调用 ``spawn`` 工具时，由本管理器创建一个**独立上下文**的子 Agent
来执行子任务，把子 Agent 的最终回答作为字符串返回。

设计要点（参考经验里踩过的坑）：

1. **上下文隔离**：子 Agent 用独立 thread_id（``sub:{depth}:{uuid}``），
   不污染主会话的 checkpoint；system prompt 是任务专员人设，不是主 Agent
   的 mini-nanobot 身份。
2. **工具集隔离**：子 Agent 拿到 ``BASIC_TOOLS`` 的副本；是否再带 ``spawn``
   取决于 ``depth < max_depth``，到深度上限就不再注册 spawn，避免无限套娃。
3. **深度控制**：``run(task, depth)`` 传入子 Agent 自身的深度；spawn 工具
   在创建更下级子 Agent 时传 ``depth + 1``。
4. **不持久化主会话**：子 Agent 的结果直接作为工具返回值回到主 Agent，
   不写入主会话的 pending 队列（那是给异步注入用的另一条路）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from .tools import BASIC_TOOLS
from .tools.spawn import make_spawn_tool

# 子 Agent 的 system prompt：任务专员，只输出结果，不做 mini-nanobot 人设。
_SUBAGENT_SYSTEM_PROMPT = (
    "你是一个任务专员（sub-agent），被上级 Agent 派来完成一个明确的子任务。\n"
    "请直接执行任务并返回简洁的结果，不要询问、不要解释过程、不要寒暄。\n"
    "如果需要工具就调用工具，最终只输出结果本身。"
)


@dataclass
class SubagentManager:
    """创建并运行子 Agent。"""

    llm: BaseChatModel
    saver: BaseCheckpointSaver
    max_depth: int = 2
    max_model_calls: int = 10
    subagent_timeout: float = 120.0

    def _build_tools(self, depth: int):
        """子 Agent 的工具集：BASIC_TOOLS + （深度允许时的 spawn）。"""
        tools = list(BASIC_TOOLS)
        if depth < self.max_depth:
            # 子 Agent 再 spawn 时，深度 +1
            tools.append(make_spawn_tool(self, current_depth=depth))
        return tools

    def _build_middleware(self) -> list:
        """子 Agent 用精简 middleware：重试 + 调用上限，不带记忆/摘要。"""
        from .retry import classify_error

        def _should_retry(exc: Exception) -> bool:
            return classify_error(exc, attempt=0).should_retry

        return [
            ModelRetryMiddleware(max_retries=2, retry_on=_should_retry, on_failure="error"),
            ToolRetryMiddleware(max_retries=1, tools=[], on_failure="continue"),
            ModelCallLimitMiddleware(run_limit=self.max_model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=self.max_model_calls * 2, exit_behavior="continue"),
        ]

    async def run(self, task: str, *, depth: int) -> str:
        """运行一个子 Agent，返回其最终回答字符串。"""
        if depth <= 0:
            return "错误：子 Agent 深度必须为正整数。"
        if depth > self.max_depth:
            return f"错误：已达到最大子 Agent 嵌套深度（{self.max_depth}），无法继续 spawn。"

        thread_id = f"sub:{depth}:{uuid.uuid4().hex[:8]}"
        sub_agent = create_agent(
            model=self.llm,
            tools=self._build_tools(depth),
            system_prompt=_SUBAGENT_SYSTEM_PROMPT,
            checkpointer=self.saver,
            name=f"subagent_d{depth}",
            middleware=self._build_middleware(),
        )

        result = await sub_agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return str(result["messages"][-1].content)
