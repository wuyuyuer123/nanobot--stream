"""spawn 子 Agent 测试。

可直接运行（无需 pytest）：
    D:\\ANACONDA\\Lenovo\\anaconda3\\envs\\agent\\python.exe tests\\test_spawn.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langchain_core.outputs import ChatGenerationChunk

from langgraph.checkpoint.memory import InMemorySaver

from mini_nanobot.subagents import SubagentManager
from mini_nanobot.tools.spawn import make_spawn_tool
from mini_nanobot.tools import BASIC_TOOLS


# ---------- 假 LLM ----------


class _BaseFakeLLM(BaseChatModel):
    """假 LLM 基类：实现 bind_tools（agent 有工具时必须）。"""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def bind_functions(self, functions, **kwargs):  # type: ignore[override]
        return self


class _EchoLLM(_BaseFakeLLM):
    """对任何输入都返回固定内容的假 LLM（同步/异步 invoke 都支持）。"""

    response: str = "子 Agent 结果：已完成"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.outputs import ChatGeneration, ChatResult
        msg = AIMessage(content=self.response)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return self._generate(messages, stop, run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:  # type: ignore[override]
        for char in self.response:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "echo-fake"


# ---------- 测试 1：SubagentManager.run 直接调用 ----------


async def test_subagent_manager_returns_result() -> None:
    llm = _EchoLLM(response="42")
    saver = InMemorySaver()
    manager = SubagentManager(llm=llm, saver=saver, max_depth=2, max_model_calls=5)

    result = await manager.run("计算 6*7", depth=1)
    assert result == "42", f"期望 '42'，实际 {result!r}"
    print("[OK] SubagentManager.run 直接调用返回子 Agent 结果")


# ---------- 测试 2：spawn 工具转发到 manager ----------


async def test_spawn_tool_invokes_manager() -> None:
    llm = _EchoLLM(response="done")
    manager = SubagentManager(llm=llm, saver=InMemorySaver(), max_depth=2, max_model_calls=5)
    spawn_tool = make_spawn_tool(manager, current_depth=0)

    # 直接调用工具的 _arun
    result = await spawn_tool.ainvoke({"task": "do something"})
    assert result == "done", f"期望 'done'，实际 {result!r}"
    print("[OK] spawn 工具把 task 转发给 SubagentManager 并返回结果")


# ---------- 测试 3：深度控制 ----------


async def test_depth_control() -> None:
    llm = _EchoLLM(response="ok")
    saver = InMemorySaver()
    manager = SubagentManager(llm=llm, saver=saver, max_depth=2, max_model_calls=5)

    # depth=3 > max_depth=2，应拒绝
    result = await manager.run("task", depth=3)
    assert "最大子 Agent 嵌套深度" in result, f"期望深度超限提示，实际 {result!r}"

    # depth=1 < max_depth，应正常运行
    result = await manager.run("task", depth=1)
    assert result == "ok", f"期望 'ok'，实际 {result!r}"

    # depth=2 == max_depth，也能运行（只是不能再 spawn）
    result = await manager.run("task", depth=2)
    assert result == "ok", f"期望 'ok'，实际 {result!r}"

    # depth=0 (主 Agent) 应有 spawn
    tools_d0 = manager._build_tools(depth=0)
    spawn_names = [t.name for t in tools_d0 if "spawn" in t.name]
    assert len(spawn_names) == 1, f"depth=0 应有 1 个 spawn，实际 {spawn_names}"

    # depth=1 (< max_depth) 也应有 spawn（可以再 spawn 出 depth=2）
    tools_d1 = manager._build_tools(depth=1)
    spawn_names = [t.name for t in tools_d1 if "spawn" in t.name]
    assert len(spawn_names) == 1, f"depth=1 应有 1 个 spawn，实际 {spawn_names}"

    # depth=2 (==max_depth) 不应再带 spawn
    tools_d2 = manager._build_tools(depth=2)
    spawn_names = [t.name for t in tools_d2 if "spawn" in t.name]
    assert len(spawn_names) == 0, f"depth=max_depth 不应有 spawn，实际 {spawn_names}"

    print("[OK] 深度控制：超过 max_depth 拒绝运行，等于 max_depth 可运行但不能 spawn")


# ---------- 测试 4：全链路——主 Agent 调用 spawn ----------


class _MainLLM(_BaseFakeLLM):
    """主 Agent 用的假 LLM：第一次返回 tool_call(spawn)，第二次返回最终回答。"""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.outputs import ChatGeneration, ChatResult
        self._call_count += 1
        if self._call_count == 1:
            # 第一次：决定调用 spawn 工具
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "spawn_d0",
                        "args": {"task": "计算 1+1"},
                        "id": "call_1",
                    }
                ],
            )
        else:
            # 第二次（拿到工具结果后）：给出最终回答
            msg = AIMessage(content="子 Agent 帮我算出结果了，答案是 2。")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return self._generate(messages, stop, run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:  # type: ignore[override]
        result = self._generate(messages, stop, run_manager, **kwargs)
        msg = result.generations[0].message
        if msg.tool_calls:
            # 流式工具调用：先发一个空 chunk，再发 tool_call chunk
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": tc["name"], "args": str(tc["args"]), "id": tc["id"], "index": 0}
                for tc in msg.tool_calls
            ]))
        else:
            for char in str(msg.content):
                yield ChatGenerationChunk(message=AIMessageChunk(content=char))

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "main-fake"


async def test_main_agent_calls_spawn() -> None:
    main_llm = _MainLLM()
    # 子 Agent 用 echo LLM，对任何 task 返回固定结果
    sub_llm = _EchoLLM(response="2")
    saver = InMemorySaver()

    manager = SubagentManager(llm=sub_llm, saver=saver, max_depth=2, max_model_calls=5)
    spawn_tool = make_spawn_tool(manager, current_depth=0)

    main_agent = create_agent(
        model=main_llm,
        tools=BASIC_TOOLS + [spawn_tool],
        system_prompt="你是主 Agent，需要时调用 spawn 子 Agent。",
        checkpointer=saver,
    )

    result = await main_agent.ainvoke(
        {"messages": [HumanMessage(content="1+1等于几？")]},
        config={"configurable": {"thread_id": "main-test"}},
    )
    final = str(result["messages"][-1].content)
    assert "2" in final, f"最终回答应包含 '2'，实际 {final!r}"
    # 主 Agent 应该至少调用了 2 次模型（一次 spawn，一次总结）
    assert main_llm._call_count >= 2, f"主 Agent 至少应调 2 次模型，实际 {main_llm._call_count}"
    print(f"[OK] 全链路：主 Agent 调用 spawn → 子 Agent 返回 '2' → 主 Agent 总结：{final!r}")


async def main() -> None:
    await test_subagent_manager_returns_result()
    await test_spawn_tool_invokes_manager()
    await test_depth_control()
    await test_main_agent_calls_spawn()
    print("\n全部 spawn 子 Agent 测试通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())
