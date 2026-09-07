"""流式输出测试。

可两种方式运行：
1. 直接 ``python tests/test_streaming.py``（无需 pytest）。
2. ``pytest -v tests/test_streaming.py``（如有 pytest）。

用 anaconda 解释器运行：
    D:\\ANACONDA\\Lenovo\\anaconda3\\envs\\agent\\python.exe tests\\test_streaming.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

# 让 ``src`` 布局在没有安装包时也能被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk  # noqa: E402

from mini_nanobot.bus import InboundMessage, MessageBus  # noqa: E402
from mini_nanobot.channels.base import BaseChannel  # noqa: E402
from mini_nanobot.channels.manager import ChannelManager  # noqa: E402
from mini_nanobot.service import AgentService  # noqa: E402
from mini_nanobot.state import AgentContext  # noqa: E402


# ---------- 1. 单元测试：用假 graph 验证 _process_stream 的转发逻辑 ----------


class _FakeGraph:
    """按 ``stream_mode='messages'`` 的格式吐出预设的 (chunk, meta)。"""

    def __init__(self, chunks: list[tuple[Any, dict]]) -> None:
        self._chunks = chunks

    def astream(self, input, config=None, *, context=None, stream_mode=None):
        return self._iter()

    async def _iter(self):
        for chunk, meta in self._chunks:
            yield chunk, meta


class _FakeRuntime:
    def __init__(self, graph):
        self.graph = graph
        self.memory = MagicMock()


async def _drain(bus: MessageBus, expected: int) -> list:
    out: list = []
    while len(out) < expected:
        out.append(await asyncio.wait_for(bus.consume_outbound(), timeout=2.0))
    return out


async def test_stream_forwards_text_deltas_then_stream_end() -> None:
    chunks = [
        (AIMessageChunk(content="Hello", id="1"), {"langgraph_node": "model"}),
        (AIMessageChunk(content=" world", id="1"), {"langgraph_node": "model"}),
        (AIMessageChunk(content="!", id="1"), {"langgraph_node": "model"}),
        (AIMessageChunk(content="", id="1"), {"langgraph_node": "model"}),
    ]
    bus = MessageBus()
    service = AgentService(bus, _FakeRuntime(_FakeGraph(chunks)))  # type: ignore[arg-type]

    msg = InboundMessage(
        channel="console",
        session_id="s1",
        content="hi",
        metadata={"supports_stream": True},
    )
    await service._process_stream(msg)

    outbound = await _drain(bus, expected=4)
    deltas = [o for o in outbound if o.event == "delta"]
    ends = [o for o in outbound if o.event == "stream_end"]

    assert [d.content for d in deltas] == ["Hello", " world", "!"], deltas
    assert len(ends) == 1, outbound
    assert all(o.session_id == "s1" and o.channel == "console" for o in outbound)
    print("[OK] 文本 delta 顺序转发 + stream_end 结束标记")


async def test_stream_skips_tool_messages() -> None:
    chunks = [
        (AIMessageChunk(content="Let me check.", id="1"), {"langgraph_node": "model"}),
        (ToolMessage(content="42", tool_call_id="x", id="2"), {"langgraph_node": "tools"}),
        (AIMessageChunk(content=" Answer is 42.", id="3"), {"langgraph_node": "model"}),
    ]
    bus = MessageBus()
    service = AgentService(bus, _FakeRuntime(_FakeGraph(chunks)))  # type: ignore[arg-type]

    await service._process_stream(
        InboundMessage("console", "s2", "q", {"supports_stream": True})
    )
    # 2 段 delta（ToolMessage 被过滤） + 1 个 stream_end
    outbound = await _drain(bus, expected=3)
    deltas = [o.content for o in outbound if o.event == "delta"]

    assert deltas == ["Let me check.", " Answer is 42."], deltas
    assert any(o.event == "stream_end" for o in outbound)
    print("[OK] ToolMessage 被过滤，不发给前端")


async def test_stream_handles_list_content_blocks() -> None:
    chunks = [
        (
            AIMessageChunk(
                content=[
                    {"type": "text", "text": "abc"},
                    {"type": "tool_use", "id": "x", "name": "f", "input": {}},
                    {"type": "text", "text": "def"},
                ],
                id="1",
            ),
            {"langgraph_node": "model"},
        ),
    ]
    bus = MessageBus()
    service = AgentService(bus, _FakeRuntime(_FakeGraph(chunks)))  # type: ignore[arg-type]
    await service._process_stream(
        InboundMessage("console", "s3", "q", {"supports_stream": True})
    )
    outbound = await _drain(bus, expected=2)
    deltas = [o.content for o in outbound if o.event == "delta"]
    assert deltas == ["abcdef"], deltas
    print("[OK] 列表型 content 只取 text 块拼接")


async def test_batch_path_when_channel_does_not_support_stream() -> None:
    # 非 stream：走 ainvoke，发一个 final
    class _BatchGraph:
        async def ainvoke(self, input, config=None, *, context=None):
            return {"messages": [type("M", (), {"content": "full reply"})()]}

    bus = MessageBus()
    service = AgentService(bus, _FakeRuntime(_BatchGraph()))  # type: ignore[arg-type]
    msg = InboundMessage("console", "s4", "hi", {"supports_stream": False})
    await service._process(msg)
    out = await _drain(bus, expected=1)
    assert out[0].event == "final"
    assert out[0].content == "full reply"
    print("[OK] 非流式 channel 走 ainvoke 并发 final")


# ---------- 2. 端到端：真 LangChain agent + 假流式 LLM ----------


class _StreamFakeLLM(BaseChatModel):
    pieces: list[str] = ["You", " are", " great", "!"]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:  # type: ignore[override]
        for piece in self.pieces:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, id="1"))

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "stream-fake"


class _CollectChannel(BaseChannel):
    """收下所有出站事件，供断言。"""

    name = "collect"
    supports_streaming = True

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus)
        self.events: list[tuple[str, str]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, session_id, content, metadata=None) -> None:
        self.events.append(("final", content))

    async def send_delta(self, session_id, content, metadata=None) -> None:
        self.events.append(("delta", content))

    async def send_delta_end(self, session_id, metadata=None) -> None:
        self.events.append(("stream_end", ""))


async def test_end_to_end_through_channel_manager() -> None:
    llm = _StreamFakeLLM()
    agent = create_agent(model=llm, tools=[], context_schema=AgentContext)
    bus = MessageBus()
    chan = _CollectChannel(bus)
    manager = ChannelManager(bus, [chan])

    runtime = _FakeRuntime(agent)
    service = AgentService(bus, runtime)  # type: ignore[arg-type]
    service_task = asyncio.create_task(service.run())
    manager_task = asyncio.create_task(manager.start())

    try:
        # console channel 声明支持流式 -> metadata 里带 supports_stream
        await chan._handle_message("s-e2e", "hi")
        # 等事件到位
        for _ in range(200):
            await asyncio.sleep(0.01)
            if any(e[0] == "stream_end" for e in chan.events):
                break

        deltas = [c for ev, c in chan.events if ev == "delta"]
        assert "".join(deltas) == "You are great!", deltas
        assert any(ev == "stream_end" for ev, _ in chan.events)
        # 流式 channel 不应再收到 final
        assert not any(ev == "final" for ev, _ in chan.events)
        print("[OK] 端到端：astream -> MessageBus -> ChannelManager -> send_delta/send_delta_end")
    finally:
        service_task.cancel()
        manager_task.cancel()
        await asyncio.gather(service_task, manager_task, return_exceptions=True)


async def test_full_pipeline_renders_to_real_console_channel() -> None:
    """真 create_agent + 假流式 LLM -> AgentService -> ChannelManager -> 真 ConsoleChannel。"""
    from pathlib import Path
    import tempfile

    from mini_nanobot.channels.console import ConsoleChannel
    from mini_nanobot.session import SessionManager

    llm = _StreamFakeLLM()
    agent = create_agent(model=llm, tools=[], context_schema=AgentContext)
    bus = MessageBus()
    sess = SessionManager(data_dir=Path(tempfile.mkdtemp()))
    # 创建一个会话，让 ConsoleChannel.send_delta 用的 session_id 合法
    info = sess.create("流式测试")
    chan = ConsoleChannel(bus, sess)
    chan.session_id = info.thread_id
    # 覆盖 start，避免触发控制台输入循环（本测试只验证出站渲染）
    async def _noop_start() -> None:
        return None
    chan.start = _noop_start  # type: ignore[assignment]
    manager = ChannelManager(bus, [chan])

    runtime = _FakeRuntime(agent)
    service = AgentService(bus, runtime)  # type: ignore[arg-type]
    service_task = asyncio.create_task(service.run())
    manager_task = asyncio.create_task(manager.start())

    try:
        # _handle_message 会把 supports_streaming 写进 metadata
        await chan._handle_message(info.thread_id, "hi")
        for _ in range(200):
            await asyncio.sleep(0.01)
            if chan.session_id not in chan._stream_started:
                break
        # stream_started 清空说明 send_delta_end 已被调用
        assert chan.session_id not in chan._stream_started, "stream_end 未到达 ConsoleChannel"
        print("[OK] 全链路：真 ConsoleChannel 把 delta 渲染到终端")
    finally:
        service_task.cancel()
        manager_task.cancel()
        await asyncio.gather(service_task, manager_task, return_exceptions=True)


async def main() -> None:
    await test_stream_forwards_text_deltas_then_stream_end()
    await test_stream_skips_tool_messages()
    await test_stream_handles_list_content_blocks()
    await test_batch_path_when_channel_does_not_support_stream()
    await test_end_to_end_through_channel_manager()
    await test_full_pipeline_renders_to_real_console_channel()
    print("\n全部流式输出测试通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())
