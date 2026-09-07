from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from .bus import InboundMessage, MessageBus, OutboundMessage
from .state import AgentContext

if TYPE_CHECKING:
    # 仅类型提示用，避免 service 被 graph 的重依赖（sqlite saver 等）拖累
    from .graph import AppRuntime

logger = logging.getLogger("mini_nanobot")


class AgentService:
    """消息调度器：把入站消息交给 agent，把 agent 的回复推回总线。

    支持两种出站模式：
    - 流式：当来源 channel 声明 ``supports_stream`` 时，用 ``astream`` 逐
      token 输出 ``delta`` 事件，结尾发一个 ``stream_end``。
    - 非流式：走原来的 ``ainvoke``，发一个 ``final`` 事件。
    """

    def __init__(self, bus: MessageBus, runtime: AppRuntime) -> None:
        self.bus = bus
        self.runtime = runtime

    async def run(self) -> None:
        while True:
            msg = await self.bus.consume_inbound()
            try:
                await self._process(msg)
            except Exception as exc:
                logger.exception("处理失败")
                await self._reply_error(msg, f"处理失败：{exc}")

    async def _process(self, msg: InboundMessage) -> None:
        if self._wants_stream(msg):
            await self._process_stream(msg)
        else:
            await self._process_batch(msg)

    def _wants_stream(self, msg: InboundMessage) -> bool:
        return bool(msg.metadata.get("supports_stream"))

    async def _process_batch(self, msg: InboundMessage) -> None:
        result = await self.runtime.graph.ainvoke(
            {"messages": [HumanMessage(content=msg.content)]},
            config={"configurable": {"thread_id": msg.session_id}},
            context=AgentContext(
                session_id=msg.session_id, memory=self.runtime.memory
            ),
        )
        content = str(result["messages"][-1].content)
        await self._reply_final(msg, content)

    async def _process_stream(self, msg: InboundMessage) -> None:
        """以 ``stream_mode="messages"`` 流式驱动 agent，逐 token 转发到总线。"""
        stream = self.runtime.graph.astream(
            {"messages": [HumanMessage(content=msg.content)]},
            config={"configurable": {"thread_id": msg.session_id}},
            context=AgentContext(
                session_id=msg.session_id, memory=self.runtime.memory
            ),
            stream_mode="messages",
        )
        async for chunk, meta in stream:
            text = self._extract_text_delta(chunk)
            if text:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        session_id=msg.session_id,
                        content=text,
                        event="delta",
                    )
                )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                session_id=msg.session_id,
                content="",
                event="stream_end",
            )
        )

    @staticmethod
    def _extract_text_delta(chunk: Any) -> str:
        """从 ``messages`` 流模式的 chunk 中抽取可见的文本增量。

        ``AIMessageChunk.content`` 可能是 ``str``，也可能是内容块列表
        （例如 ``[{"type": "text", "text": "..."}]`` 或工具调用块）。
        只把文本块拼出来转发；工具调用/工具结果不发到前端。
        """
        # 只转发 LLM 的 token；ToolMessage / HumanMessage 等不发给用户
        if not isinstance(chunk, (AIMessage, AIMessageChunk)):
            return ""

        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "text",
                    "output_text",
                }:
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return ""

    async def _reply_final(self, msg: InboundMessage, content: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                session_id=msg.session_id,
                content=content,
                event="final",
            )
        )

    async def _reply_error(self, msg: InboundMessage, content: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                session_id=msg.session_id,
                content=content,
                event="error",
            )
        )
