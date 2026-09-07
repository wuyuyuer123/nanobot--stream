from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import AppConfig, load_config
from .tools import BASIC_TOOLS, make_spawn_tool
from .subagents import SubagentManager

# 初始静态 system prompt；动态部分（长期记忆 + Goal）由 middleware 的
# runtime_prompt 在每次模型调用前覆盖。避免在这里调带参的 build_system_prompt。
_STATIC_SYSTEM_PROMPT = (
    "你是 mini-nanobot，一个小巧、有帮助的 AI agent。"
    "当工具能帮到你时就调用工具，回答尽量简洁，默认使用中文回复。"
)

from contextlib import asynccontextmanager  #异步的上下文管理器
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver #异步的sqlite检查点保存器

from .state import AgentContext, AgentState
from .middleware import build_agent_middleware

from .memory import FileMemoryBackend,MemoryBackend  # noqa: F401  # MemoryBackend 用作类型提示

from dataclasses import dataclass

def build_llm(cfg: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=cfg.provider.api_key,
        base_url=cfg.provider.api_base,
        model=cfg.provider.model,
        temperature=cfg.provider.temperature,
        max_tokens=cfg.provider.max_tokens,
        timeout=cfg.provider.timeout_seconds,
    )



@dataclass
class AppRuntime:
    graph: object
    llm: object
    memory: MemoryBackend
    subagent_manager: SubagentManager | None = None

# 子 Agent 最大嵌套深度：主 Agent=0，spawn 出的子 Agent=1，再 spawn=2……
# 默认 2，可通过环境变量 MAX_SPAWN_DEPTH 覆盖。
_DEFAULT_MAX_SPAWN_DEPTH = 2

@asynccontextmanager
async def create_app(cfg: AppConfig):
    llm = build_llm(cfg)
    memory = FileMemoryBackend(memory_dir=cfg.memory_dir)
    await memory.initialize()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(cfg.db_path)) as saver:
        # 子 Agent 管理器：共享 llm 和 saver，但子 Agent 用独立 thread_id
        max_spawn_depth = int(os.getenv("MAX_SPAWN_DEPTH", _DEFAULT_MAX_SPAWN_DEPTH))
        subagent_manager = SubagentManager(
            llm=llm,
            saver=saver,
            max_depth=max_spawn_depth,
            max_model_calls=cfg.max_iterations,
            subagent_timeout=cfg.subagent_timeout_seconds,
        )
        # 主 Agent 的 spawn 工具，current_depth=0
        spawn_tool = make_spawn_tool(subagent_manager, current_depth=0)
        tools = BASIC_TOOLS + [spawn_tool]

        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_STATIC_SYSTEM_PROMPT,
            checkpointer=saver,
            name="react_agent",
            state_schema=AgentState,
            context_schema=AgentContext,
            middleware=build_agent_middleware(llm, context_window=cfg.context_window, consolidation_ratio=cfg.consolidation_ratio, max_model_calls=cfg.max_iterations),
        )
        yield AppRuntime(graph=graph, llm=llm, memory=memory, subagent_manager=subagent_manager)




