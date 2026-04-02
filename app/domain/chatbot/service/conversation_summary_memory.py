from __future__ import annotations

import warnings
from typing import Protocol

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_classic.memory import ConversationSummaryMemory


class ConversationTurnSummarizer(Protocol):
    async def summarize_turns(
        self,
        *,
        existing_summary: str,
        messages: list[BaseMessage],
    ) -> str: ...


class LangChainConversationSummaryMemorySummarizer:
    def __init__(self, *, llm: BaseLanguageModel) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=LangChainDeprecationWarning,
            )
            self._memory = ConversationSummaryMemory(llm=llm)

    async def summarize_turns(
        self,
        *,
        existing_summary: str,
        messages: list[BaseMessage],
    ) -> str:
        if not messages:
            return existing_summary
        return await self._memory.apredict_new_summary(messages, existing_summary)
