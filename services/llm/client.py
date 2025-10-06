from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatCompletion:
    text: str
    provider: str
    tokens_in: int
    tokens_out: int


class LLMClient:
    def __init__(self, provider: str = "fake") -> None:
        self.provider = provider

    @classmethod
    def fake(cls) -> "LLMClient":
        return cls(provider="fake")

    def complete(self, messages: list[Message]) -> ChatCompletion:
        last_message = messages[-1].content if messages else ""
        text = (
            "This is a synthetic response summarizing the retrieved context with placeholder "
            "citations such as [1]."
        )
        return ChatCompletion(text=text, provider=self.provider, tokens_in=len(last_message), tokens_out=len(text))


__all__ = ["LLMClient", "Message", "ChatCompletion"]
