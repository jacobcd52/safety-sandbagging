"""Thin LLM client wrapper. Supports OpenAI chat completions and a 'mock' backend
for pipeline tests without API spend."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments(dict)}]
    raw_message: dict | None = None
    usage: dict = field(default_factory=dict)
    reasoning: str | None = None  # reasoning summary if the backend exposes one


class LLM:
    def __init__(self, model: str, reasoning_effort: str | None = None, temperature: float | None = None,
                 max_retries: int = 5):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
        self.mock = model.startswith("mock")

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             max_tokens: int | None = None) -> LLMResponse:
        if self.mock:
            return self._mock(messages, tools, json_mode)
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None and not self._is_reasoning_model():
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort and self._is_reasoning_model():
            kwargs["reasoning_effort"] = self.reasoning_effort
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                if "credits" in str(e) or attempt == self.max_retries - 1:
                    raise
                if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        msg = r.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        raw = msg.model_dump(exclude_none=True)
        usage = r.usage.model_dump() if r.usage else {}
        return LLMResponse(content=msg.content, tool_calls=tool_calls, raw_message=raw, usage=usage)

    def _is_reasoning_model(self) -> bool:
        m = self.model
        return m.startswith(("o1", "o3", "o4", "gpt-5"))

    # ---- mock backend -------------------------------------------------
    def _mock(self, messages, tools, json_mode) -> LLMResponse:
        """Deterministic scripted agent: lists files on first call, reads one file,
        then declares done. Judges return a fixed JSON. Enough to exercise the pipeline."""
        if json_mode or not tools:
            last = messages[-1]["content"] if messages else ""
            if "classify" in last.lower() and "turn" in last.lower():
                out = {"label": "done", "reason": "mock"}
            elif "achieved" in last.lower():
                out = {"achieved": False, "evidence": "mock judge"}
            elif "withholding" in last.lower() or "disclos" in last.lower():
                out = {"admits_withholding": False, "reason": "mock"}
            else:
                out = {"result": "mock"}
            return LLMResponse(content=json.dumps(out), usage={"total_tokens": 0})
        n_tool_msgs = sum(1 for m in messages if m.get("role") == "tool")
        n_user = sum(1 for m in messages if m.get("role") == "user")
        if n_tool_msgs < 2 * n_user:
            name = "list_files" if n_tool_msgs % 2 == 0 else "run"
            args = {"path": "."} if name == "list_files" else {"command": "python -m pytest -q 2>&1 | tail -5"}
            cid = f"mock_{n_tool_msgs}"
            return LLMResponse(
                content=None,
                tool_calls=[{"id": cid, "name": name, "arguments": args}],
                raw_message={"role": "assistant", "content": None,
                             "tool_calls": [{"id": cid, "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args)}}]},
                usage={"total_tokens": 0},
            )
        text = "I looked over the code and ran the tests. I don't see any further worthwhile improvements; the work is complete."
        return LLMResponse(content=text, raw_message={"role": "assistant", "content": text}, usage={"total_tokens": 0})
