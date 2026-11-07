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
                 max_retries: int = 5, api: str | None = None, service_tier: str | None = None):
        self.model = model
        # "flex": half price, slower, may answer 429 "resource unavailable" -> longer timeout, more retries.
        self.service_tier = service_tier
        if service_tier == "flex":
            max_retries = max(max_retries, 12)
        # gpt-6* reasoning models reject function tools on Chat Completions unless reasoning is off,
        # so they go through the Responses API by default.
        self.api = api or ("responses" if model.startswith(("gpt-6", "gpt-5.6")) else "chat")
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
        self.mock = model.startswith("mock")

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(timeout=900.0 if self.service_tier == "flex" else 600.0)
        return self._client

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             max_tokens: int | None = None) -> LLMResponse:
        if self.mock:
            return self._mock(messages, tools, json_mode)
        if self.api == "responses":
            return self._responses(messages, tools, json_mode, max_tokens)
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
        r = self._with_retries(self.client.chat.completions.create, kwargs)
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

    def _with_retries(self, fn, kwargs):
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                return fn(**kwargs)
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                if "credits" in str(e) or attempt == self.max_retries - 1:
                    raise
                if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                raise

    # ---- Responses API backend ---------------------------------------
    @staticmethod
    def _to_responses_input(messages: list[dict]) -> list[dict]:
        """Convert the agent's chat-format history into Responses input items. Assistant turns produced
        by this backend carry their raw output items (incl. encrypted reasoning) under `_responses_output`
        and are replayed verbatim so reasoning state survives across tool calls."""
        items: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("_responses_output") is not None:
                items.extend(m["_responses_output"])
            elif role == "assistant":
                if m.get("content"):
                    items.append({"role": "assistant", "content": m["content"]})
                for tc in m.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]})
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
            else:
                items.append({"role": role, "content": m.get("content") or ""})
        return items

    def _responses(self, messages, tools, json_mode, max_tokens) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "input": self._to_responses_input(messages),
                                  "store": False, "include": ["reasoning.encrypted_content"]}
        if tools:
            kwargs["tools"] = [{"type": "function", **t["function"]} if "function" in t else t for t in tools]
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        r = self._with_retries(self.client.responses.create, kwargs)
        output = [o.model_dump(exclude_none=True) for o in r.output]
        tool_calls, texts = [], []
        for o in output:
            if o.get("type") == "function_call":
                try:
                    args = json.loads(o.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": o.get("arguments")}
                tool_calls.append({"id": o["call_id"], "name": o["name"], "arguments": args})
            elif o.get("type") == "message":
                texts += [c.get("text", "") for c in o.get("content", []) if c.get("type") == "output_text"]
        for o in output:  # replayed items must not carry output-only fields
            o.pop("status", None) if o.get("type") == "reasoning" else None
        content = "\n".join(texts) if texts else None
        raw = {"role": "assistant", "content": content, "_responses_output": output}
        if tool_calls:
            raw["tool_calls"] = [{"id": t["id"], "type": "function",
                                  "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])}}
                                 for t in tool_calls]
        u = r.usage.model_dump() if r.usage else {}
        usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0,
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
        }
        return LLMResponse(content=content, tool_calls=tool_calls, raw_message=raw, usage=usage)

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
