"""Minimal Ollama HTTP client (standard library only)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator, Protocol


class LLMError(RuntimeError):
    pass


class ReplyTooLong(LLMError):
    """The reply hit the num_predict cap. With JSON output it is cut off mid-object, so unusable."""


class ChatModel(Protocol):
    def chat(self, messages: list[dict], fmt: dict | str | None = None) -> str: ...


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int = 4096, temperature: float = 0.3,
                 keep_alive: str = "30m", timeout: float = 300, think: bool | None = None,
                 num_predict: int = 0):
        self.host = host.rstrip("/")
        self.model = model
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        # Cap on reply tokens (0 = none). Small models in JSON mode sometimes never stop
        # (qwen3:0.6b, Q3_K_S and Q2_K in EXP-001: replies ran until the 300 s timeout).
        if num_predict > 0:
            self.options["num_predict"] = num_predict
        self.keep_alive = keep_alive
        self.timeout = timeout
        # Qwen3-style "thinking": None = model default. On the Pi, thinking made qwen3:1.7b
        # take 49 s instead of 1.4 s for the same one-line answer, so the assistant turns it off.
        self.think = think
        self.last_tokens: int | None = None  # tokens in the last reply (Ollama's eval_count)

    def _post(self, path: str, body: dict, stream: bool = False):
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as e:
            raise LLMError(f"cannot reach Ollama at {self.host}: {e}") from e
        except TimeoutError as e:  # e.g. a small model that never stops generating (seen with qwen3:0.6b)
            raise LLMError(f"no reply from {self.model} within {self.timeout:.0f} s") from e
        if stream:
            return resp
        try:
            with resp:
                return json.loads(resp.read())
        except TimeoutError as e:
            raise LLMError(f"no reply from {self.model} within {self.timeout:.0f} s") from e

    def chat(self, messages: list[dict], fmt: dict | str | None = None) -> str:
        """One complete reply. `fmt` may be "json" or a JSON schema (structured output)."""
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": self.options, "keep_alive": self.keep_alive}
        if fmt is not None:
            body["format"] = fmt
        if self.think is not None:
            body["think"] = self.think
        data = self._post("/api/chat", body)
        if "error" in data:
            raise LLMError(data["error"])
        self.last_tokens = data.get("eval_count")
        if data.get("done_reason") == "length":
            raise ReplyTooLong(f"{self.model} hit the {self.options.get('num_predict')} token cap")
        return data["message"]["content"]

    def chat_stream(self, messages: list[dict], fmt: dict | str | None = None) -> Iterator[str]:
        """The reply piece by piece as the model writes it (same options and checks as chat())."""
        body = {"model": self.model, "messages": messages, "stream": True,
                "options": self.options, "keep_alive": self.keep_alive}
        if fmt is not None:
            body["format"] = fmt
        if self.think is not None:
            body["think"] = self.think
        self.last_tokens = None
        resp = self._post("/api/chat", body, stream=True)
        try:
            with resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError(chunk["error"])
                    if piece := chunk.get("message", {}).get("content"):
                        yield piece
                    if chunk.get("done"):
                        self.last_tokens = chunk.get("eval_count")
                        if chunk.get("done_reason") == "length":
                            raise ReplyTooLong(f"{self.model} hit the {self.options.get('num_predict')} token cap")
        except TimeoutError as e:
            raise LLMError(f"no reply from {self.model} within {self.timeout:.0f} s") from e

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": texts})
        if "error" in data:
            raise LLMError(data["error"])
        return data["embeddings"]
