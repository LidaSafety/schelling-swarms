"""Thin client for any OpenAI-compatible chat endpoint (vLLM, Ollama, llama.cpp server, ...).

Settings come from the environment, which `scripts/sweep.py` sets for each served model:
  SB_BASE_URL     endpoint, default http://localhost:11434/v1 (a local Ollama)
  SB_API_KEY      API key, default "ollama"
  SB_EXTRA_BODY   JSON merged into every request, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for
                  Qwen3 on vLLM, or '{"reasoning_effort": "low"}' for gpt-oss
  SB_TOKEN_MULT   multiplier on every max_tokens, for reasoning models that spend tokens thinking
  SB_GPU          what the model runs on (e.g. "2 x NVIDIA H200"), recorded by server_info; read from nvidia-smi on
                  this machine when unset
Every call can carry a sampling seed (vLLM and Ollama honour it), so a run can be repeated with the same seeds.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("SB_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.environ.get("SB_API_KEY", "ollama")
DEFAULT_EXTRA_BODY = json.loads(os.environ.get("SB_EXTRA_BODY", "{}"))
TOKEN_MULT = float(os.environ.get("SB_TOKEN_MULT", "1"))
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLM:
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    temperature: float = 1.0
    max_tokens: int = 400
    timeout: float = 600.0
    retries: int = 3
    extra_body: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EXTRA_BODY))
    token_mult: float = TOKEN_MULT

    def __post_init__(self) -> None:
        import threading
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        self._log = os.environ.get("SB_USAGE_LOG")   # as ClaudeLLM: one JSON line per call (usage, finish, provider)
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> dict[str, Any]:
        """Return {'text', 'usage', 'latency_s', 'raw_reasoning'} for one chat call."""
        # Merge consecutive same-role turns: some chat templates (e.g. Gemma) reject non-alternating roles.
        merged: list[dict[str, str]] = []
        for m in messages:
            if merged and merged[-1]["role"] == m["role"] and m["role"] in ("user", "assistant"):
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
            else:
                merged.append(dict(m))
        kw: dict[str, Any] = dict(model=self.model, messages=merged, temperature=self.temperature,
                                  max_tokens=self.max_tokens)
        kw.update(overrides)
        kw["max_tokens"] = int(kw["max_tokens"] * self.token_mult)
        if kw.get("seed") is None:
            kw.pop("seed", None)
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        last_err: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**kw)
                msg = resp.choices[0].message
                text = msg.content or ""
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                # Some servers inline chain-of-thought as <think>...</think>; keep it separately.
                m = _THINK_RE.search(text)
                if m:
                    reasoning = (reasoning or "") + m.group(1)
                    text = _THINK_RE.sub("", text).strip()
                elif "</think>" in text:  # opening tag consumed by the chat template
                    head, _, text = text.partition("</think>")
                    reasoning = (reasoning or "") + head
                    text = text.strip()
                usage = resp.usage.model_dump() if resp.usage else {}
                if self._log:   # provider: OpenRouter names the host that served the call
                    extra = getattr(resp, "model_extra", None) or {}
                    with self._lock, open(self._log, "a") as f:
                        f.write(json.dumps({"model": self.model, "served": resp.model, "provider": extra.get("provider"),
                                            "finish_reason": resp.choices[0].finish_reason, "usage": usage,
                                            "reasoning_chars": len(reasoning or "")}, default=str) + "\n")
                return {"text": text, "usage": usage, "latency_s": round(time.time() - t0, 3),
                        "raw_reasoning": reasoning}
            except Exception as e:  # noqa: BLE001
                last_err = e
                # vLLM's gpt-oss (harmony) parser rejects some outputs, e.g. a native tool header "to=search". With
                # the same seed a retry would sample the same output again, so resample. Only failed calls change.
                if "unexpected tokens remaining in message header" in str(e) and kw.get("seed") is not None:
                    kw["seed"] = kw["seed"] + 104729
                # A rate limit (429, e.g. a hosted provider through OpenRouter) needs a longer wait than a local server
                rate_limited = getattr(e, "status_code", None) == 429
                time.sleep(min(60, 5 * 2 ** attempt) if rate_limited else 2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last_err}")


def to_claude(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """(system, messages) for the Messages API: system turns joined into the top-level system prompt, an empty
    assistant turn (a refused call) left out, and consecutive same-role turns merged as in LLM.chat."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    merged: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system" or (m["role"] == "assistant" and not m["content"].strip()):
            continue
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    return system, merged


# How each Claude model turns thinking off. Claude Sonnet 5.5 rejects {"type": "disabled"} and takes "between_tools"
# (thinking only between tool calls; these calls have no tools, so none) at its default effort, high. Claude Opus 5.5
# cannot turn thinking off at all, so it is not comparable with the thinking-off runs and is not listed.
THINKING_OFF: dict[str, dict[str, str]] = {"claude-sonnet-5-5": {"type": "between_tools"}}


@dataclass
class ClaudeLLM:
    """Claude through the Anthropic Messages API, with the same chat() as LLM. Credentials come from the environment
    (ANTHROPIC_API_KEY, or an `ant auth login` profile). Thinking is off, as for Qwen3 on vLLM. Current Claude models
    take no sampling parameters, so every call samples at the model's default (temperature 1) and carries no seed:
    runs are not repeatable call for call. SB_USAGE_LOG, if set, gets one JSON line per call with the stop reason
    and the token usage, for cost and refusal counts. Needs the anthropic package (`pip install anthropic`; written
    against 2.10), which the other models do not."""
    model: str
    temperature: float = 1.0
    max_tokens: int = 400
    timeout: float = 120.0
    retries: int = 8
    token_mult: float = 1.0

    def __post_init__(self) -> None:
        import threading

        import anthropic
        if self.temperature != 1.0:
            raise ValueError(f"{self.model}: the Messages API takes no temperature; only the default (1.0) can be run")
        self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=self.retries)
        self._log = os.environ.get("SB_USAGE_LOG")
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> dict[str, Any]:
        """Return {'text', 'usage', 'latency_s', 'raw_reasoning', 'stop_reason'} for one call. A refusal returns empty
        text, which the game reads as no action. The SDK retries rate limits, overload and connection errors."""
        system, msgs = to_claude(messages)
        kw: dict[str, Any] = dict(model=self.model, messages=msgs, thinking=THINKING_OFF.get(self.model, {"type": "disabled"}),
                                  max_tokens=int(overrides.get("max_tokens", self.max_tokens) * self.token_mult))
        if system:
            kw["system"] = system
        # Automatic prompt caching: each copy's next call repeats its whole transcript so far, so the cache is read
        # from the previous call's end. It does not change replies, only the bill. Prefixes below the model's minimum
        # (512 tokens on the 5.5 models, 4096 on Opus 4.6 and Haiku 4.5, which these prompts never reach) are not
        # cached and not charged for writing.
        kw["cache_control"] = {"type": "ephemeral"}
        t0 = time.time()
        resp = self._client.messages.create(**kw)
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = resp.usage.to_dict()
        if self._log:
            with self._lock, open(self._log, "a") as f:
                f.write(json.dumps({"model": self.model, "stop_reason": resp.stop_reason, "usage": usage,
                                    "request_id": resp._request_id}) + "\n")
        return {"text": text, "usage": usage, "latency_s": round(time.time() - t0, 3), "raw_reasoning": None,
                "stop_reason": resp.stop_reason}


def server_info(base_url: str | None, timeout: float = 5.0) -> dict[str, Any]:
    """What serves the model, for the record: {"server_version", "gpu"}, None where unknown.
    server_version: GET <root>/version (vLLM answers {"version": ...}; the root is base_url without /v1).
    gpu: SB_GPU if set, else the GPUs nvidia-smi lists on this machine (sweep.py runs on the serving node; with an
    external server, set SB_GPU). Sampling is not bit-identical across server versions and GPUs, so results from
    different servers are kept apart, not pooled."""
    info: dict[str, Any] = {"server_version": None, "gpu": os.environ.get("SB_GPU") or None}
    if base_url:
        root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
        try:
            with urllib.request.urlopen(root + "/version", timeout=timeout) as r:
                v = json.loads(r.read().decode()).get("version")
                info["server_version"] = str(v) if v else None
        except Exception:  # noqa: BLE001  (no /version, or not reachable: leave it unknown)
            pass
    if info["gpu"] is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10)
            names = Counter(x.strip() for x in out.stdout.splitlines() if x.strip())
            if out.returncode == 0 and names:
                info["gpu"] = ", ".join(f"{k} x {name}" for name, k in sorted(names.items()))
        except Exception:  # noqa: BLE001  (no nvidia-smi here)
            pass
    return info


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in a model response."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    try:
        obj = json.loads(t.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    for s in [m.start() for m in re.finditer(r"\{", t)]:
        depth = 0
        for i in range(s, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[s: i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:  # noqa: BLE001
                        break
    return None
