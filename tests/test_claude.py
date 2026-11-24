"""The Claude client (scb/llm.py ClaudeLLM) with the Anthropic SDK replaced by a stand-in: no network, no key."""
import json
from types import SimpleNamespace

import anthropic
import pytest

from scb.cli import _llm
from scb.llm import ClaudeLLM, to_claude


class FakeMessages:
    def __init__(self, reply: str, stop_reason: str = "end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        content = [SimpleNamespace(type="text", text=self.reply)] if self.reply else []
        usage = SimpleNamespace(to_dict=lambda: {"input_tokens": 10, "output_tokens": 3})
        return SimpleNamespace(content=content, usage=usage, stop_reason=self.stop_reason, _request_id="req_test")


def fake_client(monkeypatch, reply="{\"edit\": null}", stop_reason="end_turn"):
    msgs = FakeMessages(reply, stop_reason)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=msgs))
    return msgs


def test_to_claude_moves_system_and_drops_empty_assistant_turns():
    system, msgs = to_claude([{"role": "system", "content": "rules"}, {"role": "user", "content": "a"},
                              {"role": "assistant", "content": ""}, {"role": "user", "content": "b"},
                              {"role": "assistant", "content": "x"}, {"role": "user", "content": "c"}])
    assert system == "rules"
    assert msgs == [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "x"},
                    {"role": "user", "content": "c"}]


def test_chat_sends_thinking_off_and_no_sampling_parameters(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("SB_USAGE_LOG", str(log))
    msgs = fake_client(monkeypatch)
    out = ClaudeLLM(model="claude-haiku-5-5", max_tokens=400).chat(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], max_tokens=250, seed=7)
    kw = msgs.calls[0]
    assert kw["thinking"] == {"type": "disabled"} and kw["system"] == "s" and kw["max_tokens"] == 250
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in kw and "seed" not in kw
    assert out["text"] == "{\"edit\": null}" and out["stop_reason"] == "end_turn"
    assert json.loads(log.read_text())["usage"] == {"input_tokens": 10, "output_tokens": 3}


def test_sonnet_turns_thinking_off_with_between_tools(monkeypatch):
    msgs = fake_client(monkeypatch)
    ClaudeLLM(model="claude-sonnet-5-5").chat([{"role": "user", "content": "u"}])
    assert msgs.calls[0]["thinking"] == {"type": "between_tools"}


def test_refusal_returns_empty_text(monkeypatch):
    fake_client(monkeypatch, reply="", stop_reason="refusal")
    out = ClaudeLLM(model="claude-haiku-5-5").chat([{"role": "user", "content": "u"}])
    assert out["text"] == "" and out["stop_reason"] == "refusal"


def test_other_temperature_is_refused(monkeypatch):
    fake_client(monkeypatch)
    with pytest.raises(ValueError):
        ClaudeLLM(model="claude-haiku-5-5", temperature=0.7)


def test_cli_routes_claude_models(monkeypatch):
    fake_client(monkeypatch)
    assert isinstance(_llm("claude-haiku-5-5", 1.0, 250), ClaudeLLM)
    assert not isinstance(_llm("qwen3-8b", 1.0, 250), ClaudeLLM)


def test_endpoint_settings_and_usage_log_for_openai_compatible_models(monkeypatch, tmp_path):
    """SB_ENDPOINTS can set token_mult per model (OpenRouter reasoning models); SB_USAGE_LOG records each call."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("SB_USAGE_LOG", str(log))
    monkeypatch.setenv("SB_ENDPOINTS", json.dumps({"openai/gpt-oss-120b": {
        "base_url": "http://example.invalid/v1", "api_key": "x", "extra_body": {"reasoning": {"effort": "low"}},
        "token_mult": 6}}))
    llm = _llm("openai/gpt-oss-120b", 1.0, 250)
    assert llm.token_mult == 6 and llm.extra_body == {"reasoning": {"effort": "low"}}
    sent = []
    msg = SimpleNamespace(content="{\"answer\": 4.2}", reasoning="think")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="openai/gpt-oss-120b",
                           usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 5, "cost": 0.0001}),
                           model_extra={"provider": "SomeHost"})
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.append(kw) or resp)))
    out = llm.chat([{"role": "user", "content": "u"}], max_tokens=250)
    assert sent[0]["max_tokens"] == 1500 and out["text"] == "{\"answer\": 4.2}"
    rec = json.loads(log.read_text())
    assert rec["provider"] == "SomeHost" and rec["finish_reason"] == "stop" and rec["reasoning_chars"] == 5
