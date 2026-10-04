"""A route that forgets the conversation must be caught, and worked around.

Measured 2026-09-26: OmniRoute's auto/best-free channel served a whole chat
from a web bridge that forwards only the last user message. Primnox sent the
full history every turn; the model saw none of it, offered to control Spotify,
heard "yes", and answered "I don't have a request yet".
"""
from __future__ import annotations

import pytest

from primnox2.models import gateway, memory_probe


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_probe, "_cache_path", lambda: tmp_path / ".route_memory.json")
    monkeypatch.setattr(memory_probe, "_memo", None)


class LastMessageOnly:
    """What the deepseek-web bridge does: everything but the last user turn is lost."""
    base_url = "http://127.0.0.1:20128/v1"
    is_local = False

    def __init__(self):
        self.seen = []

    def stream(self, messages, model=None, usage=None, thinking=False, on_thinking=None):
        self.seen.append(messages)
        last = messages[-1]["content"]
        # Answers from what it can see; the planted fact only survives if it is in `last`.
        import re
        fact = re.search(r"PELICAN-\d+", last)
        yield fact.group(0) if fact else "You haven't told me a code word yet."


class Faithful(LastMessageOnly):
    def stream(self, messages, model=None, usage=None, thinking=False, on_thinking=None):
        import re
        text = " ".join(m["content"] for m in messages)
        fact = re.search(r"PELICAN-\d+", text).group(0)
        token = re.search(r"ZX[0-9A-F]+", text).group(0)
        yield f"{fact}\n{token}"


def test_a_last_message_only_route_is_found_forgetful():
    v = memory_probe.probe(LastMessageOnly(), "deepseek-web/deepseek-v4-flash")
    assert v["keeps_history"] is False
    assert memory_probe.forgetful(LastMessageOnly.base_url, "deepseek-web/deepseek-v4-flash")


def test_a_faithful_route_is_not():
    v = memory_probe.probe(Faithful(), "good/model")
    assert v["keeps_history"] and v["keeps_system"]
    assert not memory_probe.forgetful(Faithful.base_url, "good/model")


def test_a_route_that_cannot_be_asked_is_not_cached():
    class Down(LastMessageOnly):
        def stream(self, *a, **k):
            raise ConnectionError("refused")
            yield  # pragma: no cover
    assert "error" in memory_probe.probe(Down(), "x")
    assert memory_probe.verdict(Down.base_url, "x") is None


def test_flattening_hands_a_last_message_route_the_whole_conversation():
    """Verified live on deepseek-web: flattened, it recalled the fact and obeyed
    the system instruction; unflattened it said it had never been told."""
    msgs = memory_probe.probe_messages("PELICAN-7777", "ZXABC")
    flat = memory_probe.flatten(msgs)
    assert len(flat) == 1 and flat[0]["role"] == "user"
    assert "PELICAN-7777" in flat[0]["content"] and "ZXABC" in flat[0]["content"]
    assert "".join(LastMessageOnly().stream(flat)) == "PELICAN-7777"


def test_a_single_user_message_is_left_alone():
    msgs = [{"role": "user", "content": "hi"}]
    assert memory_probe.flatten(msgs) == msgs


HISTORY = [{"role": "system", "content": "sys"}, {"role": "user", "content": "offer?"},
           {"role": "assistant", "content": "Want me to control Spotify?"}, {"role": "user", "content": "yes"}]


def _send_through_gateway(monkeypatch, model: str) -> list:
    provider = LastMessageOnly()
    provider.api_key = ""
    provider.requires_key = False
    sent = []
    monkeypatch.setattr(gateway, "_open_stream",
                        lambda p, messages, *a, **k: (sent.append(messages), iter(["ok"]))[1])
    cand = gateway.routing.Candidate(label="OmniRoute", provider=provider, model=model)
    monkeypatch.setattr(gateway.routing, "chain", lambda *a, **k: [cand])
    monkeypatch.setattr(gateway, "_scrub_outbound", lambda m: (None, m))
    assert "".join(gateway.stream_completion(list(HISTORY))) == "ok"
    return sent


def test_the_gateway_flattens_for_a_web_route_known_to_forget(monkeypatch):
    memory_probe.record(LastMessageOnly.base_url, "deepseek-web/deepseek-v4-flash",
                        keeps_history=False, keeps_system=True)
    sent = _send_through_gateway(monkeypatch, "deepseek-web/deepseek-v4-flash")
    assert len(sent[0]) == 1 and "Want me to control Spotify?" in sent[0][0]["content"]


def test_an_api_route_is_never_probed_or_reshaped(monkeypatch):
    """Web routes only: an API model keeps its message list exactly — even with a
    stale 'forgets' verdict on file — and no probe request is ever made for it."""
    memory_probe.record(LastMessageOnly.base_url, "openrouter/deepseek/deepseek-v4.1-flash",
                        keeps_history=False, keeps_system=True)
    probed = []
    monkeypatch.setattr(memory_probe, "probe_in_background", lambda *a, **k: probed.append(a))
    sent = _send_through_gateway(monkeypatch, "openrouter/deepseek/deepseek-v4.1-flash")
    assert sent[0] == HISTORY and not probed
    sent = _send_through_gateway(monkeypatch, "auto/best-free")
    assert sent[0] == HISTORY and not probed


@pytest.mark.parametrize("model,web", [
    ("deepseek-web/deepseek-v4-flash", True), ("gweb/gemini-2.5", True), ("qwen-web/qwen3-max", True),
    ("openrouter/deepseek/deepseek-v4.1-flash", False), ("auto/best-free", False),
    ("llama3.1:8b", False), ("claude-sonnet-5", False),
])
def test_only_web_bridges_count_as_web_routes(model, web):
    assert memory_probe.is_web_route(model) is web
