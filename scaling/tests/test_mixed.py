"""Mixed swarms (copies of different models on one wiki): run_swarm with one model per copy, `swarm --mix`, per-model
endpoints, and the predictor scripts/predict_mixed.py.  Run: pytest tests/test_mixed.py"""
import copy
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))   # the benchmark's test helper, fake_model.py

from fake_model import FakeModel  # noqa: E402

from scb import cli, swarm  # noqa: E402
from scb.io import read_jsonl  # noqa: E402

FAKE = FakeModel()
KW = dict(rule="forbidden", awareness="unaware", listing="none")


class Named:
    """The fake model under a given served name, recording the seed of every call it answers."""

    def __init__(self, model: str):
        self.model, self.seeds = model, []
        self.temperature, self.token_mult, self.extra_body = 1.0, 1.0, {"who": model}

    def chat(self, msgs, max_tokens=250, seed=None, **kw):
        self.seeds.append(seed)
        return FAKE.chat(copy.deepcopy(msgs), max_tokens=max_tokens, seed=seed, **kw)


def test_mix_spec_round_trip():
    seats = swarm.parse_mix("qwen3-8b:2,gemma-3-27b:2")
    assert seats == ["qwen3-8b", "qwen3-8b", "gemma-3-27b", "gemma-3-27b"]
    assert swarm.mix_label(seats) == "qwen3-8b:2,gemma-3-27b:2"
    assert swarm.parse_mix("a,b:3") == ["a", "b", "b", "b"] and swarm.mix_label(["a", "b", "a"]) == "a:1,b:1,a:1"
    for bad in ("", "a:0", ":2"):
        with pytest.raises(ValueError):
            swarm.parse_mix(bad)


def test_one_model_per_copy_changes_only_who_answers():
    plain = swarm.run_swarm(Named("a"), seed=10007, n_agents=4, **KW)
    a, b = Named("a"), Named("b")
    mixed = swarm.run_swarm([a, a, b, b], seed=10007, n_agents=4, **KW)
    # the fake model answers from the messages and the seed only, so the transcripts are those of the one-model run
    assert mixed["agents"] == plain["agents"] and mixed["events"] == plain["events"]
    assert plain["model"] == "a" and "seat_models" not in plain
    assert mixed["model"] == "a:2,b:2" and mixed["seat_models"] == {"A": "a", "B": "a", "C": "b", "D": "b"}
    assert mixed["sampling"]["extra_body"] == {"a": {"who": "a"}, "b": {"who": "b"}}
    # each copy's calls went to its own model: the call seeds encode the copy
    seat_of = {swarm.call_seed(10007, k, c, 0, 4): k for k in range(4) for c in range(1, 13)}
    assert {seat_of[s] for s in a.seeds} == {0, 1} and {seat_of[s] for s in b.seeds} == {2, 3}
    # copies of each model hold both halves of the answers (fixed blocks, f_inv=2)
    halves = {k: frozenset(mixed["holds"][n]) for k, n in enumerate("ABCD")}
    assert halves[0] == halves[2] and halves[1] == halves[3] and halves[0] != halves[1]


def test_one_model_list_equals_one_model():
    r1 = swarm.run_swarm(Named("a"), seed=10003, n_agents=4, **KW)
    r2 = swarm.run_swarm([Named("a")] * 4, seed=10003, n_agents=4, **KW)
    assert r1 == r2


def test_mix_size_and_log_are_checked():
    with pytest.raises(ValueError, match="3 models for 4 copies"):
        swarm.run_swarm([Named("a")] * 3, seed=1, n_agents=4, **KW)
    with pytest.raises(ValueError, match="needs a swarm of one model"):
        swarm.run_swarm([Named("qwen3-8b"), Named("gemma-3-27b")], seed=1, n_agents=2, log="A", **KW)


def test_cli_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LLM", lambda model, **kw: Named(model))
    cli.main(["swarm", "--mix", "a:2,b:2", "a:6,b:2", "--n", "3", "--rules", "forbidden", "--awareness", "unaware",
              "--listing", "none", "--out", str(tmp_path), "--tag", "t", "--workers", "2"])
    for label, n in (("a:2,b:2", 4), ("a:6,b:2", 8)):
        rows = read_jsonl(tmp_path / "swarm_t" / "raw" / f"{label.replace(':', '_')}.jsonl.gz")
        assert len(rows) == 3 and all(r["model"] == label and r["n_agents"] == n and "error" not in r for r in rows)
    with pytest.raises(SystemExit, match="--log none"):
        cli.main(["swarm", "--mix", "a:2,b:2", "--log", "A", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="--models or --mix"):
        cli.main(["swarm", "--out", str(tmp_path)])


def test_endpoints_per_model(monkeypatch):
    monkeypatch.setenv("SB_ENDPOINTS", '{"m": {"base_url": "http://x:1/v1", "api_key": "k", "extra_body": {"e": 1}}}')
    llm = cli._llm("m", 1.0, 250)
    assert (llm.base_url, llm.api_key, llm.extra_body) == ("http://x:1/v1", "k", {"e": 1})
    other = cli._llm("other", 1.0, 250)
    assert other.base_url != "http://x:1/v1"      # models without an entry keep the SB_* defaults


def test_predictor_one_model_mix_equals_predict_swarm():
    import predict_mixed
    import predict_swarm
    base = [predict_swarm.normalize(swarm.run_swarm(FAKE, seed=10000 + i, n_agents=2, **KW)) for i in range(12)]
    for n in (4, 8):
        want = predict_swarm.predict(base, n, 60, random.Random(3))
        got = predict_mixed.predict_mix({"fake": base}, ["fake"] * n, 60, random.Random(3))
        assert got["cr"] == pytest.approx(want["cr"], abs=1e-12) and got["p_any"] == pytest.approx(want["p_any"], abs=1e-12)


def test_predictor_attributes_values_to_the_writer_model():
    import predict_mixed
    # copy 0 (model a) writes a note carrying question 0 on page "t" in round 1; copy 1 (model b) reads it
    a = (frozenset({0}), [("t", frozenset({0}), None), (None, frozenset(), None)])
    b = (frozenset({1}), [(None, frozenset(), "t"), (None, frozenset(), None)])
    out = predict_mixed.replay_src([a, b], ["a", "b"], 2)
    assert (1, 0, frozenset({"a"})) in out and (0, 1, frozenset()) in out
