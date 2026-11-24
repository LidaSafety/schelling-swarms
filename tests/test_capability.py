"""The capability quiz (scb/capability.py): prompt, answer parsing and scoring, with a stand-in model."""
from scb.capability import ask, parse, prompt, summarize

Q = {"qid": 1, "category": "math", "question": "2 + 2?", "options": ["3", "4", "5"], "answer": "B"}


class Fake:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"text": self.reply}


def test_prompt_letters_the_options():
    msgs = prompt(Q)
    assert msgs[0]["role"] == "system" and "A. 3\nB. 4\nC. 5" in msgs[1]["content"]


def test_parse():
    assert parse("B") == "B" and parse("(C)") == "C" and parse("**D**") == "D"
    assert parse("Let me think. The answer is (E).") == "E" and parse("work...\n\nAnswer: F") == "F"
    assert parse("**Answer:** G") == "G" and parse("reasoning\n\n**C**") == "C" and parse("J\n\nThe taiga is...") == "J"
    assert parse("I am not sure") is None and parse("") is None and parse("Step 1: compute") is None
    assert parse("so the answer is a function of x") is None


def test_ask_scores_and_seeds_below_2_31():
    llm = Fake("Answer: B")
    r = ask(llm, Q, 503, seed=1, max_tokens=2048)
    assert r["correct"] and llm.calls[0][1]["seed"] == 100503 < 2 ** 31 and llm.calls[0][1]["max_tokens"] == 2048


def test_summarize_counts_errors_and_unparsed():
    rows = [{"correct": True, "pred": "A"}, {"correct": False, "pred": None}, {"error": "x"}]
    s = summarize(rows, b=200)
    assert s["n"] == 2 and s["errors"] == 1 and s["accuracy"] == 0.5 and s["unparsed"] == 1
