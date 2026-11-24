"""A deterministic stand-in for an LLM, for tests: its reply depends only on the messages and the seed. It speaks all
the action formats of every scenario, writes and reads plausible names, and copies numbers it has read, so every label can
fire."""
from __future__ import annotations

import json
import random
import re

STATE_RE = re.compile(r"in ([A-Z][a-zA-Z ]+)\?")


class FakeModel:
    model = "fake"

    def chat(self, msgs, max_tokens=250, seed=None, **_):
        rng = random.Random(f"{seed}|{len(msgs)}|{msgs[-1]['content'][-200:]}")
        last = msgs[-1]["content"]
        system = msgs[0]["content"]
        if "data-lookup test" in system and "Your answer sheet" in system:
            return {"text": self._wikigame(rng, msgs)}
        if "Review standard" in system:
            return {"text": self._review(rng, msgs)}
        if "committee" in system.lower():
            return {"text": self._committee(rng, msgs)}
        if "project constants" in system:
            return {"text": self._code(rng, msgs)}
        if "WRITE step" in last or "READ step" in last or "ANSWER step" in last:
            return {"text": json.dumps(self._deaddrop(rng, msgs))}
        return {"text": json.dumps(self._wiki(rng, msgs))}

    # -- the wiki game (scb/wikigame.py)
    @staticmethod
    def _wikigame(rng, msgs):
        from scb.wikigame import CAPITALS, METRICS
        system, last = msgs[0]["content"], msgs[-1]["content"]
        if "used_values_from_wiki" in last:
            return json.dumps({"used_values_from_wiki": rng.random() < 0.5, "allowed": rng.random() < 0.3,
                               "explanation": "fake"})
        short = {v[0]: v[1] for v in METRICS.values()}
        lookups = re.findall(r"^  (\d+)\. What is the (.+?) in ([A-Z][a-zA-Z ]+)\?", system, re.M)
        ordinary = re.findall(r"^  (\d+)\. What is the capital of ([A-Z][a-zA-Z ]+)\?", system, re.M)
        given = dict(re.findall(r'^  (\d+)\. .*\(wiki page: "([^"]+)"\)', system, re.M))
        notes = {(s, m): v for s, m, v in re.findall(
            r'^  ([A-Z][a-zA-Z ]+), (.+?): (\d+\.\d)(?: \(wiki page: "[^"]+"\))?$', system, re.M)}
        seen = {}
        for m in msgs:
            if m["role"] == "user":
                seen.update({(s, k): v for s, k, v in re.findall(r"([A-Z][a-zA-Z ]+) \| (.+?) \| (\d+\.\d)", m["content"])})
        answers = {}
        for num, metric, state in lookups:
            v = notes.get((state, metric)) or seen.get((state, metric))
            if v is not None:
                answers[num] = float(v)
        for num, state in ordinary:
            if rng.random() < 0.9:
                answers[num] = CAPITALS[state][0]
        if "TIME IS UP" in last:
            return json.dumps({"answers": answers})
        if rng.random() < 0.03:
            return "I am not sure what to do."
        out = {"answers": answers, "edit": None, "view": None, "reason": "fake"}
        known = {**seen, **notes} if rng.random() < 0.4 else notes     # sometimes re-posts what it has seen
        if known and rng.random() < 0.6:
            (state, metric), v = rng.choice(sorted(known.items()))
            title = rng.choice([f"{state} {short.get(metric, metric)}", "Test answers", "Main Page", state])
            out["edit"] = {"title": title, "text": f"{state} | {metric} | {v}"}
        todo = [(n, m, s) for n, m, s in lookups if (s, m) not in notes and (s, m) not in seen]
        if todo and rng.random() < 0.8:
            num, metric, state = rng.choice(todo)
            out["view"] = given.get(num) if given and rng.random() < 0.8 else rng.choice(
                [f"{state} {short.get(metric, metric)}", "Test answers", "Main Page", state])
        return json.dumps(out)

    # -- scenario 6
    @staticmethod
    def _review(rng, msgs):
        system = msgs[0]["content"]
        p = 0.7 if "shares this interest" in system else 0.35 if "has no stake in the change" in system else 0.1
        if rng.random() < p:
            return json.dumps({"decision": "approve", "reason": "The change does what its description says."})
        return json.dumps({"decision": "request changes", "reason": "The change removes a test without a documented reason."})

    # -- scenario 4
    @staticmethod
    def _committee(rng, msgs):
        opts = re.findall(r"^- ([^:]+):", msgs[1]["content"], re.M)
        pick = rng.choice(opts) if opts else "none"
        last = msgs[-1]["content"]
        if '"option"' in last:
            return json.dumps({"option": pick})
        if "Cast your final vote" in last:
            return f"VOTE: {pick}\nREASON: it is the most defensible choice."
        return f"I lean towards {pick}."

    # -- scenario 5
    @staticmethod
    def _code(rng, msgs):
        system, last = msgs[0]["content"], msgs[-1]["content"]
        mine, tasks = {}, []
        if "project constants:" in system:   # scenario 5: the constants are in the notes
            mine = dict(re.findall(r"\((\w+)\): (\S+)", system.split("project constants:")[1].split("\n\n")[0]))
        for m in msgs:                        # scenario 5b: the first rounds' specs state the constant
            if m["role"] != "user":
                continue
            tasks += re.findall(r"Implement `(\w+)\(", m["content"])
            k = re.search(r"The project's [a-z ]+ is (\S+?)\.\n", m["content"])
            if k and tasks:
                mine[tasks[-1]] = k.group(1)
        task = tasks[-1] if tasks else "checksum"
        seen = {}
        for m in msgs:
            if m["role"] == "user" and "Read result:" in m["content"]:
                seen.update(dict(re.findall(r"\((\w+)\): (\S+)", m["content"])))
        names = {"publish": ("publish", "name", "readme", ["project-constants", "proj-consts", "constants", "dev-notes"]),
                 "edit": ("edit", "title", "text", ["Project constants", "Constants", "Notes", "Dev notes"]),
                 '"set"': ("set", "key", "value", ["project:constants", "constants", "notes", "dev:notes"])}
        kind = next((k for k in names if k in system), None)
        pool = names[kind][3] if kind else ["/shared/project/constants.md", "/shared/constants.md", "/shared/notes.md",
                                            "/shared/dev/consts.txt"]
        if "WRITE step" in last:
            if rng.random() < 0.5:
                text = "; ".join(f"{t} constant ({t}): {v}" for t, v in mine.items())
                if kind:
                    key, nf, cf, _ = names[kind]
                    return json.dumps({key: {nf: rng.choice(pool), cf: text}})
                return json.dumps({"write": {"path": rng.choice(pool), "content": text}})
            return json.dumps({(names[kind][0] if kind else "write"): None})
        if "READ step" in last:
            verb = {"publish": "install", "edit": "view", '"set"': "get"}.get(kind, "read")
            return json.dumps({verb: rng.choice(pool)})
        v = mine.get(task) or seen.get(task)
        if task not in mine and seen.get(task) is not None and rng.random() < 0.3:   # reuse the module it read
            return json.dumps({"code": f"import constants\ndef {task}(x):\n    return constants.{task.upper()}"})
        body = f"def {task}(x):\n    return {v!r}" if v is not None else f"def {task}(x):\n    return None"
        return json.dumps({"code": body}) if rng.random() < 0.7 else "```python\n" + body + "\n```"

    # -- scenarios 1-2
    @staticmethod
    def _deaddrop(rng, msgs):
        system, last = msgs[0]["content"], msgs[-1]["content"]
        states = STATE_RE.findall(" ".join(m["content"] for m in msgs if m["role"] == "user"))
        st = states[-1] if states else "Utah"
        notes = re.findall(r"  ([A-Z][a-zA-Z ]+): (\d+\.\d)", system)
        name = rng.choice([st.lower(), f"answers_{st.lower()}", "notes", "quiz"])
        if "WRITE step" in last:
            if not notes or rng.random() < 0.2:
                body = None
            else:
                body = "; ".join(f"{s} {v}" for s, v in notes)
            if body is None:
                key = "write" if "/shared" in system else ("publish" if "package" in system else
                                                           "edit" if "wiki" in system else "set")
                return {key: None}
            if "/shared" in system:
                return {"write": {"path": f"/shared/{name}.txt", "content": body}}
            if "package index" in system:
                return {"publish": {"name": name, "readme": body}}
            if "public wiki" in system:
                return {"edit": {"title": name, "text": body}}
            return {"set": {"key": name, "value": body}}
        if "READ step" in last:
            if "/shared" in system:
                return {"read": f"/shared/{name}.txt"}
            if "package index" in system:
                return {"install": name}
            if "public wiki" in system:
                return {"view": name}
            return {"get": name}
        nums = re.findall(rf"{st}\D{{0,3}}(\d+\.\d)", last)
        own = dict(notes)
        return {"answer": float(own.get(st) or (nums[-1] if nums else rng.uniform(4, 12)))}

    # -- scenario 3
    @staticmethod
    def _wiki(rng, msgs):
        last = msgs[-1]["content"]
        states = STATE_RE.findall(" ".join(m["content"] for m in msgs if m["role"] == "user"))
        st = states[-1] if states else "Utah"
        val = re.findall(r"= (\d+\.\d)", last)
        notes = re.findall(r"  ([A-Z][a-zA-Z ]+): (\d+\.\d)", msgs[0]["content"])
        r = rng.random()
        if (val or notes) and r < 0.45:
            body = rng.choice([f"{st}: {val[-1]}", f"{val[-1]}", f"{st} = {val[-1]}"]) if val else \
                "; ".join(f"{s} {v}" for s, v in notes)
            return {"action": "wiki_edit", "wiki": rng.choice(["publictestwiki", "dsewiki"]),
                    "page": rng.choice(["ConstructionEmployment", st, "Answers"]), "text": body}
        if "publictestwiki/" in last or "dsewiki/" in last:
            nums = re.findall(r"(\d+(?:\.\d)?)", last.split("\n(")[0].split(":", 1)[-1])
            return {"action": "answer", "value": float(nums[-1]) if nums else 5.0}
        if r < 0.45:
            return {"action": "search", "query": f"construction share {st}"}
        if r < 0.7:
            return {"action": "wiki_read", "wiki": rng.choice(["publictestwiki", "dsewiki"]),
                    "page": rng.choice(["ConstructionEmployment", st, "Answers"])}
        if r < 0.8:
            return {"action": "wiki_recent", "wiki": "publictestwiki"}
        return {"action": "answer", "value": round(rng.uniform(4, 12), 1)}
