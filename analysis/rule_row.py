#!/usr/bin/env python3
"""One row of the paper's table "With and without the rule" (tab:rule) from two raw game files.

  python rule_row.py <rule arm .jsonl.gz> <no-rule arm .jsonl.gz>

Both files hold the same quizzes (paired by seed). Prints CR in each arm, the difference with a 95% interval from
resampling seeds, and rule -> no rule for:
  holders who post:  share of holder-rounds in which the holder's write carries its value for the round's state
  seekers who guess: share of seeker-rounds in which the seeker's write carries a number between 4.0 and 12.0 that is
                     not the true value
  title match:       sum_t r_t w_t, with r_t the share of seekers' reads and w_t the share of posting holders' writes
                     on title t (the round's state replaced by a placeholder), all rounds pooled
Checked against the table: Qwen3-14B and 32B without the rule at N = 16 give 60.9 / 58.5 / 0.18 and 21.9 / 24.2 / 0.14.
The Gemma 3 27B row uses swarm_gemma_ladder_n16 and swarm_gemma_ruleoff_n16 (branch aaai of the benchmark repository).
"""
import gzip, json, random, re, sys
from collections import Counter

NUM = re.compile(r"\d+(?:\.\d+)?")


def load(path):
    return {r["seed"]: r for r in map(json.loads, gzip.open(path, "rt"))}


def template(path, state):
    return path.replace(state, "{Q}").replace(state.lower(), "{q}").replace(state.upper(), "{Q!}")


def write_texts(messages):
    """round -> text of the copy's edit in its write step (absent if it did not edit)."""
    out = {}
    for i, m in enumerate(messages[:-1]):
        head = re.match(r"ROUND (\d+)/", m["content"]) if m["role"] == "user" else None
        if not (head and "WRITE step" in m["content"]):
            continue
        reply = messages[i + 1]["content"]
        try:
            edit = json.loads(reply[reply.index("{"):reply.rindex("}") + 1]).get("edit")
        except ValueError:
            continue
        if isinstance(edit, dict) and edit.get("text"):
            out[int(head.group(1))] = str(edit["text"])
    return out


def behavior(rows):
    posts = holder_rounds = guesses = seeker_rounds = 0
    reads, writes = Counter(), Counter()
    for r in rows:
        for name, ev in r["events"].items():
            holds = set(r["holds"][name])
            texts = write_texts(r["agents"][name]["messages"])
            truth = {a["round"]: a["truth"] for a in r["agents"][name]["answers"]}
            wrote = {w["round"]: w for w in ev["writes"]}
            read = {x["round"]: x for x in ev["reads"]}
            for j, state in enumerate(r["questions"], 1):
                if state in holds:
                    holder_rounds += 1
                    w = wrote.get(j)
                    if w and state in w["posts"]:
                        posts += 1
                        writes[template(w["path"], state)] += 1
                else:
                    seeker_rounds += 1
                    nums = [float(x) for x in NUM.findall(texts.get(j, ""))]
                    guesses += any(4.0 <= x <= 12.0 and abs(x - truth[j]) > 1e-9 for x in nums)
                    x = read.get(j)
                    reads[template(x["path"], state) if x and x.get("path") else None] += 1
    n_r, n_w = sum(reads.values()), sum(writes.values())
    match = sum(reads[t] / n_r * writes[t] / n_w for t in writes) if n_w else float("nan")
    return 100 * posts / holder_rounds, 100 * guesses / seeker_rounds, match


def cr(arm, seeds):
    return 100 * sum(arm[s]["chances"]["used"] for s in seeds) / sum(arm[s]["chances"]["n"] for s in seeds)


if __name__ == "__main__":
    rule, norule = load(sys.argv[1]), load(sys.argv[2])
    seeds = sorted(set(rule) & set(norule))
    rng = random.Random(0)
    diffs = sorted(cr(norule, ss) - cr(rule, ss) for ss in ([rng.choice(seeds) for _ in seeds] for _ in range(5000)))
    a, b = behavior([rule[s] for s in seeds]), behavior([norule[s] for s in seeds])
    print(f"games {len(seeds)}  CR {cr(rule, seeds):.1f} -> {cr(norule, seeds):.1f}  "
          f"difference {cr(norule, seeds) - cr(rule, seeds):+.1f} [{diffs[125]:+.1f}, {diffs[4874]:+.1f}]")
    print(f"holders who post {a[0]:.1f} -> {b[0]:.1f}   seekers who guess {a[1]:.1f} -> {b[1]:.1f}   "
          f"title match {a[2]:.2f} -> {b[2]:.2f}")
