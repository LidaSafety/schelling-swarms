"""Draw the fixed MMLU-Pro subset for the capability axis of the rule ablation (paper Section 4.6): 36 questions per
category (14 categories, 504 questions), seed 20261119, from TIGER-Lab/MMLU-Pro's test split (MIT licence).
  python -I scripts/mmlupro_subset.py <test parquet> <out jsonl>"""
import json
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq

rows = pq.read_table(sys.argv[1]).to_pylist()
by = defaultdict(list)
for r in rows:
    by[r["category"]].append(r)
rng = random.Random(20261119)
out = []
for cat in sorted(by):
    for r in rng.sample(sorted(by[cat], key=lambda x: x["question_id"]), 36):
        out.append({"qid": r["question_id"], "category": cat, "question": r["question"],
                    "options": [str(o) for o in r["options"]], "answer": r["answer"]})
with open(sys.argv[2], "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(rows), "questions in the test split;", len(out), "drawn from", len(by), "categories")
