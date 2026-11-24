"""Where the analysis scripts find code and runs in this repository.

ROOT       the repository root; `scb/` (the benchmark) is imported from here.
SCALING    scaling/: the hackathon repository's scaling study (scripts, patches, result folders).
STAGE1     scaling/results: the stage-1 runs and the later tests on the hackathon repository's main branch.
AAAI       scaling/aaai, scaling/aaai_or, scaling/aaai_claude: the runs of its aaai branch, in the three folders
           that scaling/README.md asks for.
S4R        wiki_game/data/swarm4round: the 4-round reference runs, where the hackathon repository keeps them.
RUNS       runs/: runs that are in none of these (see runs/README.md).
RES / name the result folder `name`, from whichever of these holds it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALING = ROOT / "scaling"
STAGE1 = SCALING / "results"
AAAI = [SCALING / "aaai", SCALING / "aaai_or", SCALING / "aaai_claude"]
S4R = ROOT / "wiki_game" / "data" / "swarm4round"
RUNS = ROOT / "runs"


class _Results:
    """`RES / "swarm_x"` gives the folder swarm_x from scaling/results, the aaai folders, wiki_game/data or runs/."""

    def __truediv__(self, name):
        for base in (STAGE1, *AAAI, S4R.parent, RUNS):
            if (base / name).exists():
                return base / name
        return STAGE1 / name


RES = _Results()
