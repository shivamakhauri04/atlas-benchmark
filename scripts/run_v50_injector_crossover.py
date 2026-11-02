"""Mean change in trace length from the shared anomaly injector, as a function of clean length.

`_inject_anomaly` keeps a random prefix of between half and all of the trace and appends a 2-3 step
pattern plus up to 2 tail actions, so short traces get longer and long traces get shorter.
"""

import json
import random
from pathlib import Path

import numpy as np

from data.schema import Action
from data.trace_loader import _inject_anomaly

OUT = Path("output/tables/v50_injector_crossover.json")
LENGTHS = (2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 30, 50)
DRAWS = 4000


def main() -> None:
    rng = random.Random(0)
    deltas: dict[str, float] = {}
    for n in LENGTHS:
        acts = [Action(tool="t", args="", context="", label=0) for _ in range(n)]
        deltas[str(n)] = float(np.mean([len(_inject_anomaly(list(acts), rng)) - n
                                        for _ in range(DRAWS)]))
    sign_change = next(int(n) for n, v in deltas.items() if v < 0)
    OUT.write_text(json.dumps({"draws": DRAWS, "seed": 0, "mean_delta": deltas,
                               "first_length_with_negative_delta": sign_change}, indent=2))
    print(json.dumps(deltas), "first negative at", sign_change)


if __name__ == "__main__":
    main()
