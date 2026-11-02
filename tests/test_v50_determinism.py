"""Injected corpora must not depend on Python's per-process string-hash seed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = """
from data import domain_loader as dl
seen = []
orig = dl._inject_api_misuse
def spy(actions, all_tools, rng):
    if not seen:
        seen.append(list(all_tools[:50]))
    return orig(actions, all_tools, rng)
dl._inject_api_misuse = spy
dl.load_{name}_domain(seed=42)
print(repr(seen[0]))
"""


def _vocabulary(name: str, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": f"{ROOT}/src:{ROOT}/../../lib"}
    out = subprocess.run([sys.executable, "-c", SNIPPET.format(name=name)], env=env, cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode:
        pytest.skip(f"{name} data unavailable: {out.stderr[-200:]}")
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("name", ["apibank", "toolace"])
def test_wrong_tool_vocabulary_is_independent_of_hash_seed(name):
    assert _vocabulary(name, "1") == _vocabulary(name, "2")
