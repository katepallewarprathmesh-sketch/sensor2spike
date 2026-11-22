"""Every Python block in the README must actually run.

Documentation that has drifted from the code is worse than no documentation: it
is a confident lie. These tests execute the examples in order against a real
synthetic dataset, so a broken snippet fails the build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import sensor2spike
from sensor2spike.demo import generate_demo_dataset

README = Path(__file__).resolve().parents[1] / "README.md"


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_has_code_blocks():
    assert len(_blocks()) >= 3


def test_every_readme_block_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    # The README writes data/radar.npz in step 1; supply it the way a user would.
    generate_demo_dataset(n_trials=24, seed=7).save("data/radar.npz")

    namespace: dict[str, object] = {}
    for index, block in enumerate(_blocks(), start=1):
        # 40 trials is the documented default; trim it so the test stays quick
        # without touching anything else about the snippet.
        code = block.replace("n_trials=40", "n_trials=8")
        try:
            exec(compile(code, f"<README block {index}>", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - only on a real regression
            pytest.fail(f"README block {index} failed: {exc!r}")

    # The snippets claim these objects; check the claims.
    best = namespace["best"]
    assert best.best is not None
    assert "my_encoder" in {e["name"] for e in sensor2spike.list_encoders()}
    assert 0.0 <= best.best.accuracy <= 1.0
    assert (tmp_path / "out" / "result.json").is_file()
    assert (tmp_path / "out" / "best_config.json").is_file()
