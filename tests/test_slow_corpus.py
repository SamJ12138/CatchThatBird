"""The corpus baseline as regression tests (marker `slow`): one paced run per
clip with the default config, scored against the clip's ground truth, must
be no worse than the recorded baseline (data/samples/corpus/baseline.json,
docs/corpus-baseline.md) by more than the tolerance. Skipped for clips that
have not been fetched (python scripts/fetch_clips.py)."""
from __future__ import annotations

import json

import pytest

from scripts import corpus_eval as ce
from scripts import fetch_clips as fc
from tests.conftest import ROOT

BASELINE = json.loads((fc.CORPUS / "baseline.json").read_text(encoding="utf-8"))
WORSE_IF_HIGHER = ("missed", "split", "merged", "false")


@pytest.mark.slow
@pytest.mark.parametrize("clip", fc.CLIPS, ids=lambda c: c.name)
def test_clip_is_no_worse_than_its_baseline(clip, tmp_path) -> None:
    if not clip.out.exists():
        pytest.skip(f"{clip.name} not fetched: python scripts/fetch_clips.py {clip.name}")
    if not (ROOT / "yolov8n.pt").exists():
        pytest.skip("yolov8n.pt not present (ultralytics would download it)")
    runs = BASELINE["clips"][clip.name]["runs"]
    tol = BASELINE["tolerance"]
    result = ce.run_clip(clip.name, tmp_path)
    worse = {k: (result[k], max(r[k] for r in runs)) for k in WORSE_IF_HIGHER
             if result[k] > max(r[k] for r in runs) + tol}
    assert not worse, f"{clip.name} run {result['run_id']}: {worse} (got, baseline max)"
    assert result["max_yolo_per_s"] <= BASELINE["max_yolo_per_s_limit"], result
