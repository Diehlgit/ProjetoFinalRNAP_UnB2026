"""As configurações comparadas só podem diferir no modelo (regra 7: mesmas condições)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs/experiments"


def _load(name):
    return yaml.safe_load((EXPERIMENTS / name).read_text(encoding="utf-8"))


def test_baseline_and_gmnet_differ_only_in_model():
    a, b = _load("gmnet_s3_ft.yaml"), _load("starnet_s4_ft.yaml")
    for cfg in (a, b):
        cfg.pop("name")
        cfg.pop("model")
    assert a == b


def test_pretrained_entries_exist_for_each_experiment():
    pretrained = yaml.safe_load((ROOT / "configs/pretrained.yaml").read_text(encoding="utf-8"))
    for f in EXPERIMENTS.glob("*.yaml"):
        key = yaml.safe_load(f.read_text(encoding="utf-8"))["model"]["pretrained"]
        assert key in pretrained and len(pretrained[key]["sha256"]) == 64, f.name
