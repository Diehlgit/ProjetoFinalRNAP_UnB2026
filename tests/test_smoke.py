"""Testes mínimos: o pacote é importável e os metadados de execução são gerados."""

import json

import cerrado_flora
from cerrado_flora.utils import run_metadata


def test_package_importable():
    assert cerrado_flora.__version__


def test_run_metadata_save(tmp_path):
    out = tmp_path / "meta.json"
    run_metadata.save(out, extra={"experiment": "smoke"})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["experiment"] == "smoke"
    assert "git_commit" in data
