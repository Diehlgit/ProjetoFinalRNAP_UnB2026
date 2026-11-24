"""Testes da preparação do dataset: licenças, validação de nativas e divisão sem vazamento."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cerrado_flora.data.flora_brasil import FloraBrasil
from cerrado_flora.data.inaturalist import eligible_photos, photo_url
from cerrado_flora.data.split import (
    check_leakage,
    class_rng,
    dhash,
    group_observations,
    near_duplicate_pairs,
    split_class,
)

LICENSES = {"cc0", "cc-by", "cc-by-nc"}
URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/square.jpg"


def _photo(pid, lic, pos, **kw):
    return {"position": pos, "photo": {"id": pid, "license_code": lic, "url": URL.format(pid),
                                       "attribution": "(c) x", "flags": [], **kw}}


def _obs(rank="species", photos=()):
    return {"id": 1, "taxon": {"id": 10, "name": "Caryocar brasiliense", "rank": rank},
            "user": {"login": "u"}, "observed_on": "2020-01-01", "observation_photos": photos}


def test_license_checked_per_photo():
    o = _obs(photos=[
        _photo(1, "cc-by-nc", 0),
        _photo(2, None, 1),  # todos os direitos reservados
        _photo(3, "cc-by-sa", 2),
        _photo(4, "cc0", 3, hidden=True),
        _photo(5, "cc-by", 4, flags=[{"resolved": False}]),
        _photo(6, "cc-by", 5, flags=[{"resolved": True}]),
    ])
    rows = eligible_photos(o, LICENSES, "medium")
    assert [r["photo_id"] for r in rows] == [1, 6]
    assert rows[0]["url"].endswith("/photos/1/medium.jpg")


def test_non_species_rank_excluded():
    assert eligible_photos(_obs(rank="genus", photos=[_photo(1, "cc0", 0)]), LICENSES, "m") == []


def test_photo_url_rejects_unexpected():
    with pytest.raises(ValueError):
        photo_url("https://x/photos/1/original.jpg", "medium")


@pytest.fixture
def flora():
    cols = ["id", "taxonRank", "genus", "specificEpithet", "taxonomicStatus",
            "acceptedNameUsageID", "scientificName", "family"]
    taxon = pd.DataFrame([
        ["1", "ESPECIE", "Caryocar", "brasiliense", "NOME_ACEITO", "", "Caryocar brasiliense Cambess.", "Caryocaraceae"],
        ["2", "ESPECIE", "Tridax", "procumbens", "NOME_ACEITO", "", "Tridax procumbens L.", "Asteraceae"],
        ["3", "ESPECIE", "Velha", "sinonima", "SINONIMO", "1", "Velha sinonima Autor", "Caryocaraceae"],
        ["4", "ESPECIE", "Homo", "nimo", "NOME_ACEITO", "", "Homo nimo A.", "X"],
        ["5", "ESPECIE", "Homo", "nimo", "NOME_ACEITO", "", "Homo nimo B.", "X"],
        ["6", "ESPECIE", "Mata", "atlantica", "NOME_ACEITO", "", "Mata atlantica C.", "X"],
        ["7", "ESPECIE", "Calliandra", "dysantha", "NOME_ACEITO", "", "Calliandra dysantha Benth.", "Fabaceae"],
        ["8", "VARIEDADE", "Calliandra", "dysantha", "NOME_ACEITO", "", "C. dysantha var. dysantha", "Fabaceae"],
        ["9", "VARIEDADE", "Calliandra", "dysantha", "NOME_ACEITO", "", "C. dysantha var. opulenta", "Fabaceae"],
        ["10", "ESPECIE", "Misto", "varietal", "NOME_ACEITO", "", "Misto varietal D.", "X"],
        ["11", "VARIEDADE", "Misto", "varietal", "NOME_ACEITO", "", "Misto varietal var. a", "X"],
        ["12", "VARIEDADE", "Misto", "varietal", "NOME_ACEITO", "", "Misto varietal var. b", "X"],
    ], columns=cols)
    rem = '{{"endemism":"Não endemica","phytogeographicDomain":{}}}'
    dist = pd.DataFrame([
        ["1", "BR-DF", "Nativa", rem.format('["Cerrado","Caatinga"]')],
        ["1", "BR-GO", "Nativa", rem.format('["Cerrado","Caatinga"]')],
        ["2", "BR-DF", "Exótica", rem.format('["Cerrado"]')],
        ["6", "BR-RJ", "Nativa", rem.format('["Mata Atlântica"]')],
        ["7", "", "Nativa", '{"endemism":"Não endemica"}'],  # espécie sem domínio
        ["8", "BR-DF", "Nativa", rem.format('["Caatinga","Cerrado"]')],
        ["9", "BR-GO", "Nativa", rem.format('["Cerrado"]')],
        ["11", "BR-DF", "Nativa", rem.format('["Cerrado"]')],
        ["12", "BR-DF", "Exótica", rem.format('["Cerrado"]')],
    ], columns=["id", "locationID", "establishmentMeans", "occurrenceRemarks"])
    return FloraBrasil(taxon, dist)


def test_native_accepted_name(flora):
    c = flora.check("Caryocar brasiliense")
    assert c.match == "nome_aceito" and c.is_native("Nativa", "Cerrado")
    assert c.as_row()["occurs_df"]


def test_exotic_and_other_domain_rejected(flora):
    assert not flora.check("Tridax procumbens").is_native("Nativa", "Cerrado")
    assert not flora.check("Mata atlantica").is_native("Nativa", "Cerrado")


def test_synonym_resolves_to_accepted(flora):
    c = flora.check("Velha sinonima")
    assert c.match == "sinonimo" and c.ffb_id == "1" and c.is_native("Nativa", "Cerrado")


def test_species_without_domain_inherits_from_varieties(flora):
    c = flora.check("Calliandra dysantha")
    assert c.match == "nome_aceito+infraespecificos"
    assert c.is_native("Nativa", "Cerrado")
    assert c.states == ["BR-DF", "BR-GO"] and c.domains == ["Caatinga", "Cerrado"]
    mixed = flora.check("Misto varietal")
    assert mixed.establishment == "misto" and not mixed.is_native("Nativa", "Cerrado")


def test_ambiguous_and_missing_names(flora):
    assert flora.check("Homo nimo").match == "ambiguo"
    assert flora.check("Nao existe").match == "nao_encontrado"
    assert flora.check("Nao existe", manual={"Nao existe": {"ffb_id": "1"}}).match == "manual"


def test_split_class_presence_and_groups():
    proportions = {"train": 0.7, "val": 0.15, "test": 0.15}
    obs = list(range(100, 150))  # 50 observações, o mínimo do dataset
    groups = {o: o for o in obs}
    groups[101] = groups[102] = groups[103] = 100  # um grupo de quase-duplicatas
    for seed in range(20):
        a = split_class(groups, proportions, class_rng(seed, 7))
        assert set(a) == set(obs)
        assert set(a.values()) == {"train", "val", "test"}
        assert len({a[o] for o in (100, 101, 102, 103)}) == 1
        counts = {s: sum(v == s for v in a.values()) for s in ("val", "test")}
        assert all(7 <= c <= 11 for c in counts.values())


def test_split_is_deterministic():
    g = {o: o for o in range(60)}
    p = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert split_class(g, p, class_rng(1, 5)) == split_class(g, p, class_rng(1, 5))


def test_near_duplicates_and_grouping(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(base).resize((48, 48)).save(tmp_path / "b.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "c.png")
    h = [dhash(tmp_path / f) for f in ("a.png", "b.png", "c.png")]
    pairs = {(i, j) for i, j, _ in near_duplicate_pairs(h, 4)}
    assert pairs == {(0, 1)}
    assert group_observations([1, 2, 3, 4], [(3, 2), (4, 3)]) == {1: 1, 2: 2, 3: 2, 4: 2}


def _manifest(split_of_obs):
    rows = []
    for k, (obs, split) in enumerate(split_of_obs):
        rows.append({"photo_id": k, "observation_id": obs, "group_id": obs, "split": split,
                     "class_index": 0, "sha256": str(k)})
    return pd.DataFrame(rows)


def test_check_leakage():
    ok = [(1, "train"), (1, "train"), (2, "val"), (3, "test")]
    check_leakage(_manifest(ok))
    with pytest.raises(AssertionError, match="observation_id"):
        check_leakage(_manifest(ok + [(1, "test")]))
    with pytest.raises(AssertionError, match="ausentes"):
        check_leakage(_manifest([(1, "train"), (2, "val")]))
