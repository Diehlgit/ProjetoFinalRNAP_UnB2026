"""Seleção por classe, agrupamento de quase-duplicatas e divisão treino/validação/teste.

Unidade de divisão: a observação do iNaturalist. Observações que compartilham fotos
quase idênticas (dHash) são unidas num mesmo grupo e vão juntas para a mesma divisão.
A divisão é feita **classe a classe**, garantindo ao menos uma observação de cada classe
em validação e em teste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test")


def class_rng(seed: int, taxon_id: int) -> np.random.Generator:
    """Gerador por classe: incluir/excluir uma classe não altera o sorteio das outras."""
    return np.random.default_rng([seed, int(taxon_id)])


def cap_observations(obs_ids: list[int], max_obs: int, rng: np.random.Generator) -> list[int]:
    ids = sorted(obs_ids)
    if len(ids) <= max_obs:
        return ids
    return sorted(rng.choice(ids, size=max_obs, replace=False).tolist())


def dhash(path, size: int = 8) -> int:
    """Hash de diferença (64 bits) para detectar fotos quase idênticas."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS),
                       dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_pairs(hashes: list[int], max_distance: int) -> list[tuple[int, int, int]]:
    """Pares (i, j, distância) com distância de Hamming ≤ `max_distance`."""
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        d = np.bitwise_count(h[i] ^ h[i + 1:])
        for j in np.nonzero(d <= max_distance)[0]:
            pairs.append((i, i + 1 + int(j), int(d[j])))
    return pairs


def group_observations(obs_ids: list[int], links: list[tuple[int, int]]) -> dict[int, int]:
    """União-busca: observação → id do grupo (menor id de observação do componente)."""
    parent = {o: o for o in obs_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {o: find(o) for o in obs_ids}


def split_class(
    groups: dict[int, int], proportions: dict[str, float], rng: np.random.Generator
) -> dict[int, str]:
    """Divide as observações de uma classe; grupos inteiros vão para a mesma divisão.

    Metas em número de observações: val e teste = max(1, round(p · n)); o restante, treino.
    Os grupos são sorteados e preenchem teste, depois validação; o que sobra vai para treino.
    """
    by_group: dict[int, list[int]] = {}
    for o, g in sorted(groups.items()):
        by_group.setdefault(g, []).append(o)
    n = len(groups)
    target = {s: max(1, round(proportions[s] * n)) for s in ("test", "val")}
    if target["test"] + target["val"] >= n:
        raise ValueError(f"classe com {n} observações é pequena demais para dividir")
    order = list(by_group)
    rng.shuffle(order)
    filled = {"test": 0, "val": 0}
    assign: dict[int, str] = {}
    for g in order:
        members = by_group[g]
        split = next((s for s in ("test", "val") if filled[s] < target[s]), "train")
        if split != "train":
            filled[split] += len(members)
        for o in members:
            assign[o] = split
    return assign


def check_leakage(images: pd.DataFrame) -> None:
    """Falha se uma foto, observação ou grupo aparecer em mais de uma divisão."""
    if images["photo_id"].duplicated().any():
        raise AssertionError("photo_id repetido no manifesto")
    for col in ("observation_id", "group_id", "sha256"):
        n = images.groupby(col)["split"].nunique()
        if (n > 1).any():
            raise AssertionError(f"vazamento: {col} em mais de uma divisão: {list(n[n > 1].index)[:5]}")
    missing = images.groupby("class_index")["split"].nunique()
    if (missing < len(SPLITS)).any():
        raise AssertionError(f"classes ausentes de alguma divisão: {list(missing[missing < 3].index)}")


def observer_overlap(images: pd.DataFrame) -> pd.DataFrame:
    """Por classe: fração das observações de val/teste cujo observador também está no treino."""
    obs = images.drop_duplicates("observation_id")
    rows = []
    for (cls, name), d in obs.groupby(["class_index", "scientific_name"]):
        train_obs = set(d.loc[d["split"] == "train", "observer"])
        row = {"class_index": cls, "scientific_name": name}
        for s in ("val", "test"):
            e = d[d["split"] == s]
            row[f"{s}_obs_observer_in_train"] = float(e["observer"].isin(train_obs).mean())
        rows.append(row)
    return pd.DataFrame(rows)
