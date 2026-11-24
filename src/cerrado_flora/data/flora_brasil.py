"""Validação de espécies nativas do Cerrado com a Flora e Funga do Brasil (JBRJ).

Fonte: Darwin Core Archive `lista_especies_flora_brasil` do IPT do JBRJ, em versão
fixada em `configs/dataset.yaml`. Critério (decidido em 2026-11-24): `establishmentMeans`
= "Nativa" e "Cerrado" entre os domínios fitogeográficos (`occurrenceRemarks`).

Correspondência de nomes (iNaturalist → Flora e Funga), na espécie (gênero + epíteto):
1. um único nome aceito (`NOME_ACEITO`) com o mesmo nome → `nome_aceito`;
2. nenhum aceito, mas sinônimos que apontam para um único aceito → `sinonimo`;
3. mais de um candidato → `ambiguo`; nenhum → `nao_encontrado` (ambos excluídos e
   listados para revisão manual; correções vão em `native.manual` no config).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

_READ = {"sep": "\t", "quoting": 3, "dtype": str, "keep_default_na": False}


@dataclass
class NativeCheck:
    inat_name: str
    # nome_aceito | sinonimo | manual | ambiguo | nao_encontrado (+infraespecificos se herdou)
    match: str
    ffb_id: str = ""
    ffb_name: str = ""
    family: str = ""
    establishment: str = ""
    domains: list[str] = field(default_factory=list)
    endemism: str = ""
    states: list[str] = field(default_factory=list)
    note: str = ""

    def is_native(self, establishment: str, domain: str) -> bool:
        return self.establishment == establishment and domain in self.domains

    def as_row(self) -> dict:
        d = asdict(self)
        d["domains"] = ";".join(self.domains)
        d["states"] = ";".join(self.states)
        d["occurs_df"] = "BR-DF" in self.states
        return d


class FloraBrasil:
    def __init__(self, taxon: pd.DataFrame, distribution: pd.DataFrame):
        sp = taxon[taxon["taxonRank"] == "ESPECIE"].copy()
        sp["canonical"] = sp["genus"].str.strip() + " " + sp["specificEpithet"].str.strip()
        # Status vazio corresponde a nomes mal aplicados; não servem para correspondência.
        self._species = sp[sp["taxonomicStatus"].isin(["NOME_ACEITO", "SINONIMO"])]
        self._by_id = taxon.set_index("id")
        self._dist = distribution.groupby("id")

    @classmethod
    def from_dwca(cls, path: str | Path) -> FloraBrasil:
        with zipfile.ZipFile(path) as z:
            with z.open("taxon.txt") as f:
                taxon = pd.read_csv(f, **_READ)
            with z.open("distribution.txt") as f:
                dist = pd.read_csv(f, **_READ)
        return cls(taxon, dist)

    def _describe(self, inat_name: str, ffb_id: str, match: str, note: str = "") -> NativeCheck:
        t = self._by_id.loc[ffb_id]
        chk = NativeCheck(
            inat_name, match, ffb_id=ffb_id, ffb_name=t["scientificName"],
            family=t["family"], note=note,
        )
        self._fill_distribution(chk, ffb_id)
        if not chk.domains or not chk.establishment:
            self._inherit_from_infraspecific(chk, t)
        return chk

    def _distribution(self, ffb_id: str) -> tuple[str, list[str], list[str], str]:
        if ffb_id not in self._dist.groups:
            return "", [], [], ""
        d = self._dist.get_group(ffb_id)
        remarks = json.loads(d["occurrenceRemarks"].iloc[0] or "{}")
        states = sorted(s for s in d["locationID"].unique() if s)
        return (d["establishmentMeans"].iloc[0], list(remarks.get("phytogeographicDomain", [])),
                states, remarks.get("endemism", ""))

    def _fill_distribution(self, chk: NativeCheck, ffb_id: str) -> None:
        chk.establishment, chk.domains, chk.states, chk.endemism = self._distribution(ffb_id)

    def _inherit_from_infraspecific(self, chk: NativeCheck, t: pd.Series) -> None:
        """Registro da espécie sem distribuição: usa os infraespecíficos aceitos.

        Decidido em 2026-11-24: nativa apenas se **todos** os infraespecíficos com
        informação forem "Nativa"; domínios e estados = união. Exemplo: *Calliandra
        dysantha* (espécie sem distribuição; as 4 variedades são nativas do Cerrado).
        """
        infra = self._by_id[
            (self._by_id["genus"] == t["genus"])
            & (self._by_id["specificEpithet"] == t["specificEpithet"])
            & self._by_id["taxonRank"].isin(["VARIEDADE", "SUB_ESPECIE", "FORMA"])
            & (self._by_id["taxonomicStatus"] == "NOME_ACEITO")
        ]
        dists = [self._distribution(i) for i in infra.index]
        dists = [d for d in dists if d[0] and d[1]]
        if not dists:
            return
        ests = {d[0] for d in dists}
        chk.establishment = ests.pop() if len(ests) == 1 else "misto"
        chk.domains = sorted({x for d in dists for x in d[1]})
        chk.states = sorted({x for d in dists for x in d[2]})
        chk.endemism = ";".join(sorted({d[3] for d in dists if d[3]}))
        chk.match += "+infraespecificos"
        chk.note = (chk.note + " | " if chk.note else "") + (
            f"distribuição herdada de {len(dists)} infraespecíficos aceitos")

    def check(self, inat_name: str, manual: dict | None = None) -> NativeCheck:
        if manual and inat_name in manual:
            m = manual[inat_name]
            return self._describe(inat_name, str(m["ffb_id"]), "manual", m.get("note", ""))
        rows = self._species[self._species["canonical"] == inat_name]
        accepted = rows[rows["taxonomicStatus"] == "NOME_ACEITO"]
        if len(accepted) == 1:
            return self._describe(inat_name, accepted["id"].iloc[0], "nome_aceito")
        if len(accepted) > 1:
            return NativeCheck(inat_name, "ambiguo", note="homônimos aceitos: "
                               + ";".join(accepted["id"]))
        targets = sorted(set(rows["acceptedNameUsageID"]) - {""})
        if len(targets) == 1:
            return self._describe(inat_name, targets[0], "sinonimo",
                                  "sinônimo: " + ";".join(rows["scientificName"]))
        if len(targets) > 1:
            return NativeCheck(inat_name, "ambiguo", note="sinônimo de: " + ";".join(targets))
        return NativeCheck(inat_name, "nao_encontrado")
