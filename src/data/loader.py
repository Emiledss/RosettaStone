"""Chargement du corpus brut Tatoeba EN-FR (étape 0).

Transpose en snake_case la logique de chargement déjà validée dans le notebook
AED (cellule 5) : deux chemins de chargement, avec bascule automatique via la
stratégie "auto".

- `load_from_huggingface` : chemin officiel via `datasets` (nécessite
  `datasets<4.0` et `trust_remote_code=True`, dataset "à script").
- `load_from_opus` : repli qui télécharge (ou réutilise) directement l'archive
  OPUS-Tatoeba et reproduit le même prétraitement que le script HF (`.strip()`
  ligne par ligne).

⚠️ L'archive `data/tatoeba_en-fr_v2021-07-22.zip` est déjà en cache : le repli ne
doit jamais relancer de téléchargement dans ce cas.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset et paire de langues fixés par le projet (cf. docs/plan-seq2seq.md).
DATASET_NAME = "Helsinki-NLP/tatoeba"
LANG1 = "en"
LANG2 = "fr"
CORPUS_VERSION = "v2021-07-22"

OPUS_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/{version}/moses/{l1}-{l2}.txt.zip"
OPUS_MEMBER = "Tatoeba.{l1}-{l2}.{lang}"


def load_from_huggingface(name: str, lang1: str, lang2: str) -> pd.DataFrame:
    """Chargement officiel via `datasets` (nécessite datasets<4.0 + trust_remote_code)."""
    from datasets import load_dataset

    ds = load_dataset(name, lang1=lang1, lang2=lang2, trust_remote_code=True)
    split = ds["train"]
    translations = split["translation"]
    return pd.DataFrame(
        {
            "id": split["id"],
            lang1: [t[lang1] for t in translations],
            lang2: [t[lang2] for t in translations],
        }
    )


def download_opus_archive(lang1: str, lang2: str, version: str, data_dir: str) -> Path:
    """Télécharge (et met en cache) l'archive OPUS-Tatoeba utilisée par le script HF."""
    dest_path = Path(data_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    archive = dest_path / f"tatoeba_{lang1}-{lang2}_{version}.zip"
    if archive.exists():
        print(f"Archive déjà en cache: {archive} ({archive.stat().st_size / 1e6:.1f} Mo)")
        return archive
    url = OPUS_URL.format(version=version, l1=lang1, l2=lang2)
    print(f"Téléchargement: {url}")
    urllib.request.urlretrieve(url, archive)
    print(f"Archive écrite: {archive} ({archive.stat().st_size / 1e6:.1f} Mo)")
    return archive


def load_from_opus(lang1: str, lang2: str, version: str, data_dir: str) -> pd.DataFrame:
    """Repli sans `datasets` : lit les deux fichiers Moses alignés ligne à ligne."""
    archive = download_opus_archive(lang1, lang2, version, data_dir)
    with zipfile.ZipFile(archive) as zf:
        member1 = OPUS_MEMBER.format(l1=lang1, l2=lang2, lang=lang1)
        member2 = OPUS_MEMBER.format(l1=lang1, l2=lang2, lang=lang2)
        lines1 = zf.read(member1).decode("utf-8").splitlines()
        lines2 = zf.read(member2).decode("utf-8").splitlines()
    if len(lines1) != len(lines2):
        raise ValueError(f"Corpus désaligné: {len(lines1)} lignes {lang1} vs {len(lines2)} lignes {lang2}")
    # Le script HF applique un .strip() sur chaque ligne : on reproduit ce comportement.
    return pd.DataFrame(
        {
            "id": np.arange(len(lines1)).astype(str),
            lang1: [s.strip() for s in lines1],
            lang2: [s.strip() for s in lines2],
        }
    )


def load_raw_corpus(strategy: str = "auto", data_dir: str = "data") -> pd.DataFrame:
    """Charge le corpus brut Tatoeba EN-FR selon la stratégie choisie.

    strategy : "hf" (via `datasets`), "opus" (repli direct sur l'archive), ou
    "auto" (tente `hf`, puis bascule sur `opus` en cas d'échec).
    Renvoie un DataFrame `[id, en, fr]`. Logge le nombre de paires et de NaN
    (points de contrôle de l'étape 0 : ~264 905 paires, 0 NaN).
    """
    strategy = strategy.lower()
    if strategy == "hf":
        df = load_from_huggingface(DATASET_NAME, LANG1, LANG2)
    elif strategy == "opus":
        df = load_from_opus(LANG1, LANG2, CORPUS_VERSION, data_dir)
    elif strategy == "auto":
        try:
            df = load_from_huggingface(DATASET_NAME, LANG1, LANG2)
        except Exception as exc:  # noqa: BLE001 - repli volontaire sur tout échec HF
            print(f"Chargement via `datasets` indisponible ({type(exc).__name__}: {exc})")
            print("--> repli sur l'archive OPUS brute (source identique).")
            df = load_from_opus(LANG1, LANG2, CORPUS_VERSION, data_dir)
    else:
        raise ValueError(f"strategy inconnue: {strategy}")

    df = df[["id", LANG1, LANG2]].reset_index(drop=True)
    n_nan = int(df.isna().sum().sum())
    print(f"[etape 0] {len(df)} paires chargées, {n_nan} valeurs manquantes (strategy={strategy})")
    return df
