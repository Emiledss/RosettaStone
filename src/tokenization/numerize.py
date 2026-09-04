"""Numérisation & padding (étape 5) : phrases -> tenseurs prêts pour le modèle.

`TranslationDataset` encode chaque phrase `[<sos>] + ids + [<eos>]`, tronque et
padde à `max_len` (indice `<pad>`=0). `make_dataloaders` construit les
`DataLoader` train/val/test ; si `max_len=None`, sa valeur est **dérivée
automatiquement** du p99 des longueurs en tokens mesurées *après* encodage sur
le TRAIN (+ 2 bornes `<sos>`/`<eos>`) -- indispensable car les sous-mots
allongent les séquences par rapport aux mots entiers (docs/plan-seq2seq.md,
étape 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenization.vocab_builder import Vocabulary

PERCENTILE_MAX_LEN = 99  # p99 des longueurs en tokens (étape 5)
BORNES_SOS_EOS = 2  # <sos> + <eos>


class TranslationDataset(Dataset):
    """Dataset PyTorch : source FR / cible EN, encodées `[<sos>] + ids + [<eos>]`,
    tronquées et paddées (`<pad>`=0) à `max_len`."""

    def __init__(
        self,
        df: pd.DataFrame,
        fr_vocab: Vocabulary,
        en_vocab: Vocabulary,
        max_len: int,
    ) -> None:
        self.max_len = max_len
        self.src_tensors = [self._numerize(text, fr_vocab) for text in df["fr"]]
        self.tgt_tensors = [self._numerize(text, en_vocab) for text in df["en"]]

    def _numerize(self, text: str, vocab: Vocabulary) -> torch.Tensor:
        ids = [vocab.sos_id, *vocab.encode(text), vocab.eos_id]
        ids = ids[: self.max_len]
        ids = ids + [vocab.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.src_tensors[idx], self.tgt_tensors[idx]


def _encoded_lengths(df: pd.DataFrame, fr_vocab: Vocabulary, en_vocab: Vocabulary) -> list[int]:
    """Longueurs en tokens (avec bornes `<sos>`/`<eos>`) de toutes les phrases
    FR et EN d'un split -- mesure faite *après* encodage (étape 5)."""
    longueurs_fr = [len(fr_vocab.encode(t)) + BORNES_SOS_EOS for t in df["fr"]]
    longueurs_en = [len(en_vocab.encode(t)) + BORNES_SOS_EOS for t in df["en"]]
    return longueurs_fr + longueurs_en


def derive_max_len(df: pd.DataFrame, fr_vocab: Vocabulary, en_vocab: Vocabulary) -> int:
    """Dérive `max_len` (en tokens) : p99 des longueurs encodées (FR et EN) sur
    `df`, + 2 bornes `<sos>`/`<eos>` déjà comptées dans `_encoded_lengths`."""
    longueurs = _encoded_lengths(df, fr_vocab, en_vocab)
    p99 = int(np.ceil(np.percentile(longueurs, PERCENTILE_MAX_LEN)))
    return max(p99, BORNES_SOS_EOS + 1)


def make_dataloaders(
    splits: dict[str, pd.DataFrame],
    fr_vocab: Vocabulary,
    en_vocab: Vocabulary,
    max_len: int | None = None,
    batch_size: int = 64,
) -> dict[str, DataLoader]:
    """Construit les `DataLoader` train/val/test (shuffle=True sur le train seul).

    `max_len=None` -> dérivé automatiquement depuis le TRAIN (voir
    `derive_max_len`). Logge le `max_len` retenu et le taux de troncature par
    part (point de contrôle de l'étape 5).
    """
    if max_len is None:
        max_len = derive_max_len(splits["train"], fr_vocab, en_vocab)
        print(f"[etape 5] max_len dérivé automatiquement (p99 + bornes <sos>/<eos>): {max_len}")

    loaders: dict[str, DataLoader] = {}
    for part, df in splits.items():
        longueurs = _encoded_lengths(df, fr_vocab, en_vocab)
        n_tronquees = sum(1 for longueur in longueurs if longueur > max_len)
        taux = n_tronquees / len(longueurs) if longueurs else 0.0
        print(
            f"[etape 5] {part}: max_len={max_len}, {n_tronquees}/{len(longueurs)} "
            f"séquences tronquées ({taux:.2%})"
        )
        dataset = TranslationDataset(df, fr_vocab, en_vocab, max_len)
        loaders[part] = DataLoader(dataset, batch_size=batch_size, shuffle=(part == "train"))

    return loaders
