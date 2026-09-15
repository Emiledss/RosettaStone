"""Numérisation & padding dynamique (étape 5) : phrases -> tenseurs prêts pour
le modèle.

`TranslationDataset` encode chaque phrase `[<sos>] + ids + [<eos>]`, tronquée
à `max_len` (garde-fou, indice `<pad>`=0) mais **non paddée** : chaque exemple
reste un tenseur 1-D de longueur variable. Le padding est reporté au niveau du
BATCH (`collate_fn`), à sa propre longueur maximale -- source et cible
indépendamment, puisque rien n'impose qu'elles aient la même longueur.

`make_dataloaders` construit les `DataLoader` train/val/test ; si
`max_len=None`, sa valeur est **dérivée automatiquement** du MAXIMUM des
longueurs en tokens mesurées *après* encodage sur le TRAIN (+ 2 bornes
`<sos>`/`<eos>`) -- ce n'est plus qu'un garde-fou contre une aberration
(le nettoyage de l'étape 1 a déjà écarté les phrases > 40 mots), plus un
levier de performance : c'est le padding dynamique par batch qui absorbe
l'hétérogénéité des longueurs (docs/plan-seq2seq.md, étape 5).

Le TRAIN est en plus regroupé par longueur voisine (`bucket_by_length=True`,
défaut) via `LengthGroupedBatchSampler` : les batchs sont formés à partir
d'indices triés par longueur -- ce qui réduit fortement le padding moyen --
puis leur ORDRE est mélangé à chaque epoch pour ne pas présenter les phrases
courtes puis les longues dans un ordre systématique. val/test restent en
ordre séquentiel (jamais regroupés ni mélangés) : `src/evaluation/generate.py`
associe les hypothèses aux références PAR INDICE, une réorganisation de l'ordre
d'itération y casserait silencieusement la correspondance.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.tokenization.vocab_builder import Vocabulary

BORNES_SOS_EOS = 2  # <sos> + <eos>
PAD_ID = 0  # src.tokenization.vocab_builder.Vocabulary.pad_id


class TranslationDataset(Dataset):
    """Dataset PyTorch : source FR / cible EN, encodées `[<sos>] + ids +
    [<eos>]`, tronquées (garde-fou) à `max_len`, **non paddées** -- le padding
    est appliqué par batch (`collate_fn`), pas ici."""

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
        ids = ids[: self.max_len]  # garde-fou contre une aberration, jamais atteint en pratique
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.src_tensors[idx], self.tgt_tensors[idx]

    def lengths(self) -> list[int]:
        """Longueur `max(len(src), len(tgt))` par exemple -- clé de tri du
        regroupement par longueur (`LengthGroupedBatchSampler`) : c'est le
        MAX des deux langues qui gouverne le coût d'un batch."""
        return [
            max(len(src), len(tgt)) for src, tgt in zip(self.src_tensors, self.tgt_tensors)
        ]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Padde un batch de couples `(src, tgt)` (tenseurs 1-D de longueurs
    quelconques) à SON PROPRE maximum -- source et cible **indépendamment**,
    `pad_id`=0. Renvoie `(src, tgt)` de formes `(batch, max_src_du_batch)` et
    `(batch, max_tgt_du_batch)`."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(list(src_batch), batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(list(tgt_batch), batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_padded


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Regroupe les indices d'un dataset en batchs homogènes en longueur.

    Les indices sont triés une fois par `lengths[i]` (le max des deux langues,
    cf. `TranslationDataset.lengths`) puis découpés en tranches de
    `batch_size`. À chaque nouvel appel de `__iter__` (= chaque epoch sous
    `DataLoader`), l'ORDRE des batchs est remélangé (`shuffle=True`, défaut) --
    le contenu de chaque batch, lui, ne change pas : c'est ce qui évite de
    présenter systématiquement les phrases courtes puis les longues au sein
    d'une epoch, tout en gardant des batchs homogènes.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, shuffle: bool = True) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        indices_triees = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        self._batches: list[list[int]] = [
            indices_triees[i : i + batch_size] for i in range(0, len(indices_triees), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        ordre = list(range(len(self._batches)))
        if self.shuffle and ordre:
            ordre = torch.randperm(len(ordre)).tolist()
        for i in ordre:
            yield self._batches[i]

    def __len__(self) -> int:
        return len(self._batches)


def _encoded_lengths(df: pd.DataFrame, fr_vocab: Vocabulary, en_vocab: Vocabulary) -> list[int]:
    """Longueurs en tokens (avec bornes `<sos>`/`<eos>`) de toutes les phrases
    FR et EN d'un split -- mesure faite *après* encodage (étape 5)."""
    longueurs_fr = [len(fr_vocab.encode(t)) + BORNES_SOS_EOS for t in df["fr"]]
    longueurs_en = [len(en_vocab.encode(t)) + BORNES_SOS_EOS for t in df["en"]]
    return longueurs_fr + longueurs_en


def derive_max_len(df: pd.DataFrame, fr_vocab: Vocabulary, en_vocab: Vocabulary) -> int:
    """Dérive `max_len` (en tokens), un simple GARDE-FOU (le padding dynamique
    par batch absorbe l'hétérogénéité, `max_len` n'est plus un levier de
    performance) : le MAXIMUM des longueurs encodées (FR et EN) sur
    `df`, + 2 bornes `<sos>`/`<eos>` déjà comptées dans `_encoded_lengths`."""
    longueurs = _encoded_lengths(df, fr_vocab, en_vocab)
    maximum = max(longueurs) if longueurs else 0
    return max(maximum, BORNES_SOS_EOS + 1)


def make_dataloaders(
    splits: dict[str, pd.DataFrame],
    fr_vocab: Vocabulary,
    en_vocab: Vocabulary,
    max_len: int | None = None,
    batch_size: int = 64,
    bucket_by_length: bool = True,
) -> dict[str, DataLoader]:
    """Construit les `DataLoader` train/val/test, à padding **dynamique par
    batch** (`collate_fn`) : chaque batch est paddé à son propre maximum, pas
    à un `max_len` global.

    `max_len=None` -> dérivé automatiquement depuis le TRAIN (voir
    `derive_max_len`) -- un garde-fou, plus un levier de performance. Logge le
    `max_len` retenu et le taux de troncature par part (doit valoir 0 en
    pratique : le nettoyage de l'étape 1 a déjà écarté les aberrations).

    `bucket_by_length=True` (défaut) : le TRAIN est regroupé par longueur
    voisine (`LengthGroupedBatchSampler`, ordre des batchs remélangé à chaque
    epoch) -- réduit le padding moyen sans corréler les exemples au sein d'une
    epoch. `bucket_by_length=False` retombe sur des batchs aléatoires
    classiques (padding dynamique conservé) -- réserve si le regroupement
    dégrade l'entraînement (diversité du gradient appauvrie par des batchs
    homogènes).

    val/test restent TOUJOURS en ordre séquentiel (`shuffle=False`, pas de
    regroupement) : `src/evaluation/generate.py` associe les hypothèses aux
    références par indice, une réorganisation de l'ordre d'itération casserait
    silencieusement cette correspondance.
    """
    if max_len is None:
        max_len = derive_max_len(splits["train"], fr_vocab, en_vocab)
        print(f"[etape 5] max_len dérivé automatiquement (maximum observé sur le train + bornes <sos>/<eos>, garde-fou): {max_len}")

    loaders: dict[str, DataLoader] = {}
    for part, df in splits.items():
        longueurs = _encoded_lengths(df, fr_vocab, en_vocab)
        n_tronquees = sum(1 for longueur in longueurs if longueur > max_len)
        taux = n_tronquees / len(longueurs) if longueurs else 0.0
        print(
            f"[etape 5] {part}: max_len={max_len} (garde-fou, padding dynamique par batch), "
            f"{n_tronquees}/{len(longueurs)} séquences tronquées ({taux:.2%})"
        )
        dataset = TranslationDataset(df, fr_vocab, en_vocab, max_len)

        if part == "train" and bucket_by_length:
            batch_sampler = LengthGroupedBatchSampler(dataset.lengths(), batch_size, shuffle=True)
            loaders[part] = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
        else:
            loaders[part] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(part == "train"),  # train non regroupé -> mélange aléatoire classique
                collate_fn=collate_fn,
            )

    return loaders
