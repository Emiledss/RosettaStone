"""Caractérisation par config de tokenisation, avant Optuna (étape 5bis).

Calcule, pour un vocabulaire déjà construit (étape 4) et une série de phrases
(typiquement le TRAIN), les 5 stats clés du plan : longueur en tokens, ratio
tokens/mots (fragmentation), vocabulaire effectif, taux d'OOV et couverture.
Fournit aussi les données pour les figures de comparaison BPE vs Unigram
(Zipf, longueur des tokens en caractères).

Ce module ne construit rien : il mesure un `Vocabulary` déjà entraîné.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.data.text import count_words
from src.tokenization.vocab_builder import Vocabulary

SP_WORD_BOUNDARY = "▁"  # "▁" -- marqueur de début de mot SentencePiece


def _is_special(vocab: Vocabulary, token_id: int) -> bool:
    return token_id in (vocab.pad_id, vocab.unk_id, vocab.sos_id, vocab.eos_id)


def _piece_text(vocab: Vocabulary, token_id: int) -> str:
    """Forme texte d'un id, marqueur de début de mot SentencePiece retiré."""
    return vocab.id_to_piece(token_id).replace(SP_WORD_BOUNDARY, "")


def _encode_corpus(vocab: Vocabulary, series: pd.Series) -> list[list[int]]:
    """Encode chaque phrase de `series` une seule fois (réutilisé par les stats)."""
    return [vocab.encode(str(text)) for text in series]


def token_length_stats(
    vocab: Vocabulary, series: pd.Series, _encoded: list[list[int]] | None = None
) -> dict[str, float]:
    """Longueur en tokens par phrase : moyenne, médiane, p95, p99, max.

    `_encoded` (privé) permet de réutiliser un encodage déjà calculé (voir
    `tokenization_stats_row`) plutôt que de ré-encoder tout le corpus.
    """
    encoded = _encoded if _encoded is not None else _encode_corpus(vocab, series)
    longueurs = np.array([len(ids) for ids in encoded], dtype=float)
    if longueurs.size == 0:
        return {"moyenne": 0.0, "mediane": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "moyenne": float(longueurs.mean()),
        "mediane": float(np.median(longueurs)),
        "p95": float(np.percentile(longueurs, 95)),
        "p99": float(np.percentile(longueurs, 99)),
        "max": float(longueurs.max()),
    }


def fragmentation_ratio(
    vocab: Vocabulary, series: pd.Series, lang: str, _encoded: list[list[int]] | None = None
) -> float:
    """`nb_tokens / nb_mots` agrégé sur `series` (1.0 = mots entiers, >1 = sous-mots).

    `nb_mots` compté avec `count_words` (règle de l'apostrophe, cohérent avec
    le découpage utilisé par `WordVocabulary`). `_encoded` : voir `token_length_stats`.
    """
    encoded = _encoded if _encoded is not None else _encode_corpus(vocab, series)
    nb_tokens = sum(len(ids) for ids in encoded)
    nb_mots = sum(count_words(text, lang) for text in series)
    return nb_tokens / nb_mots if nb_mots else float("nan")


def oov_rate(
    vocab: Vocabulary, series: pd.Series, _encoded: list[list[int]] | None = None
) -> float:
    """Fraction des tokens encodés égaux à `<unk>` (~0 attendu pour les sous-mots).
    `_encoded` : voir `token_length_stats`."""
    encoded = _encoded if _encoded is not None else _encode_corpus(vocab, series)
    nb_tokens = sum(len(ids) for ids in encoded)
    nb_unk = sum(ids.count(vocab.unk_id) for ids in encoded)
    return nb_unk / nb_tokens if nb_tokens else float("nan")


def effective_vocab_size(
    vocab: Vocabulary, series: pd.Series, _encoded: list[list[int]] | None = None
) -> int:
    """Nombre de tokens distincts réellement utilisés en encodant `series`.
    `_encoded` : voir `token_length_stats`."""
    encoded = _encoded if _encoded is not None else _encode_corpus(vocab, series)
    used: set[int] = set()
    for ids in encoded:
        used.update(ids)
    return len(used)


def zipf_frequencies(vocab: Vocabulary, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Rangs et fréquences (triées décroissant) des tokens de `series`.

    Les tokens spéciaux (0-3) sont exclus : ils ne font pas partie du lexique
    observé, leur inclusion fausserait la courbe de Zipf.
    """
    counter: Counter[int] = Counter()
    for ids in _encode_corpus(vocab, series):
        counter.update(tid for tid in ids if not _is_special(vocab, tid))
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    return ranks, freqs


def token_char_lengths(vocab: Vocabulary, series: pd.Series) -> np.ndarray:
    """Longueur en caractères de chaque occurrence de token (tokens spéciaux
    exclus, marqueur de début de mot SentencePiece "▁" retiré avant mesure)."""
    lengths: list[int] = []
    cache: dict[int, int] = {}
    for ids in _encode_corpus(vocab, series):
        for tid in ids:
            if _is_special(vocab, tid):
                continue
            if tid not in cache:
                cache[tid] = len(_piece_text(vocab, tid))
            lengths.append(cache[tid])
    return np.array(lengths, dtype=int)


def tokenization_stats_row(
    vocab: Vocabulary,
    series: pd.Series,
    lang: str,
    config_name: str,
    part: str = "train",
) -> dict[str, float | str | int]:
    """Une ligne du tableau des 5 stats clés (étape 5bis) pour (config, lang, part).

    Encode `series` une seule fois puis réutilise cet encodage pour toutes les
    sous-stats (au lieu de 4 passes indépendantes) -- important sur le corpus
    réel (train ~185k phrases x 6 configs x 2 langues).
    """
    encoded = _encode_corpus(vocab, series)
    longueur = token_length_stats(vocab, series, _encoded=encoded)
    oov = oov_rate(vocab, series, _encoded=encoded)
    ratio = fragmentation_ratio(vocab, series, lang, _encoded=encoded)
    vocab_effectif = effective_vocab_size(vocab, series, _encoded=encoded)
    return {
        "config": config_name,
        "lang": lang,
        "part": part,
        "longueur_moyenne": longueur["moyenne"],
        "longueur_mediane": longueur["mediane"],
        "longueur_p95": longueur["p95"],
        "longueur_p99": longueur["p99"],
        "longueur_max": longueur["max"],
        "ratio_tokens_mots": ratio,
        "vocab_effectif": vocab_effectif,
        "taux_oov": oov,
        "couverture": 1.0 - oov if not np.isnan(oov) else float("nan"),
    }


def build_stats_table(
    splits: dict[str, pd.DataFrame],
    vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
    parts: tuple[str, ...] = ("train",),
) -> pd.DataFrame:
    """Assemble le tableau des 5 stats clés pour plusieurs configs x langues x parts.

    `vocabs` : `{config_name: (fr_vocab, en_vocab)}`, déjà construits (étape 4).
    """
    rows = []
    for config_name, (fr_vocab, en_vocab) in vocabs.items():
        for part in parts:
            df = splits[part]
            rows.append(tokenization_stats_row(fr_vocab, df["fr"], "fr", config_name, part))
            rows.append(tokenization_stats_row(en_vocab, df["en"], "en", config_name, part))
    return pd.DataFrame(rows)
