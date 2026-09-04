"""Vocabulaires FR/EN construits sur le TRAIN seul (étape 4).

Deux implémentations exposant la **même interface** (`encode`/`decode`/`__len__`,
indices spéciaux `<pad>`=0, `<unk>`=1, `<sos>`=2, `<eos>`=3) :

- `WordVocabulary` : mots entiers, découpage via `tokenize_words` de
  `src.data.text` (règle de l'apostrophe d'élision, ADR-0002/ADR-0004) --
  jamais `str.split()`. Configs `full` (couverture 100 %) et `words95`
  (coupée à 95 % de couverture des occurrences).
- `SentencePieceVocabulary` : sous-mots (BPE/Unigram), un modèle entraîné par
  langue, mis en cache sur disque. Configs `bpe4k`, `bpe8k`, `unigram4k`,
  `unigram8k`.

`build_vocabs(train_df, config)` est le point d'entrée unique : il renvoie
`(fr_vocab, en_vocab)`, quel que soit le type de vocabulaire choisi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import pandas as pd
import sentencepiece as spm

from src.data.text import tokenize_words

# Indices spéciaux imposés par le projet (docs/plan-seq2seq.md, étape 4).
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]  # ordre = indices 0-3

# Les 6 configurations de tokenisation comparées (ADR-0010).
TOKENIZATION_CONFIGS: dict[str, dict[str, object]] = {
    "full": {"kind": "word", "coverage": 1.0},
    "words95": {"kind": "word", "coverage": 0.95},
    "bpe4k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 4000},
    "bpe8k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 8000},
    "unigram4k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 4000},
    "unigram8k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 8000},
}


class Vocabulary(ABC):
    """Interface commune aux vocabulaires (mots entiers ou SentencePiece).

    `encode` renvoie des indices **sans** `<sos>`/`<eos>` (c'est la
    numérisation, étape 5, qui les ajoute). `decode` ignore `<pad>`, `<sos>`
    et `<eos>`.
    """

    pad_id = 0
    unk_id = 1
    sos_id = 2
    eos_id = 3

    config_name: str
    lang: str

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode `text` en indices, sans bornes `<sos>`/`<eos>` (OOV -> `<unk>`)."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Décode une liste d'indices en texte, en ignorant `<pad>`/`<sos>`/`<eos>`."""

    @abstractmethod
    def id_to_piece(self, token_id: int) -> str:
        """Forme texte d'un indice de token (utilisé par les stats, étape 5bis)."""

    @abstractmethod
    def __len__(self) -> int:
        """Taille du vocabulaire."""


class WordVocabulary(Vocabulary):
    """Vocabulaire mots entiers (configs `full` et `words95`)."""

    def __init__(self, config_name: str, lang: str) -> None:
        self.config_name = config_name
        self.lang = lang
        self._token_to_id: dict[str, int] = dict(
            zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS)))
        )
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}
        self.coverage_atteinte: float = 0.0

    @classmethod
    def fit(
        cls, texts: pd.Series, lang: str, config_name: str, coverage: float
    ) -> WordVocabulary:
        """Construit le vocabulaire sur `texts` (le TRAIN, anti-fuite étape 4).

        `coverage=1.0` -> tous les mots. `coverage<1.0` -> tri par fréquence
        décroissante, coupe dès que le cumul des occurrences atteint `coverage`.
        """
        vocab = cls(config_name, lang)
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize_words(str(text), lang))

        total = sum(counts.values())
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        if coverage >= 1.0:
            selected = ordered
        else:
            cible = coverage * total
            cumul = 0
            selected = []
            for mot, freq in ordered:
                selected.append((mot, freq))
                cumul += freq
                if cumul >= cible:
                    break

        for mot, _freq in selected:
            idx = len(vocab._token_to_id)
            vocab._token_to_id[mot] = idx
            vocab._id_to_token[idx] = mot

        vocab.coverage_atteinte = (
            sum(freq for _, freq in selected) / total if total else 0.0
        )

        print(
            f"[etape 4] vocab {config_name}/{lang}: {len(vocab)} mots, "
            f"couverture réelle {vocab.coverage_atteinte:.2%} (sur {total} occurrences)"
        )
        return vocab

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id.get(tok, self.unk_id) for tok in tokenize_words(str(text), self.lang)]

    def decode(self, ids: list[int]) -> str:
        ignore = {self.pad_id, self.sos_id, self.eos_id}
        return " ".join(self._id_to_token.get(i, UNK_TOKEN) for i in ids if i not in ignore)

    def id_to_piece(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self._token_to_id)


class SentencePieceVocabulary(Vocabulary):
    """Vocabulaire sous-mots (BPE/Unigram) via SentencePiece, un modèle par langue.

    Entraîné sur le TRAIN seul. Modèle écrit dans
    `{processed_dir}/tokenizers/{config_name}_{lang}.model` et rechargé depuis
    le cache si `force_rebuild=False` et que le fichier existe déjà.
    """

    def __init__(self, config_name: str, lang: str, model_path: Path) -> None:
        self.config_name = config_name
        self.lang = lang
        self.model_path = model_path
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @classmethod
    def fit(
        cls,
        texts: pd.Series,
        lang: str,
        config_name: str,
        model_type: str,
        vocab_size: int,
        processed_dir: str,
        force_rebuild: bool = False,
        character_coverage: float = 1.0,
    ) -> SentencePieceVocabulary:
        tokenizers_dir = Path(processed_dir) / "tokenizers"
        tokenizers_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = tokenizers_dir / f"{config_name}_{lang}"
        model_path = model_prefix.with_suffix(".model")

        if force_rebuild or not model_path.exists():
            input_path = tokenizers_dir / f"_train_{config_name}_{lang}.txt"
            input_path.write_text(
                "\n".join(str(t) for t in texts if str(t).strip()), encoding="utf-8"
            )
            try:
                spm.SentencePieceTrainer.train(
                    input=str(input_path),
                    model_prefix=str(model_prefix),
                    vocab_size=vocab_size,
                    model_type=model_type,
                    character_coverage=character_coverage,
                    pad_id=0,
                    unk_id=1,
                    bos_id=2,
                    eos_id=3,
                    # Le corpus d'entraînement (notamment en test) peut être trop
                    # petit pour atteindre exactement vocab_size : on tolère un
                    # vocabulaire plus petit plutôt que de lever une erreur.
                    hard_vocab_limit=False,
                )
            finally:
                input_path.unlink(missing_ok=True)
            print(f"[etape 4] vocab {config_name}/{lang}: modèle SentencePiece entraîné -> {model_path}")
        else:
            print(f"[etape 4] vocab {config_name}/{lang}: modèle SentencePiece en cache -> {model_path}")

        return cls(config_name, lang, model_path)

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(str(text), out_type=int)

    def decode(self, ids: list[int]) -> str:
        ignore = {self.pad_id, self.sos_id, self.eos_id}
        return self._sp.decode([i for i in ids if i not in ignore])

    def id_to_piece(self, token_id: int) -> str:
        return self._sp.id_to_piece(token_id)

    def __len__(self) -> int:
        return self._sp.get_piece_size()


def build_vocabs(
    train_df: pd.DataFrame,
    config: str = "words95",
    processed_dir: str = "data/processed",
    force_rebuild: bool = False,
) -> tuple[Vocabulary, Vocabulary]:
    """Construit les vocabulaires FR (source) et EN (cible) sur `train_df` seul.

    `config` doit être une clé de `TOKENIZATION_CONFIGS`. Renvoie
    `(fr_vocab, en_vocab)` -- ordre imposé par le contrat consommé par
    `notebooks/Rosetta_Optuna.ipynb`.
    """
    if config not in TOKENIZATION_CONFIGS:
        raise ValueError(
            f"config de tokenisation inconnue: {config!r}. "
            f"Configs disponibles: {sorted(TOKENIZATION_CONFIGS)}"
        )
    spec = TOKENIZATION_CONFIGS[config]

    vocabs: list[Vocabulary] = []
    for lang in ("fr", "en"):
        texts = train_df[lang]
        if spec["kind"] == "word":
            vocab: Vocabulary = WordVocabulary.fit(
                texts, lang, config, coverage=float(spec["coverage"])  # type: ignore[arg-type]
            )
        elif spec["kind"] == "sentencepiece":
            vocab = SentencePieceVocabulary.fit(
                texts,
                lang,
                config,
                model_type=str(spec["model_type"]),
                vocab_size=int(spec["vocab_size"]),  # type: ignore[arg-type]
                processed_dir=processed_dir,
                force_rebuild=force_rebuild,
            )
        else:
            raise ValueError(f"kind inconnu dans TOKENIZATION_CONFIGS[{config!r}]: {spec['kind']!r}")
        vocabs.append(vocab)

    fr_vocab, en_vocab = vocabs
    print(
        f"[etape 4] config={config}: vocab FR={len(fr_vocab)}, vocab EN={len(en_vocab)} "
        f"(asymétrie morphologique FR>EN attendue pour les configs mots entiers)"
    )

    if len(train_df):
        exemple = str(train_df["fr"].iloc[0])
        roundtrip = fr_vocab.decode(fr_vocab.encode(exemple))
        print(f"[etape 4] round-trip FR (exemple): {exemple!r} -> {roundtrip!r}")

    return fr_vocab, en_vocab
