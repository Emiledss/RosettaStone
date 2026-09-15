"""Vocabulaires FR/EN construits sur le TRAIN seul (étape 4).

Deux implémentations exposant la **même interface** (`encode`/`decode`/`__len__`,
indices spéciaux `<pad>`=0, `<unk>`=1, `<sos>`=2, `<eos>`=3) :

- `WordVocabulary` : mots entiers, découpage via `tokenize_words` de
  `src.data.text` (règle de l'apostrophe d'élision) -- jamais `str.split()`.
  Configs `full` (couverture 100 %) et `words95` (coupée à 95 % de couverture
  des occurrences). Vocabulaires **séparés** FR/EN (asymétrie morphologique
  attendue entre les deux langues).
- `SentencePieceVocabulary` : sous-mots (BPE/Unigram). 4 configs historiques
  (`bpe4k`, `bpe8k`, `unigram4k`, `unigram8k`, registre statique
  `TOKENIZATION_CONFIGS`), PLUS n'importe quelle taille entière positive
  résolue dynamiquement (`bpe3264`, cf.
  `config_name()`/`_resolve_dynamic_subword_config()` ci-dessous). Vocabulaire
  **conjoint** : un seul modèle SentencePiece entraîné sur FR+EN concaténés du
  train (`vocab_size` = total partagé, pas par langue -- cohérent avec
  MarianMT). Mis en cache sur disque (`data/processed/tokenizers/`, gitignoré)
  -- avec des tailles réellement continues, chaque taille unique laisse son
  propre `.model`/`.vocab` et ça s'accumule (coût assumé et documenté).

`build_vocabs(train_df, config)` est le point d'entrée unique : il renvoie
`(fr_vocab, en_vocab)`, quel que soit le type de vocabulaire choisi. Pour les
configs conjointes, `fr_vocab` et `en_vocab` sont **le même objet** (`fr_vocab
is en_vocab`) -- ça préserve le contrat `(fr_vocab, en_vocab)` consommé par
`build_model`. `config` peut être une clé historique de `TOKENIZATION_CONFIGS`
ou un nom dynamique absent du registre (résolu à la volée).
"""

from __future__ import annotations

import re
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

# Les configurations de tokenisation comparées. `joint` est explicite : True ->
# un seul vocabulaire SentencePiece FR+EN (vocab_size = total partagé) ; False
# -> deux vocabulaires séparés FR/EN -- conjoint pour les configs sous-mots
# seulement, `full`/`words95` restent séparés (asymétrie morphologique FR/EN
# mesurée à ~1,89x en régime mots entiers).
#
# `full` et `words95` sont éliminés par les essais antérieurs (BLEU 0,88 et ~0
# contre 11,36 pour `unigram4k`) et sortent donc de l'espace de recherche
# Optuna -- ils restent implémentés ici pour que ces essais restent
# interprétables.
GENERATED_SUBWORD_METHODS: tuple[str, ...] = ("bpe", "unigram")


def config_name(method: str, vocab_size: int) -> str:
    """Nom canonique d'une config sous-mots pour TOUTE taille entière positive
    -- source de vérité UNIQUE de la convention de nom, réutilisée par
    `TOKENIZATION_CONFIGS` ci-dessous ET par le notebook Optuna (résolution
    `tokenizationMethod` x `vocabSize` -> config).

    Deux formes, selon que `vocab_size` tombe rond ou non :
    - multiple de 1000 (ex. `config_name("bpe", 4000) -> "bpe4k"`) -- forme
      HISTORIQUE, conservée pour que les 6 configs historiques et leurs caches
      disque (`data/processed/tokenizers/bpe4k_joint.model`, ...) restent
      identifiés par le même nom ;
    - sinon, taille littérale (ex. `config_name("bpe", 3264) -> "bpe3264"`) --
      c'est la forme que prennent les tailles réellement continues tirées par
      Optuna.

    Ce choix rend `bpe4000` et `bpe4k` équivalents : `config_name("bpe", 4000)`
    renvoie toujours `"bpe4k"` (jamais `"bpe4000"`), donc un tirage Optuna qui
    retombe exactement sur 4000 réutilise le cache et la spec historiques au
    lieu d'en créer un second pour le même modèle SentencePiece. Voir
    `_resolve_dynamic_subword_config` pour le sens inverse (nom -> méthode +
    taille), utilisé par `build_vocabs` quand quelqu'un appelle directement
    `build_vocabs(df, "bpe4000")`.
    """
    if method not in GENERATED_SUBWORD_METHODS:
        raise ValueError(
            f"method inconnue: {method!r} (attendu parmi {GENERATED_SUBWORD_METHODS})"
        )
    if vocab_size <= 0:
        raise ValueError(f"vocab_size doit être un entier positif, reçu {vocab_size!r}")
    if vocab_size % 1000 == 0:
        return f"{method}{vocab_size // 1000}k"
    return f"{method}{vocab_size}"


# Reconnaît un nom dynamique `<méthode><chiffres>[k]` (ex. `bpe3264`, `bpe4000`,
# `unigram4k`) -- utilisé par `_resolve_dynamic_subword_config` UNIQUEMENT pour
# les noms ABSENTS de `TOKENIZATION_CONFIGS` (les 6 noms historiques matchent
# ce motif aussi, mais sont interceptés avant, par appartenance directe au
# registre statique -- voir `build_vocabs`).
_DYNAMIC_CONFIG_PATTERN = re.compile(rf"^({'|'.join(GENERATED_SUBWORD_METHODS)})(\d+)(k)?$")


def _resolve_dynamic_subword_config(name: str) -> tuple[str, dict[str, object]]:
    """Résout un nom de config sous-mots ABSENT de `TOKENIZATION_CONFIGS` en
    `(nom_canonique, spec)`, sans exiger d'entrée préenregistrée :

    - `"bpe3264"` -> méthode `bpe`, taille LITTÉRALE 3264 (pas de suffixe `k`)
      -- taille continue tirée par Optuna, jamais vue avant. `spec` est
      construite à la volée (`joint=True`, comme toutes les configs sous-mots
      générées).
    - `"bpe4000"` -> méthode `bpe`, taille littérale 4000 -- mais
      `config_name("bpe", 4000)` vaut `"bpe4k"`, DÉJÀ dans le registre : le nom
      canonique renvoyé est `"bpe4k"` et sa spec HISTORIQUE est réutilisée telle
      quelle (même cache disque `bpe4k_joint.model`, pas de second modèle pour
      la même taille).
    - `"unigram4k"` (jamais atteint en pratique : intercepté par le registre
      statique avant d'arriver ici) -> même résolution, par cohérence.

    Lève `ValueError` si `name` ne correspond à aucun schéma `<méthode><entier>[k]`
    connu (méthode absente de `GENERATED_SUBWORD_METHODS`, ou taille <= 0).
    """
    match = _DYNAMIC_CONFIG_PATTERN.match(name)
    if not match:
        raise ValueError(
            f"config de tokenisation inconnue: {name!r}. "
            f"Configs disponibles: {sorted(TOKENIZATION_CONFIGS)} (ou un nom dynamique "
            f"'<{'|'.join(GENERATED_SUBWORD_METHODS)}><taille>', ex. 'bpe3264')"
        )
    method, digits, k_suffix = match.groups()
    vocab_size = int(digits) * 1000 if k_suffix else int(digits)
    canonical_name = config_name(method, vocab_size)  # lève ValueError si vocab_size <= 0
    spec = TOKENIZATION_CONFIGS.get(
        canonical_name,
        {"kind": "sentencepiece", "model_type": method, "vocab_size": vocab_size, "joint": True},
    )
    return canonical_name, spec


TOKENIZATION_CONFIGS: dict[str, dict[str, object]] = {
    "full": {"kind": "word", "coverage": 1.0, "joint": False},
    "words95": {"kind": "word", "coverage": 0.95, "joint": False},
    "bpe4k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 4000, "joint": True},
    "bpe8k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 8000, "joint": True},
    "unigram4k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 4000, "joint": True},
    "unigram8k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 8000, "joint": True},
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
    """Vocabulaire sous-mots (BPE/Unigram) via SentencePiece.

    Entraîné sur le TRAIN seul. Deux modes de construction :

    - `fit` : un modèle par langue, écrit dans
      `{processed_dir}/tokenizers/{config_name}_{lang}.model` (non utilisé par
      les 6 configs actuelles, conservé pour une éventuelle config sous-mots
      non conjointe future).
    - `fit_joint` : un seul modèle entraîné sur FR+EN concaténés, écrit dans
      `{processed_dir}/tokenizers/{config_name}_joint.model` -- c'est le mode
      utilisé par `bpe4k`/`bpe8k`/`unigram4k`/`unigram8k` (`joint=True` dans
      `TOKENIZATION_CONFIGS`).

    Dans les deux cas, rechargé depuis le cache si `force_rebuild=False` et
    que le fichier existe déjà.
    """

    def __init__(self, config_name: str, lang: str, model_path: Path) -> None:
        self.config_name = config_name
        self.lang = lang
        self.model_path = model_path
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @staticmethod
    def _train(
        lines: list[str],
        model_prefix: Path,
        model_type: str,
        vocab_size: int,
        character_coverage: float,
        input_stem: str,
        tokenizers_dir: Path,
    ) -> None:
        """Entraîne un modèle SentencePiece sur `lines` déjà assemblées
        (une phrase par ligne), factorisé entre `fit` (par langue) et
        `fit_joint` (FR+EN concaténés)."""
        input_path = tokenizers_dir / f"_train_{input_stem}.txt"
        input_path.write_text("\n".join(lines), encoding="utf-8")
        try:
            # Déterminisme : `SentencePieceTrainer.train` n'a PAS de paramètre
            # `seed`/`random_seed` dans `TrainerSpec` (un tel kwarg lève
            # `RuntimeError: NOT_FOUND: unknown field name ... in TrainerSpec` --
            # vérifié). La graine se fixe via la fonction MODULE séparée
            # `sentencepiece.set_random_generator_seed()`, à appeler AVANT chaque
            # entraînement -- sans elle, un tokeniseur ré-entraîné (ex. après perte
            # du cache local) ne correspond plus bit à bit au `.model` d'origine :
            # deux entraînements sans seed fixe sur le même corpus produisent des
            # `.model` différents.
            spm.set_random_generator_seed(42)
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
                # Sans ceci, SentencePiece logue chaque fusion sur stderr (des milliers de
                # lignes par tokeniseur), ce qui noie le suivi d'un essai Optuna.
                minloglevel=2,
            )
        finally:
            input_path.unlink(missing_ok=True)

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
        """Un modèle SentencePiece par langue (non utilisé par les 6 configs
        actuelles, voir la docstring de la classe)."""
        tokenizers_dir = Path(processed_dir) / "tokenizers"
        tokenizers_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = tokenizers_dir / f"{config_name}_{lang}"
        model_path = model_prefix.with_suffix(".model")

        if force_rebuild or not model_path.exists():
            lines = [str(t) for t in texts if str(t).strip()]
            cls._train(
                lines, model_prefix, model_type, vocab_size, character_coverage,
                input_stem=f"{config_name}_{lang}", tokenizers_dir=tokenizers_dir,
            )
            print(f"[etape 4] vocab {config_name}/{lang}: modèle SentencePiece entraîné -> {model_path}")
        else:
            print(f"[etape 4] vocab {config_name}/{lang}: modèle SentencePiece en cache -> {model_path}")

        return cls(config_name, lang, model_path)

    @classmethod
    def fit_joint(
        cls,
        texts_fr: pd.Series,
        texts_en: pd.Series,
        config_name: str,
        model_type: str,
        vocab_size: int,
        processed_dir: str,
        force_rebuild: bool = False,
        character_coverage: float = 1.0,
    ) -> SentencePieceVocabulary:
        """Un seul modèle SentencePiece entraîné sur FR+EN concaténés du train.

        `vocab_size` est un **total partagé** (ex. BPE 4K = 4 000 tokens au
        total, pas par langue) -- cohérent avec le vocabulaire conjoint de
        MarianMT. `build_vocabs` référence l'instance renvoyée deux fois
        (`fr_vocab is en_vocab`).
        """
        tokenizers_dir = Path(processed_dir) / "tokenizers"
        tokenizers_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = tokenizers_dir / f"{config_name}_joint"
        model_path = model_prefix.with_suffix(".model")

        if force_rebuild or not model_path.exists():
            lines = [str(t) for t in texts_fr if str(t).strip()] + [
                str(t) for t in texts_en if str(t).strip()
            ]
            cls._train(
                lines, model_prefix, model_type, vocab_size, character_coverage,
                input_stem=f"{config_name}_joint", tokenizers_dir=tokenizers_dir,
            )
            print(
                f"[etape 4] vocab {config_name}/joint (FR+EN): modèle SentencePiece entraîné -> {model_path}"
            )
        else:
            print(
                f"[etape 4] vocab {config_name}/joint (FR+EN): modèle SentencePiece en cache -> {model_path}"
            )

        return cls(config_name, "joint", model_path)

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

    `config` est soit une clé de `TOKENIZATION_CONFIGS` (les 6 configs
    historiques : `full`, `words95`, `bpe4k`, `bpe8k`, `unigram4k`, `unigram8k`),
    soit un nom dynamique ABSENT du registre, résolu à la volée en (méthode,
    taille exacte) par `_resolve_dynamic_subword_config` (ex. `bpe3264` --
    taille réellement continue tirée par Optuna, jamais préenregistrée).
    `TOKENIZATION_CONFIGS` reste statique : seules les entrées historiques y
    sont écrites, la résolution dynamique ne les modifie jamais.

    Renvoie `(fr_vocab, en_vocab)` -- ordre imposé par le contrat consommé par
    `notebooks/Rosetta_Modelisation.ipynb`.
    """
    if config in TOKENIZATION_CONFIGS:
        spec = TOKENIZATION_CONFIGS[config]
    else:
        config, spec = _resolve_dynamic_subword_config(config)
    joint = bool(spec.get("joint", False))

    if spec["kind"] == "sentencepiece" and joint:
        # Vocabulaire conjoint : un seul modèle SentencePiece pour FR+EN,
        # référencé deux fois -- `fr_vocab is en_vocab`.
        joint_vocab = SentencePieceVocabulary.fit_joint(
            train_df["fr"],
            train_df["en"],
            config,
            model_type=str(spec["model_type"]),
            vocab_size=int(spec["vocab_size"]),  # type: ignore[arg-type]
            processed_dir=processed_dir,
            force_rebuild=force_rebuild,
        )
        fr_vocab: Vocabulary = joint_vocab
        en_vocab: Vocabulary = joint_vocab
    else:
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

    note = (
        "vocabulaire CONJOINT FR+EN, même objet (fr_vocab is en_vocab)"
        if joint
        else "asymétrie morphologique FR>EN attendue (vocabulaires séparés)"
    )
    print(f"[etape 4] config={config}: vocab FR={len(fr_vocab)}, vocab EN={len(en_vocab)} ({note})")
    if spec["kind"] == "sentencepiece":
        # Taille DEMANDÉE (spec["vocab_size"]) vs OBTENUE (len(vocab)) : SentencePiece
        # tourne avec hard_vocab_limit=False et peut ne pas atteindre exactement la
        # taille demandée -- les deux doivent être visibles séparément, sous peine
        # d'analyser des résultats sur un chiffre faux.
        demande = int(spec["vocab_size"])  # type: ignore[arg-type]
        obtenue = len(fr_vocab)
        ecart = "" if obtenue == demande else f" (ÉCART: -{demande - obtenue})"
        print(f"[etape 4] config={config}: vocab_size demandé={demande}, obtenu={obtenue}{ecart}")

    if len(train_df):
        exemple = str(train_df["fr"].iloc[0])
        roundtrip = fr_vocab.decode(fr_vocab.encode(exemple))
        print(f"[etape 4] round-trip FR (exemple): {exemple!r} -> {roundtrip!r}")

    return fr_vocab, en_vocab
