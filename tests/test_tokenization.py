"""Tests des étapes 4, 5, 5bis (`src/tokenization`) : vocabulaires, numérisation,
statistiques de tokenisation.

Corpus synthétique volontairement petit mais avec un lexique (160 pseudo-mots
par langue) plus large que le budget SentencePiece de test (`vocab_size=64`,
cf. plan) : par construction, SentencePiece ne peut PAS donner un token dédié
à chaque mot entier et doit réutiliser des pièces plus courtes sur une partie
du corpus -- ce qui garantit un ratio tokens/mots > 1 (sinon, avec un budget
généreux par rapport au nombre de mots distincts, le test de fragmentation
serait vide de sens : SentencePiece pourrait apprendre un token par mot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tokenization import stats as tokstats
from src.tokenization import vocab_builder as vb
from src.tokenization.numerize import (
    TranslationDataset,
    derive_max_len,
    make_dataloaders,
)
from src.tokenization.vocab_builder import (
    TOKENIZATION_CONFIGS,
    Vocabulary,
    build_vocabs,
)

SYLLABLES = [
    "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu",
    "da", "de", "di", "do", "du", "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu", "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu",
]


def _generate_lexicon(n: int, seed: int, min_syll: int = 2, max_syll: int = 4) -> list[str]:
    """Lexique de `n` pseudo-mots distincts (2 à 4 syllabes), assez nombreux et
    assez longs pour dépasser le budget d'un vocabulaire SentencePiece de test
    et forcer une vraie fragmentation en sous-mots."""
    rng = np.random.default_rng(seed)
    words: set[str] = set()
    while len(words) < n:
        k = int(rng.integers(min_syll, max_syll + 1))
        words.add("".join(rng.choice(SYLLABLES, size=k)))
    return sorted(words)


FR_LEXICON = _generate_lexicon(160, seed=101)
EN_LEXICON = _generate_lexicon(160, seed=202)

TRAIN_ONLY_FR_WORD = "zzzmotrareseulementtrain"
TRAIN_ONLY_EN_WORD = "zzzraretrainonlyword"
VAL_ONLY_FR_WORD = "zzzmotjamaisvutrain"
VAL_ONLY_EN_WORD = "zzzunseenvalword"

# Budget SentencePiece de test : nettement < 160 mots distincts, pour forcer
# la fragmentation (cf. docstring du module).
TEST_SP_VOCAB_SIZE = 64


def _zipf_weights(n: int) -> np.ndarray:
    """Poids en 1/rang (loi de Zipf approximative) : donne une vraie traîne
    longue, indispensable pour que `words95` coupe une part significative du
    vocabulaire (cf. test de couverture)."""
    weights = 1.0 / np.arange(1, n + 1)
    return weights / weights.sum()


def _build_corpus(n: int, lexicon_fr: list[str], lexicon_en: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights_fr = _zipf_weights(len(lexicon_fr))
    weights_en = _zipf_weights(len(lexicon_en))
    rows = []
    for _ in range(n):
        length = int(rng.integers(4, 10))
        fr = " ".join(rng.choice(lexicon_fr, size=length, p=weights_fr))
        en = " ".join(rng.choice(lexicon_en, size=length, p=weights_en))
        rows.append({"fr": fr, "en": en})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def toy_splits() -> dict[str, pd.DataFrame]:
    """Petit corpus synthétique train/val/test, avec un mot exclusif au train
    et un mot exclusif au val (test de non-fuite du vocabulaire, étape 4)."""
    train = _build_corpus(300, FR_LEXICON, EN_LEXICON, seed=1)
    train.loc[0, "fr"] = f"{train.loc[0, 'fr']} {TRAIN_ONLY_FR_WORD}"
    train.loc[0, "en"] = f"{train.loc[0, 'en']} {TRAIN_ONLY_EN_WORD}"

    val = _build_corpus(60, FR_LEXICON, EN_LEXICON, seed=2)
    val.loc[0, "fr"] = f"{val.loc[0, 'fr']} {VAL_ONLY_FR_WORD}"
    val.loc[0, "en"] = f"{val.loc[0, 'en']} {VAL_ONLY_EN_WORD}"

    test = _build_corpus(60, FR_LEXICON, EN_LEXICON, seed=3)
    return {"train": train, "val": val, "test": test}


@pytest.fixture(scope="module")
def small_configs() -> dict[str, dict[str, object]]:
    """Copie de `TOKENIZATION_CONFIGS` avec un `vocab_size` réduit (64, dans la
    plage 64-128 recommandée) pour un entraînement SentencePiece rapide sur le
    petit corpus synthétique."""
    small: dict[str, dict[str, object]] = {}
    for name, spec in TOKENIZATION_CONFIGS.items():
        spec = dict(spec)
        if spec["kind"] == "sentencepiece":
            spec["vocab_size"] = TEST_SP_VOCAB_SIZE
        small[name] = spec
    return small


@pytest.fixture(scope="module")
def module_monkeypatch():
    """`monkeypatch` à portée module (la fixture standard est à portée fonction)."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def built_vocabs(
    toy_splits: dict[str, pd.DataFrame],
    small_configs: dict[str, dict[str, object]],
    module_monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Vocabulary, Vocabulary]]:
    """Construit les 6 configs une seule fois pour tout le module -- l'entraînement
    SentencePiece est le plus coûteux des tests de ce fichier. Modèles écrits
    dans un `tmp_path` pytest, jamais dans `data/`."""
    module_monkeypatch.setattr(vb, "TOKENIZATION_CONFIGS", small_configs)
    processed_dir = tmp_path_factory.mktemp("tokenization_models")
    return {
        config_name: build_vocabs(
            toy_splits["train"],
            config=config_name,
            processed_dir=str(processed_dir),
            force_rebuild=True,
        )
        for config_name in small_configs
    }


WORD_CONFIGS = ("full", "words95")
SUBWORD_CONFIGS = ("bpe4k", "bpe8k", "unigram4k", "unigram8k")


# ---------------------------------------------------------------------------
# 1. Indices spéciaux sur les 6 configs
# ---------------------------------------------------------------------------
def test_les_six_configs_ont_les_bons_indices_speciaux(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    assert set(built_vocabs) == set(TOKENIZATION_CONFIGS)
    for fr_vocab, en_vocab in built_vocabs.values():
        for vocab in (fr_vocab, en_vocab):
            assert vocab.pad_id == 0
            assert vocab.unk_id == 1
            assert vocab.sos_id == 2
            assert vocab.eos_id == 3


# ---------------------------------------------------------------------------
# 2. Round-trip encode/decode non destructif
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", list(TOKENIZATION_CONFIGS))
def test_round_trip_encode_decode(
    config_name: str, built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]]
) -> None:
    fr_vocab, _en_vocab = built_vocabs[config_name]
    phrase = " ".join(FR_LEXICON[:5])
    decoded = fr_vocab.decode(fr_vocab.encode(phrase))
    assert decoded.strip() == phrase.strip()


# ---------------------------------------------------------------------------
# 3. Vocabulaire construit sur le TRAIN seul : mot de val -> <unk>
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", WORD_CONFIGS)
def test_mot_present_seulement_dans_val_est_encode_en_unk(
    config_name: str, built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]]
) -> None:
    fr_vocab, en_vocab = built_vocabs[config_name]
    assert fr_vocab.encode(VAL_ONLY_FR_WORD) == [fr_vocab.unk_id]
    assert en_vocab.encode(VAL_ONLY_EN_WORD) == [en_vocab.unk_id]


def test_full_couvre_un_mot_present_uniquement_dans_le_train(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """`full` couvre 100 % du train : un mot du train (même rare) n'est jamais OOV."""
    fr_vocab, en_vocab = built_vocabs["full"]
    assert fr_vocab.encode(TRAIN_ONLY_FR_WORD) != [fr_vocab.unk_id]
    assert en_vocab.encode(TRAIN_ONLY_EN_WORD) != [en_vocab.unk_id]


# ---------------------------------------------------------------------------
# 4. words95 : couverture ~95 % et vocabulaire plus petit que full
# ---------------------------------------------------------------------------
def test_words95_couverture_et_taille(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    full_fr, full_en = built_vocabs["full"]
    words95_fr, words95_en = built_vocabs["words95"]

    tolerance = 0.06
    assert abs(words95_fr.coverage_atteinte - 0.95) < tolerance
    assert abs(words95_en.coverage_atteinte - 0.95) < tolerance

    assert len(words95_fr) < len(full_fr)
    assert len(words95_en) < len(full_en)


# ---------------------------------------------------------------------------
# 5. Ratio tokens/mots : 1.0 pour mots entiers, > 1 pour les sous-mots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", WORD_CONFIGS)
def test_ratio_tokens_mots_egal_1_pour_mots_entiers(
    config_name: str,
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, en_vocab = built_vocabs[config_name]
    train = toy_splits["train"]
    assert tokstats.fragmentation_ratio(fr_vocab, train["fr"], "fr") == pytest.approx(1.0)
    assert tokstats.fragmentation_ratio(en_vocab, train["en"], "en") == pytest.approx(1.0)


@pytest.mark.parametrize("config_name", SUBWORD_CONFIGS)
def test_ratio_tokens_mots_superieur_a_1_pour_sous_mots(
    config_name: str,
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, en_vocab = built_vocabs[config_name]
    train = toy_splits["train"]
    assert tokstats.fragmentation_ratio(fr_vocab, train["fr"], "fr") > 1.0
    assert tokstats.fragmentation_ratio(en_vocab, train["en"], "en") > 1.0


# ---------------------------------------------------------------------------
# 6. max_len=None dérive une valeur cohérente et n'est jamais dépassé
# ---------------------------------------------------------------------------
def test_max_len_auto_derive_et_jamais_depasse(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, en_vocab = built_vocabs["words95"]

    attendu = derive_max_len(toy_splits["train"], fr_vocab, en_vocab)
    assert attendu > 2  # au moins de la place pour <sos>/<eos> + 1 token

    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)
    for part, loader in loaders.items():
        dataset = loader.dataset
        assert dataset.max_len == attendu
        assert len(dataset) == len(toy_splits[part])
        for src, tgt in loader:
            assert src.shape[1] == attendu
            assert tgt.shape[1] == attendu


# ---------------------------------------------------------------------------
# 7. Le padding utilise bien l'indice 0
# ---------------------------------------------------------------------------
def test_padding_utilise_indice_zero(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, en_vocab = built_vocabs["words95"]
    df = pd.DataFrame({"fr": [FR_LEXICON[0]], "en": [EN_LEXICON[0]]})  # 1 seul mot

    max_len = 10
    dataset = TranslationDataset(df, fr_vocab, en_vocab, max_len)
    src, tgt = dataset[0]

    # <sos> + 1 mot + <eos> = 3 tokens utiles, le reste doit être du padding
    assert src[0].item() == fr_vocab.sos_id
    assert src[3:].tolist() == [fr_vocab.pad_id] * (max_len - 3)
    assert tgt[3:].tolist() == [en_vocab.pad_id] * (max_len - 3)
    assert fr_vocab.pad_id == 0
    assert en_vocab.pad_id == 0


# ---------------------------------------------------------------------------
# Bonus : les fonctions de stats.py tournent sans erreur et renvoient des
# formes cohérentes (utilisées par le notebook de tokenisation).
# ---------------------------------------------------------------------------
def test_stats_module_zipf_et_longueurs_de_tokens(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, _en_vocab = built_vocabs["bpe4k"]
    train = toy_splits["train"]

    ranks, freqs = tokstats.zipf_frequencies(fr_vocab, train["fr"])
    assert len(ranks) == len(freqs)
    assert (np.diff(freqs) <= 0).all()  # décroissant

    lengths = tokstats.token_char_lengths(fr_vocab, train["fr"])
    assert lengths.size > 0
    # une pièce peut être réduite à "▁" seul (marqueur de début de mot, longueur
    # 0 une fois le marqueur retiré) -- ce n'est pas une erreur, juste un cas
    # limite ; on vérifie en revanche qu'il existe bien des pièces multi-caractères.
    assert (lengths >= 0).all()
    assert lengths.max() > 1

    row = tokstats.tokenization_stats_row(fr_vocab, train["fr"], "fr", "bpe4k", "train")
    assert row["ratio_tokens_mots"] > 1.0
    assert 0.0 <= row["taux_oov"] <= 1.0
