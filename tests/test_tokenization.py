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

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.models.seq2seq import build_model
from src.tokenization import stats as tokstats
from src.tokenization import vocab_builder as vb
from src.tokenization.numerize import (
    LengthGroupedBatchSampler,
    TranslationDataset,
    collate_fn,
    derive_max_len,
    make_dataloaders,
)
from src.tokenization.vocab_builder import (
    TOKENIZATION_CONFIGS,
    SentencePieceVocabulary,
    Vocabulary,
    build_vocabs,
    config_name,
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


@pytest.mark.parametrize("config_name", SUBWORD_CONFIGS)
def test_round_trip_conjoint_fr_et_en_avec_le_meme_vocabulaire(
    config_name: str, built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]]
) -> None:
    """Sur une config conjointe, `fr_vocab` encode/décode aussi bien une
    phrase FR qu'une phrase EN -- c'est le même modèle SentencePiece."""
    fr_vocab, en_vocab = built_vocabs[config_name]
    phrase_fr = " ".join(FR_LEXICON[:5])
    phrase_en = " ".join(EN_LEXICON[:5])
    assert fr_vocab.decode(fr_vocab.encode(phrase_fr)).strip() == phrase_fr.strip()
    assert en_vocab.decode(en_vocab.encode(phrase_en)).strip() == phrase_en.strip()


# ---------------------------------------------------------------------------
# 2bis. Vocabulaire conjoint : même objet FR/EN, taille = vocab_size visé ;
# `full`/`words95` restent deux vocabulaires distincts et de tailles différentes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", SUBWORD_CONFIGS)
def test_config_conjointe_meme_objet_et_taille_visee(
    config_name: str,
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
    small_configs: dict[str, dict[str, object]],
) -> None:
    fr_vocab, en_vocab = built_vocabs[config_name]
    assert fr_vocab is en_vocab
    assert len(fr_vocab) == small_configs[config_name]["vocab_size"]
    assert len(en_vocab) == small_configs[config_name]["vocab_size"]


def test_full_et_words95_restent_deux_vocabulaires_distincts(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    for nom_config in WORD_CONFIGS:
        fr_vocab, en_vocab = built_vocabs[nom_config]
        assert fr_vocab is not en_vocab
        assert len(fr_vocab) != len(en_vocab)


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
# 6. max_len=None dérive une valeur cohérente (garde-fou) et n'est jamais
# dépassée -- mais n'est plus la forme de CHAQUE batch (padding dynamique,
# lot H) : chaque batch est paddé à SON PROPRE maximum, `<= attendu`.
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
            assert src.shape[1] <= attendu  # garde-fou : jamais dépassé
            assert tgt.shape[1] <= attendu
            assert src.shape[0] == tgt.shape[0]  # même taille de batch


# ---------------------------------------------------------------------------
# 7. Le padding (par batch, lot H) utilise bien l'indice 0
# ---------------------------------------------------------------------------
def test_padding_utilise_indice_zero(
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """`TranslationDataset` ne padde plus (séquences de longueur variable,
    lot H) -- le padding s'observe au niveau du batch, via `collate_fn`."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    # deux exemples de longueurs différentes pour forcer du padding dans le batch
    df = pd.DataFrame(
        {
            "fr": [FR_LEXICON[0], " ".join(FR_LEXICON[:3])],
            "en": [EN_LEXICON[0], " ".join(EN_LEXICON[:3])],
        }
    )

    max_len = 10
    dataset = TranslationDataset(df, fr_vocab, en_vocab, max_len)

    # un seul mot -> <sos> + 1 token + <eos> = 3 tokens, PAS paddé au niveau du dataset
    src0, _tgt0 = dataset[0]
    assert src0[0].item() == fr_vocab.sos_id
    assert len(src0) == 3
    assert fr_vocab.pad_id == 0
    assert en_vocab.pad_id == 0

    src_batch, tgt_batch = collate_fn([dataset[0], dataset[1]])
    # la ligne courte (exemple 0) est paddée après ses tokens réels, jamais avant
    longueur_reelle_src0 = len(dataset[0][0])
    longueur_reelle_tgt0 = len(dataset[0][1])
    assert src_batch[0, :longueur_reelle_src0].tolist() == dataset[0][0].tolist()
    assert src_batch[0, longueur_reelle_src0:].tolist() == [0] * (
        src_batch.shape[1] - longueur_reelle_src0
    )
    assert tgt_batch[0, :longueur_reelle_tgt0].tolist() == dataset[0][1].tolist()
    assert tgt_batch[0, longueur_reelle_tgt0:].tolist() == [0] * (
        tgt_batch.shape[1] - longueur_reelle_tgt0
    )
    # l'exemple le plus long du batch (indice 1) n'a lui aucun padding
    assert 0 not in src_batch[1].tolist()[: len(dataset[1][0])]


# ---------------------------------------------------------------------------
# 8. Padding dynamique par batch et regroupement par longueur (lot H)
# ---------------------------------------------------------------------------
def test_collate_fn_padde_source_et_cible_independamment_et_preserve_le_contenu() -> None:
    """Unitaire, sans vocabulaire : preuve directe que `collate_fn` padde
    source et cible à des longueurs INDÉPENDANTES, que le padding (`0`) est
    placé aux bonnes positions (après le contenu réel), et que ce contenu réel
    n'est jamais altéré ni tronqué."""
    batch = [
        (
            torch.tensor([2, 5, 6, 3], dtype=torch.long),
            torch.tensor([2, 7, 8, 9, 3], dtype=torch.long),
        ),
        (
            torch.tensor([2, 4, 3], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
        ),
    ]
    src, tgt = collate_fn(batch)

    assert src.shape == (2, 4)  # max longueur source du batch
    assert tgt.shape == (2, 5)  # max longueur cible du batch, INDÉPENDANTE de la source
    assert src[0].tolist() == [2, 5, 6, 3]
    assert src[1].tolist() == [2, 4, 3, 0]
    assert tgt[0].tolist() == [2, 7, 8, 9, 3]
    assert tgt[1].tolist() == [2, 3, 0, 0, 0]


def test_deux_batchs_successifs_du_train_ont_des_longueurs_differentes(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """Preuve que le padding est dynamique (pas un `max_len` global figé) :
    les batchs du train n'ont pas tous la même forme temporelle."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)

    formes_src = {src.shape[1] for src, _tgt in loaders["train"]}
    assert len(formes_src) > 1


def test_regroupement_par_longueur_produit_des_batchs_plus_homogenes(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """`LengthGroupedBatchSampler` : l'écart intra-batch (max - min des
    longueurs) est nettement plus petit que pour des batchs formés dans
    l'ordre d'origine (non triés) du même dataset."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    max_len = derive_max_len(toy_splits["train"], fr_vocab, en_vocab)
    dataset = TranslationDataset(toy_splits["train"], fr_vocab, en_vocab, max_len)
    lengths = dataset.lengths()
    batch_size = 8

    sampler = LengthGroupedBatchSampler(lengths, batch_size=batch_size, shuffle=False)
    batches_groupes = list(sampler)
    assert len(batches_groupes) == math.ceil(len(lengths) / batch_size)

    ecarts_groupes = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches_groupes]
    ecarts_naifs = [
        max(lengths[i] for i in range(start, min(start + batch_size, len(lengths))))
        - min(lengths[i] for i in range(start, min(start + batch_size, len(lengths))))
        for start in range(0, len(lengths), batch_size)
    ]

    assert sum(ecarts_groupes) / len(ecarts_groupes) < sum(ecarts_naifs) / len(ecarts_naifs)


def test_ordre_des_batchs_du_train_change_d_une_epoch_a_l_autre(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """Le CONTENU des batchs groupés par longueur est stable, mais leur ORDRE
    est remélangé à chaque epoch (chaque nouvel appel `iter(loader)`)."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)
    train_loader = loaders["train"]

    def signature() -> tuple[int, ...]:
        return tuple(src.shape[1] for src, _tgt in train_loader)

    torch.manual_seed(0)
    signature_epoch_1 = signature()
    torch.manual_seed(1)
    signature_epoch_2 = signature()

    assert signature_epoch_1 != signature_epoch_2


def test_bucket_by_length_false_fonctionne(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """`bucket_by_length=False` retombe sur des batchs aléatoires classiques,
    padding dynamique conservé -- toutes les paires du train sont bien vues,
    une seule fois par epoch."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(
        toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8, bucket_by_length=False
    )

    total = 0
    formes_src = set()
    for src, tgt in loaders["train"]:
        assert src.shape[0] == tgt.shape[0]
        formes_src.add(src.shape[1])
        total += src.shape[0]
    assert total == len(toy_splits["train"])
    assert len(formes_src) > 1  # toujours du padding dynamique, pas un max_len global figé


def test_ordre_val_et_test_reste_sequentiel_pour_la_correspondance_hypotheses_references(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """Piège du lot H : le regroupement par longueur (et son mélange) ne
    s'applique QU'AU train -- val/test restent en ordre séquentiel, condition
    nécessaire pour que `src/evaluation/generate.py` associe chaque hypothèse
    à la bonne référence par indice."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)

    for part in ("val", "test"):
        dataset = loaders[part].dataset
        attendu_src = [tenseur.tolist() for tenseur in dataset.src_tensors]

        obtenu_src: list[list[int]] = []
        for src, _tgt in loaders[part]:
            for ligne in src:
                # le padding (0) n'apparaît jamais dans le contenu réel (pad_id
                # réservé) -- le retirer en fin de ligne reconstruit la séquence
                # non paddée d'origine, sans ambiguïté.
                sans_padding = [tok for tok in ligne.tolist() if tok != 0]
                obtenu_src.append(sans_padding)

        assert obtenu_src == attendu_src


# ---------------------------------------------------------------------------
# 9. Intégration : `build_model` consomme les loaders sans erreur de
# dimension, avec et sans attention -- non-régression : la loss ignore
# toujours le `<pad>`.
# ---------------------------------------------------------------------------
def test_integration_build_model_consomme_les_loaders_sans_erreur_de_dimension(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)

    for use_attention in (False, True):
        model = build_model(
            len(fr_vocab),
            len(en_vocab),
            emb_dim=8,
            hidden_dim=16,
            cell_type="gru",
            dropout=0.0,
            use_attention=use_attention,
        )
        for src, tgt in loaders["train"]:
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            assert output.shape[0] == tgt.shape[0]
            assert output.shape[1] == tgt.shape[1]  # cohérence automatique (boucle sur tgt_len)
            assert output.shape[2] == len(en_vocab)


def test_pipeline_bout_en_bout_loss_finie_avec_padding_dynamique(
    toy_splits: dict[str, pd.DataFrame],
    built_vocabs: dict[str, tuple[Vocabulary, Vocabulary]],
) -> None:
    """Non-régression : la loss (`ignore_index=pad_id`) reste finie sur un
    batch réel à padding dynamique -- le `<pad>` n'entre jamais dans le calcul,
    quelle que soit la forme (variable) du batch."""
    fr_vocab, en_vocab = built_vocabs["words95"]
    loaders = make_dataloaders(toy_splits, fr_vocab, en_vocab, max_len=None, batch_size=8)
    model = build_model(
        len(fr_vocab), len(en_vocab), emb_dim=8, hidden_dim=16, cell_type="gru", dropout=0.0
    )
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.pad_id)

    for src, tgt in loaders["train"]:
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        loss = criterion(output[:, 1:, :].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        assert math.isfinite(loss.item())


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


# ---------------------------------------------------------------------------
# 10. Registre statique (les 6 configs historiques) et helper `config_name()`.
# Tests de métadonnées uniquement (dict/fonction pure) : aucun entraînement
# SentencePiece requis, sauf le dernier test de cette section (corpus
# synthétique minuscule dans un `tmp_path`).
# ---------------------------------------------------------------------------
def test_les_4_configs_sous_mots_ont_les_bons_attributs() -> None:
    attendu = {
        "bpe4k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 4000, "joint": True},
        "bpe8k": {"kind": "sentencepiece", "model_type": "bpe", "vocab_size": 8000, "joint": True},
        "unigram4k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 4000, "joint": True},
        "unigram8k": {"kind": "sentencepiece", "model_type": "unigram", "vocab_size": 8000, "joint": True},
    }
    for name, spec in attendu.items():
        assert TOKENIZATION_CONFIGS[name] == spec


def test_config_name_coherent_avec_les_4_configs_sous_mots() -> None:
    for method, vocab_size in (("bpe", 4000), ("bpe", 8000), ("unigram", 4000), ("unigram", 8000)):
        assert config_name(method, vocab_size) in TOKENIZATION_CONFIGS
        assert config_name(method, vocab_size) == f"{method}{vocab_size // 1000}k"


def test_config_name_rejette_une_methode_ou_une_taille_invalide() -> None:
    with pytest.raises(ValueError):
        config_name("wordpiece", 4000)
    with pytest.raises(ValueError):
        config_name("bpe", 0)
    with pytest.raises(ValueError):
        config_name("bpe", -100)


# ---------------------------------------------------------------------------
# 11. Taille de vocabulaire réellement continue (config_name accepte toute
# taille entière positive, pas seulement les multiples de 1000).
# ---------------------------------------------------------------------------
def test_config_name_accepte_une_taille_arbitraire_non_ronde() -> None:
    """3264 n'est pas un multiple de 1000 : `config_name` doit quand même
    produire un nom stable et utilisable comme clé/nom de fichier."""
    nom = config_name("bpe", 3264)
    assert nom == "bpe3264"
    # stable : deux appels avec la même taille produisent le même nom
    assert config_name("bpe", 3264) == nom


def test_config_name_bpe4000_et_bpe4k_designent_le_meme_modele() -> None:
    """`config_name("bpe", 4000)` (taille littérale, arrivée par ex. via une
    résolution dynamique) doit renvoyer LE MÊME nom canonique que la config
    historique `bpe4k`, pas un second nom qui créerait un second cache disque
    pour le même modèle SentencePiece."""
    assert config_name("bpe", 4000) == "bpe4k"
    assert config_name("bpe", 4000) in TOKENIZATION_CONFIGS


def test_les_6_noms_historiques_restent_resolvables() -> None:
    for nom in ("full", "words95", "bpe4k", "bpe8k", "unigram4k", "unigram8k"):
        assert nom in TOKENIZATION_CONFIGS


def test_build_vocabs_avec_un_nom_dynamique_absent_du_registre(
    toy_splits: dict[str, pd.DataFrame],
    tmp_path,
) -> None:
    """`bpe2500` n'est présent nulle part dans `TOKENIZATION_CONFIGS` : `build_vocabs`
    doit le résoudre dynamiquement (méthode `bpe`, taille 2500) sur un corpus
    synthétique minuscule dans `tmp_path`, sans jamais toucher `data/`."""
    fr_vocab, en_vocab = build_vocabs(
        toy_splits["train"],
        config="bpe2500",
        processed_dir=str(tmp_path),
        force_rebuild=True,
    )
    # vocabulaire conjoint (même objet FR/EN, comme toutes les configs sous-mots)
    assert fr_vocab is en_vocab
    for vocab in (fr_vocab, en_vocab):
        assert vocab.pad_id == 0
        assert vocab.unk_id == 1
        assert vocab.sos_id == 2
        assert vocab.eos_id == 3
    # la taille RÉELLEMENT obtenue est enregistrée sur l'objet vocabulaire lui-même
    # (`len(vocab)`) -- ce que le notebook journalise ensuite en user_attr Optuna.
    assert len(fr_vocab) <= 2500  # hard_vocab_limit=False : jamais dépassé, peut être inférieur
    assert (tmp_path / "tokenizers" / "bpe2500_joint.model").exists()


def test_build_vocabs_nom_totalement_inconnu_leve_value_error(
    toy_splits: dict[str, pd.DataFrame],
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        build_vocabs(toy_splits["train"], config="wordpiece1234", processed_dir=str(tmp_path))


def test_full_et_words95_toujours_presents_dans_les_configs() -> None:
    """`full` et `words95` sortent de l'espace de recherche Optuna mais restent
    implémentés (essais antérieurs interprétables)."""
    assert TOKENIZATION_CONFIGS["full"] == {"kind": "word", "coverage": 1.0, "joint": False}
    assert TOKENIZATION_CONFIGS["words95"] == {"kind": "word", "coverage": 0.95, "joint": False}


def test_une_config_sous_mots_produit_un_vocabulaire_conjoint_de_la_taille_demandee(
    tmp_path,
) -> None:
    """`bpe4k` : corpus synthétique minuscule entraîné dans un `tmp_path` pytest
    -- jamais sur le vrai corpus ni dans `data/`, pour garder le gate rapide.
    Réutilise le même lexique/corpus que la fixture `built_vocabs` (déjà
    calibré pour forcer une vraie fragmentation à `TEST_SP_VOCAB_SIZE`)."""
    corpus = _build_corpus(300, FR_LEXICON, EN_LEXICON, seed=1)

    spec = TOKENIZATION_CONFIGS["bpe4k"]
    vocab = SentencePieceVocabulary.fit_joint(
        corpus["fr"],
        corpus["en"],
        "bpe4k",
        model_type=str(spec["model_type"]),
        vocab_size=TEST_SP_VOCAB_SIZE,
        processed_dir=str(tmp_path),
        force_rebuild=True,
    )

    assert len(vocab) == TEST_SP_VOCAB_SIZE
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.sos_id == 2
    assert vocab.eos_id == 3
