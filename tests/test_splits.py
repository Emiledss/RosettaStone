"""Tests des étapes 0->3 (`src/data`) : nettoyage, déduplication, split anti-fuite.

Les tests de logique (rapides, toujours exécutés) tournent sur des fixtures
synthétiques construites à la main -- pas sur le corpus réel, pour que le gate
reste rapide. Le test d'intégration sur le corpus réel est sauté automatiquement
si `data/processed/` est absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.cleaning import clean_corpus, deduplicate_pairs, normalize_corpus
from src.data.splits import MIN_GROUPES_LONGS_TEST, split_by_target_group
from src.data.text import count_words, normalize_text, tokenize_words

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Fixture 1 : petit corpus "à la main" pour les tests de nettoyage/déduplication
# ---------------------------------------------------------------------------
def build_cleaning_fixture() -> pd.DataFrame:
    """Corpus jouet couvrant, une fois chacun, les cas visés par `clean_corpus`
    et `deduplicate_pairs` : chaîne vide (x2, fr et en), en==fr, ratio de longueur
    aberrant, paire trop longue (> max_words), doublon exact (x2 lignes), et un
    alignement 1->N légitime (même cible EN, deux sources FR différentes) qui NE
    doit PAS être supprimé par la déduplication.
    """
    rows = [
        # A: normale, conservée partout
        {"fr": "Bonjour à tous", "en": "Hello everyone"},
        # B: chaîne vide côté fr (whitespace seul) -> retirée à l'étape 1
        {"fr": "   ", "en": "Hello there"},
        # C: chaîne vide côté en -> retirée à l'étape 1
        {"fr": "Salut", "en": ""},
        # D: en == fr -> retirée à l'étape 2
        {"fr": "Bonjours", "en": "Bonjours"},
        # E: ratio de longueur aberrant (fr très court, en très long) -> étape 3
        {
            "fr": "Chat noir",
            "en": "This is an extremely long and totally unrelated sentence with way more words",
        },
        # F1/F2: doublon exact -> conservées par le nettoyage, retirées (1x) par la dédup
        {"fr": "Le chat dort", "en": "The cat sleeps"},
        {"fr": "Le chat dort", "en": "The cat sleeps"},
        # G: normale avec élision fr, conservée partout
        {"fr": "Il fait beau aujourd'hui", "en": "The weather is nice today"},
        # H: trop longue (> max_words=40) -> retirée à l'étape 4
        {"fr": " ".join(["mot"] * 45), "en": " ".join(["word"] * 45)},
        # I1/I2: alignement 1->N légitime (même cible EN, sources FR différentes)
        {"fr": "Le chien court", "en": "The dog runs"},
        {"fr": "Le toutou court", "en": "The dog runs"},
    ]
    return pd.DataFrame(rows)


def test_clean_corpus_retire_exactement_ce_qui_est_attendu_dans_l_ordre() -> None:
    """Chaque sous-étape de `clean_corpus` retire exactement ce qu'elle doit,
    dans l'ordre imposé (chaînes vides -> en==fr -> ratio -> max_words)."""
    df = build_cleaning_fixture()
    cleaned, journal = clean_corpus(df)

    assert list(journal["etape"]) == ["chaines_vides", "en_egal_fr", "ratio_longueur", "max_words"]
    assert list(journal["n_retirees"]) == [2, 1, 1, 1]
    assert list(journal["n_restantes"]) == [9, 8, 7, 6]

    # plus aucune paire vide ni en==fr après coup (point de contrôle du plan)
    assert not (cleaned["fr"].astype(str).str.strip().eq("")).any()
    assert not (cleaned["en"].astype(str).str.strip().eq("")).any()
    assert not (cleaned["fr"] == cleaned["en"]).any()
    assert len(cleaned) == 6


def test_deduplicate_pairs_retire_doublons_exacts_et_preserve_alignements_1_n() -> None:
    """Les doublons exacts (fr, en) sont retirés ; un alignement 1->N (même cible
    EN, sources FR différentes) est préservé (étape 2 du plan)."""
    df = build_cleaning_fixture()
    cleaned, _journal = clean_corpus(df)
    deduped, n_dup = deduplicate_pairs(cleaned)

    assert n_dup == 1
    assert len(deduped) == 5
    assert not deduped.duplicated(subset=["fr", "en"]).any()

    # l'alignement 1->N ("The dog runs") est bien préservé à deux lignes distinctes
    assert (deduped["en"] == "The dog runs").sum() == 2
    assert set(deduped.loc[deduped["en"] == "The dog runs", "fr"]) == {"Le chien court", "Le toutou court"}


# ---------------------------------------------------------------------------
# Fixture 2 : corpus jouet plus large pour les tests du split (groupes/strates)
# ---------------------------------------------------------------------------
def build_split_fixture() -> pd.DataFrame:
    """Corpus jouet déjà "propre", construit pour donner un nombre de groupes
    suffisant pour que les proportions de strates soient stables : 120 groupes
    courts (cible EN <= 18 mots) et 40 groupes longs (cible EN > 18 mots), avec
    un nombre de sources FR variable par groupe (1 à 3) pour couvrir les
    alignements 1->N."""
    rows = []
    for i in range(120):
        longueur = 3 + (i % 13)  # 3..15 mots (strate courte)
        cible = " ".join(f"s{i}mot{j}" for j in range(longueur))
        n_sources = 1 + (i % 3)
        for k in range(n_sources):
            rows.append({"fr": f"source courte {i} variante {k}", "en": cible})
    for i in range(40):
        longueur = 19 + (i % 12)  # 19..30 mots (strate longue)
        cible = " ".join(f"l{i}mot{j}" for j in range(longueur))
        n_sources = 1 + (i % 2)
        for k in range(n_sources):
            rows.append({"fr": f"source longue {i} variante {k}", "en": cible})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split_fixture_result() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Split (seed=42) calculé une seule fois pour les tests 3/4/5 (non-fuite,
    intégrité des groupes, stratification)."""
    df = build_split_fixture()
    with pytest.warns(UserWarning, match="groupes longs"):
        # la fixture (40 groupes longs) est volontairement plus petite que
        # MIN_GROUPES_LONGS_TEST=300 : l'avertissement doit se déclencher.
        splits, diagnostic = split_by_target_group(df, seed=42)

    n_longs_test = int(diagnostic.loc[diagnostic["part"] == "test", "n_groupes_longs"].iloc[0])
    assert n_longs_test < MIN_GROUPES_LONGS_TEST
    return splits, diagnostic, df


def test_split_non_fuite_de_cible_entre_les_trois_parts(
    split_fixture_result: tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame],
) -> None:
    """LE test qui compte : aucune cible EN commune entre train/val/test."""
    splits, _diag, _df = split_fixture_result
    train_en, val_en, test_en = (set(splits[p]["en"]) for p in ("train", "val", "test"))

    assert train_en & test_en == set()
    assert train_en & val_en == set()
    assert val_en & test_en == set()


def test_split_integrite_des_groupes(
    split_fixture_result: tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame],
) -> None:
    """Aucune cible EN (groupe) n'apparaît dans deux parts différentes, et tous
    les groupes de la fixture sont bien assignés à une part."""
    splits, _diag, df = split_fixture_result

    part_de_cible: dict[str, set[str]] = {}
    for part in ("train", "val", "test"):
        for cible in splits[part]["en"].unique():
            part_de_cible.setdefault(cible, set()).add(part)

    assert all(len(parts) == 1 for parts in part_de_cible.values())
    assert set(part_de_cible.keys()) == set(df["en"].unique())


def test_split_stratification_proportion_de_longs_homogene(
    split_fixture_result: tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame],
) -> None:
    """La proportion de groupes longs (> threshold) est comparable entre les
    trois parts (tolérance explicite : 5 points de pourcentage)."""
    _splits, diagnostic, _df = split_fixture_result
    tolerance = 0.05

    proportions = diagnostic.set_index("part")["proportion_longs"]
    ecart_max = proportions.max() - proportions.min()
    assert ecart_max <= tolerance, (
        f"Écart de proportion de longs trop élevé entre parts : {proportions.to_dict()}"
    )
    # toutes les parts contiennent au moins un groupe long (sinon la stratification
    # ne serait pas testée du tout)
    assert (diagnostic["n_groupes_longs"] > 0).all()


def test_split_determinisme_meme_graine_meme_resultat() -> None:
    """Deux appels avec la même graine donnent exactement le même split."""
    df = build_split_fixture()

    with pytest.warns(UserWarning, match="groupes longs"):
        splits_a, _diag_a = split_by_target_group(df, seed=42)
    with pytest.warns(UserWarning, match="groupes longs"):
        splits_b, _diag_b = split_by_target_group(df, seed=42)

    for part in ("train", "val", "test"):
        paires_a = sorted(zip(splits_a[part]["fr"], splits_a[part]["en"]))
        paires_b = sorted(zip(splits_b[part]["fr"], splits_b[part]["en"]))
        assert paires_a == paires_b


def test_split_ratios_invalides_leve_une_erreur() -> None:
    """Des ratios qui ne somment pas à 1.0 sont rejetés explicitement."""
    df = build_split_fixture()
    with pytest.raises(ValueError):
        split_by_target_group(df, ratios=(0.8, 0.1, 0.2))


# ---------------------------------------------------------------------------
# Comptage des mots : garde-fou anti-régression sur le piège pandas 3 / PyArrow / RE2
# ---------------------------------------------------------------------------
def test_count_words_apostrophe_et_accents() -> None:
    """`l'ami` = 2 unités (élision), `aujourd'hui` = 1 (exception insécable), et
    les caractères accentués sont bien traités comme des lettres (piège RE2/ASCII
    connu : `.str.count()` sur colonne PyArrow y traiterait
    "été" comme 1 caractère de mot au lieu de 3)."""
    assert count_words("l'ami", "fr") == 2
    assert count_words("aujourd'hui", "fr") == 1
    assert count_words("quelqu'un est là", "fr") == 3

    # mot 100% accentué : ne doit pas être ignoré (pas 0 mot)
    assert count_words("été", "fr") == 1
    assert tokenize_words("été", "fr") == ["été"]

    # cas combiné accents + apostrophe (repris du notebook AED, cellule 19)
    assert count_words("l'été à Paris", "fr") == 4
    # preuve que split() naïf sous-estime le français élidé (1 "mot" au lieu de 4)
    assert len("l'été à Paris".split()) == 3  # noqa: SIM905 - .split() naïf voulu ici, à titre de contraste


# ---------------------------------------------------------------------------
# Normalisation (minuscules + ponctuation de bord, apostrophe préservée)
# ---------------------------------------------------------------------------
def test_normalize_text_minuscules_et_ponctuation_de_bord() -> None:
    """Reproduit `normalizeToken` du notebook AED (cellule 34) : minuscules,
    ponctuation de bord retirée, apostrophe d'élision préservée comme frontière
    de mot ("l'ami." -> "l'" + "ami" -> "l' ami")."""
    assert normalize_text("The Cat.", "en") == "the cat"
    assert normalize_text("L'ami.", "fr") == "l' ami"
    assert normalize_text("Chose !", "fr") == "chose"
    assert normalize_text("Il fait beau aujourd'hui.", "fr") == "il fait beau aujourd'hui"


def test_normalize_corpus_journal_et_idempotence() -> None:
    """`normalize_corpus` normalise `fr` et `en`, journalise le nombre de
    lignes modifiées par langue, et est idempotent (une 2e passe ne modifie
    plus rien)."""
    df = pd.DataFrame(
        {
            "fr": ["Bonjour.", "déjà minuscule", "Salut !"],
            "en": ["Hello.", "already lowercase", "Hi!"],
        }
    )
    normalise, journal = normalize_corpus(df)

    assert list(normalise["fr"]) == ["bonjour", "déjà minuscule", "salut"]
    assert list(normalise["en"]) == ["hello", "already lowercase", "hi"]
    assert set(journal["langue"]) == {"fr", "en"}
    assert journal.set_index("langue").loc["fr", "n_modifiees"] == 2
    assert journal.set_index("langue").loc["en", "n_modifiees"] == 2

    normalise_deux_fois, journal_deux = normalize_corpus(normalise)
    assert normalise_deux_fois["fr"].equals(normalise["fr"])
    assert (journal_deux["n_modifiees"] == 0).all()


# ---------------------------------------------------------------------------
# Test d'intégration sur le corpus réel (sauté si data/processed/ est absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not DATA_PROCESSED_DIR.exists(),
    reason="data/processed absent : corpus réel non généré, test d'intégration sauté",
)
def test_load_splits_corpus_reel_non_fuite_et_stratification() -> None:
    """Points de contrôle chiffrés du plan sur le corpus réel déjà mis en cache
    (ne relance pas le pipeline : `force_rebuild=False` par défaut)."""
    from src.data.splits import load_splits

    splits = load_splits(force_rebuild=False)
    train, val, test = splits["train"], splits["val"], splits["test"]

    train_en, val_en, test_en = set(train["en"]), set(val["en"]), set(test["en"])
    assert len(train_en & test_en) == 0
    assert len(train_en & val_en) == 0
    assert len(val_en & test_en) == 0

    threshold = 18

    def proportion_longs(df: pd.DataFrame) -> float:
        cibles = df["en"].unique()
        n_longs = sum(1 for c in cibles if count_words(c, "en") > threshold)
        return n_longs / len(cibles)

    p_train, p_val, p_test = proportion_longs(train), proportion_longs(val), proportion_longs(test)
    assert abs(p_train - p_val) < 0.05
    assert abs(p_train - p_test) < 0.05
