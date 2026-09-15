"""Nettoyage ordonné et déduplication exacte du corpus (étapes 1 et 2).

Le comptage de mots utilisé pour le ratio de longueur et le seuil `max_words`
passe par `src.data.text.count_words` (Python pur, Unicode-correct) — jamais par
`.str.count()` / `.str.contains()` sur les colonnes pandas (piège pandas 3 /
PyArrow / RE2 : la classe "mot" de ce moteur n'est pas Unicode).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.text import count_words, normalize_text


def clean_corpus(
    df: pd.DataFrame,
    min_ratio: float = 1.0 / 3.0,
    max_ratio: float = 3.0,
    max_words: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nettoie le corpus en 4 sous-étapes, dans cet ordre imposé :

    1. chaînes vides (`len(strip()) == 0`, fr ou en) ;
    2. paires `en == fr` ;
    3. ratio de longueur en mots hors `[min_ratio, max_ratio]` ;
    4. longueur en mots (fr ou en) supérieure à `max_words`.

    Renvoie `(corpus_nettoye, journal)` où `journal` a une ligne par sous-étape
    (`etape`, `n_retirees`, `n_restantes`).
    """
    current = df.reset_index(drop=True)
    log_rows: list[dict[str, object]] = []

    # 1. chaînes vides
    fr_vide = current["fr"].astype(str).str.strip().eq("")
    en_vide = current["en"].astype(str).str.strip().eq("")
    est_vide = fr_vide | en_vide
    current = current.loc[~est_vide].reset_index(drop=True)
    log_rows.append(
        {"etape": "chaines_vides", "n_retirees": int(est_vide.sum()), "n_restantes": len(current)}
    )

    # 2. en == fr
    est_identique = current["fr"] == current["en"]
    current = current.loc[~est_identique].reset_index(drop=True)
    log_rows.append(
        {"etape": "en_egal_fr", "n_retirees": int(est_identique.sum()), "n_restantes": len(current)}
    )

    # 3. ratio de longueur (en mots) hors intervalle
    mots_fr = current["fr"].map(lambda t: count_words(t, "fr")).astype(float)
    mots_en = current["en"].map(lambda t: count_words(t, "en")).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mots_fr / mots_en
    ratio_aberrant = ~np.isfinite(ratio) | (ratio < min_ratio) | (ratio > max_ratio)
    current = current.loc[~ratio_aberrant].reset_index(drop=True)
    log_rows.append(
        {"etape": "ratio_longueur", "n_retirees": int(ratio_aberrant.sum()), "n_restantes": len(current)}
    )

    # 4. max_words
    mots_fr = current["fr"].map(lambda t: count_words(t, "fr"))
    mots_en = current["en"].map(lambda t: count_words(t, "en"))
    trop_long = (mots_fr > max_words) | (mots_en > max_words)
    current = current.loc[~trop_long].reset_index(drop=True)
    log_rows.append(
        {"etape": "max_words", "n_retirees": int(trop_long.sum()), "n_restantes": len(current)}
    )

    return current, pd.DataFrame(log_rows)


def normalize_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise les colonnes `fr` et `en` (`src.data.text.normalize_text` :
    minuscules + retrait de la ponctuation de bord, apostrophe préservée).

    Renvoie `(df_normalise, journal)` où `journal` a une ligne par langue avec
    le nombre de lignes effectivement modifiées (`n_modifiees`) sur le total
    (`n_total`).
    """
    current = df.reset_index(drop=True).copy()
    log_rows: list[dict[str, object]] = []

    for lang in ("fr", "en"):
        normalise = current[lang].map(lambda texte, lang=lang: normalize_text(texte, lang))
        n_modifiees = int((normalise != current[lang]).sum())
        current[lang] = normalise
        log_rows.append({"langue": lang, "n_modifiees": n_modifiees, "n_total": len(current)})

    return current, pd.DataFrame(log_rows)


def deduplicate_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Retire les doublons exacts sur `(fr, en)`.

    Ne déduplique PAS sur une seule langue : les alignements 1→N (une même cible
    EN pour plusieurs sources FR) sont légitimes et gérés par le split (étape 3),
    pas supprimés ici.
    """
    est_doublon = df.duplicated(subset=["fr", "en"])
    deduped = df.loc[~est_doublon].reset_index(drop=True)
    return deduped, int(est_doublon.sum())
