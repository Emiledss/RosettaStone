"""Split train/val/test anti-fuite, groupé par cible EN (étape 3 — LE point critique).

3a. Non-fuite : chaque groupe = toutes les paires qui partagent la même phrase
    cible EN. Un groupe entier va dans une seule part (jamais de groupe à cheval).
3b. Stratification : la longueur d'un groupe = nb de mots de sa cible EN (via
    `count_words(..., "en")`) ; deux strates (`<= threshold` / `> threshold`) ;
    le ratio de split s'applique À L'INTÉRIEUR de chaque strate, au niveau des
    groupes, pour que train/val/test aient la même proportion de longues.
3c. Split en deux temps : (train+val) vs test, puis train vs val — implémenté ici
    par un mélange de la liste des groupes (`np.random.default_rng(seed)`) suivi
    d'un découpage par indices (deux points de coupe successifs), équivalent à
    deux tirages séquentiels sur une liste déjà uniformément mélangée.

`load_splits` orchestre les étapes 0→3 et met le résultat en cache en parquet.
Ordre imposé du pipeline interne : `clean_corpus` -> `normalize_corpus` ->
`deduplicate_pairs` -> `split_by_target_group` (voir commentaire dans
`load_splits` pour la justification des deux placements de la normalisation).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning import clean_corpus, deduplicate_pairs, normalize_corpus
from src.data.loader import load_raw_corpus
from src.data.text import count_words

# Nombre minimal de groupes longs recherché dans le test pour des métriques stables
# (cf. docs/plan-seq2seq.md, étape 3d : "viser >= quelques centaines").
MIN_GROUPES_LONGS_TEST = 300


def _points_de_coupe(n: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    """Calcule les deux points de coupe (fin de train, fin de train+val) sur une
    liste de taille `n` déjà mélangée aléatoirement.

    `n_train` est déduit par soustraction (pas arrondi indépendamment) pour que
    les trois tailles somment exactement à `n`.
    """
    n_test = round(n * ratios[2])
    n_val = round(n * ratios[1])
    n_train = n - n_val - n_test
    return n_train, n_train + n_val


def split_by_target_group(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    threshold: int = 18,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split train/val/test groupé par cible EN, stratifié par longueur de cible.

    Renvoie `(splits, diagnostic)` : `splits` a les clés `train`/`val`/`test`
    (mêmes colonnes que `df`), `diagnostic` a une ligne par part avec `n_paires`,
    `n_groupes`, `n_groupes_longs`, `proportion_longs`.

    Émet un `warnings.warn` (ne décide pas à la place de l'humain) si le nombre
    absolu de groupes longs dans le test tombe sous `MIN_GROUPES_LONGS_TEST`.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios doit sommer à 1.0, reçu {ratios}")

    rng = np.random.default_rng(seed)

    cibles = df["en"].unique()
    longueur_cible = {cible: count_words(cible, "en") for cible in cibles}

    strate_courte = [c for c in cibles if longueur_cible[c] <= threshold]
    strate_longue = [c for c in cibles if longueur_cible[c] > threshold]

    assignation: dict[str, str] = {}
    for groupes_strate in (strate_courte, strate_longue):
        melange = rng.permutation(np.array(groupes_strate, dtype=object))
        fin_train, fin_train_val = _points_de_coupe(len(melange), ratios)
        for cible in melange[:fin_train]:
            assignation[cible] = "train"
        for cible in melange[fin_train:fin_train_val]:
            assignation[cible] = "val"
        for cible in melange[fin_train_val:]:
            assignation[cible] = "test"

    part_par_ligne = df["en"].map(assignation)
    splits = {
        part: df.loc[part_par_ligne == part].reset_index(drop=True)
        for part in ("train", "val", "test")
    }

    diag_rows = []
    for part, part_df in splits.items():
        cibles_part = part_df["en"].unique()
        n_groupes = len(cibles_part)
        n_longs = sum(1 for c in cibles_part if longueur_cible[c] > threshold)
        diag_rows.append(
            {
                "part": part,
                "n_paires": len(part_df),
                "n_groupes": n_groupes,
                "n_groupes_longs": n_longs,
                "proportion_longs": (n_longs / n_groupes) if n_groupes else 0.0,
            }
        )
    diagnostic = pd.DataFrame(diag_rows)

    n_longs_test = int(diagnostic.loc[diagnostic["part"] == "test", "n_groupes_longs"].iloc[0])
    if n_longs_test < MIN_GROUPES_LONGS_TEST:
        warnings.warn(
            f"Seulement {n_longs_test} groupes longs dans le test "
            f"(recommandé : >= {MIN_GROUPES_LONGS_TEST}). "
            "Envisager ratios=(0.7, 0.15, 0.15) pour mieux peupler le bin long "
            "(voir docs/plan-seq2seq.md, étape 3d).",
            stacklevel=2,
        )

    return splits, diagnostic


def load_splits(
    force_rebuild: bool = False,
    strategy: str = "auto",
    data_dir: str = "data",
    processed_dir: str = "data/processed",
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    threshold: int = 18,
    seed: int = 42,
    normalize: bool = True,
) -> dict[str, pd.DataFrame]:
    """Orchestre les étapes 0→3 et met le résultat en cache dans `processed_dir`.

    Relit le cache (`{train,val,test}.parquet`) s'il existe déjà et que
    `force_rebuild` est faux. Renvoie un dict avec les clés `train`/`val`/`test`,
    colonnes `fr` et `en`.
    """
    processed_path = Path(processed_dir)
    fichiers_cache = {part: processed_path / f"{part}.parquet" for part in ("train", "val", "test")}

    if not force_rebuild and all(f.exists() for f in fichiers_cache.values()):
        print(f"[etape 3] cache trouvé dans {processed_path}, relecture (force_rebuild=False).")
        return {part: pd.read_parquet(chemin) for part, chemin in fichiers_cache.items()}

    brut = load_raw_corpus(strategy=strategy, data_dir=data_dir)

    nettoye, journal = clean_corpus(brut)
    n_retirees = len(brut) - len(nettoye)
    print(f"[etape 1] nettoyage: {n_retirees} paires retirées ({n_retirees / len(brut):.2%})")
    print(journal.to_string(index=False))

    # Normalisation (minuscules + ponctuation de bord retirée, apostrophe préservée)
    # AVANT le split ET avant la déduplication -- ordre imposé, deux raisons :
    # - avant le split : le regroupement anti-fuite (étape 3) se fait sur la cible EN
    #   telle quelle. Sans normalisation préalable, "The cat." et "the cat" seraient
    #   deux groupes distincts, susceptibles de tomber de part et d'autre du split
    #   -> fuite entre train/val/test.
    # - avant la dédup : la normalisation crée de nouveaux doublons exacts (casse et
    #   ponctuation de bord neutralisées) que la déduplication doit ensuite absorber.
    #   Le compte de doublons monte donc au-dessus de 502 (corpus non normalisé) --
    #   c'est l'effet attendu de la normalisation, pas une régression.
    normalisee = nettoye
    if normalize:
        normalisee, journal_normalisation = normalize_corpus(nettoye)
        print("[etape 1bis] normalisation (minuscules + ponctuation de bord):")
        print(journal_normalisation.to_string(index=False))

    dedupliquee, n_doublons = deduplicate_pairs(normalisee)
    print(f"[etape 2] {n_doublons} doublons exacts retirés, {len(dedupliquee)} paires restantes")

    splits, diagnostic = split_by_target_group(dedupliquee, ratios=ratios, threshold=threshold, seed=seed)
    print(f"[etape 3] split terminé:\n{diagnostic.to_string(index=False)}")

    processed_path.mkdir(parents=True, exist_ok=True)
    for part, part_df in splits.items():
        part_df[["fr", "en"]].to_parquet(fichiers_cache[part], index=False)
    print(f"[etape 3] cache écrit dans {processed_path}")

    return {part: part_df[["fr", "en"]] for part, part_df in splits.items()}
