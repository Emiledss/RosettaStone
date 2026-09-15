"""Analyses de l'étape 10 : biais du teacher forcing (Q14a), exemples de
traductions par bin de longueur (Q14d), désaccords entre métriques (analyse
n°1 attendue par le plan -- typiquement les synonymes que SacreBLEU pénalise
et que METEOR/BERTScore acceptent).
"""

from __future__ import annotations

import random

import pandas as pd
import sacrebleu
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.text import count_words
from src.evaluation.metrics import corpus_meteor


def teacher_forcing_bias(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
) -> dict:
    """Compare, sur le MÊME modèle et le MÊME dataloader, l'accuracy
    token-à-token en teacher forcing pur (`teacher_forcing_ratio=1.0`, chaque
    pas reçoit le vrai token précédent) à celle en génération libre
    (`teacher_forcing_ratio=0.0`, chaque pas ne reçoit que sa propre
    prédiction précédente).

    C'est le biais du teacher forcing (Q14a du barème) : la version teacher
    forcing ne peut jamais "dérailler" (elle reçoit toujours le bon contexte),
    la version libre si -- l'écart mesure ce biais chiffres à l'appui, il ne
    s'agit PAS d'un bug à corriger.

    L'accuracy est calculée position par position contre `tgt` (comparaison
    directe, pas d'alignement), en ignorant les positions `<pad>`
    (`pad_id`) -- même convention que la loss de `src.training.loop`.

    Renvoie `{"teacherForcingAccuracy", "freeRunningAccuracy", "gap", "nTokens"}`.
    """
    model.eval()
    correct_tf = 0
    correct_free = 0
    total = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            output_tf = model(src, tgt, teacher_forcing_ratio=1.0)
            output_free = model(src, tgt, teacher_forcing_ratio=0.0)

            target = tgt[:, 1:]
            mask = target != pad_id

            pred_tf = output_tf[:, 1:, :].argmax(dim=-1)
            pred_free = output_free[:, 1:, :].argmax(dim=-1)

            correct_tf += int(((pred_tf == target) & mask).sum().item())
            correct_free += int(((pred_free == target) & mask).sum().item())
            total += int(mask.sum().item())

    tf_accuracy = correct_tf / total if total else float("nan")
    free_accuracy = correct_free / total if total else float("nan")

    return {
        "teacherForcingAccuracy": tf_accuracy,
        "freeRunningAccuracy": free_accuracy,
        "gap": tf_accuracy - free_accuracy,
        "nTokens": total,
    }


def sample_predictions(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    n: int = 20,
    seed: int = 42,
    threshold: int = 18,
) -> pd.DataFrame:
    """Échantillonne `n` triplets source/hypothèse/référence pour inspection
    manuelle (Q14d), en couvrant les DEUX bins de longueur (mesurée sur la
    référence EN, seuil `threshold`) -- pour moitié dans chaque bin quand les
    deux sont assez peuplés, rééquilibré sur le bin disponible sinon.

    Colonnes : `bin`, `source`, `hypothese`, `reference`, `longueurReference`.
    """
    if not (len(sources) == len(hypotheses) == len(references)):
        raise ValueError("sources, hypotheses et references doivent avoir la même longueur")

    rng = random.Random(seed)
    longueurs = [count_words(reference, "en") for reference in references]
    indices_courts = [i for i, l in enumerate(longueurs) if l <= threshold]
    indices_longs = [i for i, l in enumerate(longueurs) if l > threshold]

    n_courts_vise = n // 2
    n_longs_vise = n - n_courts_vise
    n_courts = min(n_courts_vise, len(indices_courts))
    n_longs = min(n_longs_vise, len(indices_longs))

    manque = (n_courts_vise - n_courts) + (n_longs_vise - n_longs)
    if manque > 0 and n_courts < len(indices_courts):
        supplement = min(manque, len(indices_courts) - n_courts)
        n_courts += supplement
        manque -= supplement
    if manque > 0 and n_longs < len(indices_longs):
        n_longs += min(manque, len(indices_longs) - n_longs)

    choisis_courts = rng.sample(indices_courts, n_courts) if n_courts else []
    choisis_longs = rng.sample(indices_longs, n_longs) if n_longs else []

    lignes = []
    for label, indices in (("courte", choisis_courts), ("longue", choisis_longs)):
        for i in indices:
            lignes.append(
                {
                    "bin": label,
                    "source": sources[i],
                    "hypothese": hypotheses[i],
                    "reference": references[i],
                    "longueurReference": longueurs[i],
                }
            )

    return pd.DataFrame(lignes)


def metric_disagreements(
    hypotheses: list[str],
    references: list[str],
    sources: list[str] | None = None,
    bertscore_f1: list[float] | None = None,
    bleu_low: float = 20.0,
    meteor_high: float = 0.5,
    bertscore_high: float = 0.85,
    top_n: int = 20,
) -> pd.DataFrame:
    """Repère les phrases où SacreBLEU (calculé PHRASE PAR PHRASE) est bas
    alors que METEOR (et/ou BERTScore, si `bertscore_f1` fourni) est élevé --
    désaccord typique des synonymes/paraphrases, l'analyse n°1 attendue par
    l'étape 10 (SacreBLEU est sévère sur les synonymes, METEOR/BERTScore sont
    plus souples).

    `bertscore_f1`, si fourni, doit être une liste de scores F1 PHRASE PAR
    PHRASE déjà calculés par l'appelant (BERTScore est trop coûteux pour être
    recalculé ici phrase par phrase à chaque appel) -- typiquement sur
    l'échantillon stratifié ~3000 paires de l'étape 10, pas le test complet.

    Renvoie un `DataFrame` trié par écart décroissant (le score alternatif le
    plus haut malgré un SacreBLEU bas en premier), limité à `top_n` lignes.
    Colonnes : `index`, `source`, `hypothese`, `reference`, `sacrebleu`,
    `meteor`, `bertscoreF1`, `desaccordMeteor`, `desaccordBertscore`.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses et references doivent avoir la même longueur")
    n = len(hypotheses)
    if sources is None:
        sources = [""] * n
    if bertscore_f1 is not None and len(bertscore_f1) != n:
        raise ValueError("bertscore_f1 doit avoir la même longueur que hypotheses")

    bleus_phrase = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypotheses, references)
    ]
    meteors_phrase = [
        corpus_meteor([hypothesis], [reference]) for hypothesis, reference in zip(hypotheses, references)
    ]

    lignes = []
    for i in range(n):
        bleu = bleus_phrase[i]
        meteor = meteors_phrase[i]
        bscore = bertscore_f1[i] if bertscore_f1 is not None else None

        desaccord_meteor = meteor is not None and bleu < bleu_low and meteor >= meteor_high
        desaccord_bertscore = bscore is not None and bleu < bleu_low and bscore >= bertscore_high

        if not (desaccord_meteor or desaccord_bertscore):
            continue

        ecart = max(
            (meteor if meteor is not None else 0.0),
            (bscore if bscore is not None else 0.0),
        ) - (bleu / 100.0)

        lignes.append(
            {
                "index": i,
                "source": sources[i],
                "hypothese": hypotheses[i],
                "reference": references[i],
                "sacrebleu": bleu,
                "meteor": meteor if meteor is not None else float("nan"),
                "bertscoreF1": bscore if bscore is not None else float("nan"),
                "desaccordMeteor": desaccord_meteor,
                "desaccordBertscore": desaccord_bertscore,
                "ecart": ecart,
            }
        )

    df = pd.DataFrame(lignes)
    if not df.empty:
        df = df.sort_values("ecart", ascending=False).head(top_n).reset_index(drop=True)
        df = df.drop(columns=["ecart"])
    return df
