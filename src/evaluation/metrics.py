"""Métriques d'évaluation (étape 10) : SacreBLEU, METEOR, BERTScore, et
ventilation par bin de longueur.

Les métriques doivent porter sur du texte RÉELLEMENT généré (génération libre,
`src.evaluation.generate.generate_translations`), jamais en teacher forcing.

METEOR (nltk) et BERTScore (modèle HuggingFace) ont besoin d'une ressource
téléchargée (`nltk.download('wordnet')`) ou d'un modèle téléchargé
respectivement -- accès réseau la première fois. Les deux fonctions
correspondantes se dégradent PROPREMENT (renvoient `None`, impriment un
avertissement `[etape 10]`) si la ressource est indisponible : jamais de
plantage pour une métrique secondaire.
"""

from __future__ import annotations

import bert_score

# Références de fonctions au niveau module (plutôt que des imports internes à
# chaque fonction) : permet aux tests de simuler une ressource indisponible par
# monkeypatch, sans dépendre du réseau ni de l'état du cache nltk de la machine.
import nltk
import pandas as pd
import sacrebleu
from nltk.translate.meteor_score import meteor_score as _meteor_score_fn

from src.data.text import count_words


def corpus_sacrebleu(hypotheses: list[str], references: list[str]) -> float:
    """SacreBLEU corpus-level entre `hypotheses` et `references` (une référence
    par phrase), en minuscules (`lowercase=True`).

    Le corpus Rosetta est normalisé en minuscules dès l'étape 1bis : scorer en
    sensible à la casse pénaliserait uniformément tous les modèles
    pour une distinction qu'aucun d'eux ne peut produire (leurs cibles
    d'entraînement sont déjà en minuscules).

    Attention -- comparabilité : les valeurs renvoyées ici ne sont PAS
    comparables aux scores SacreBLEU publiés sur Tatoeba. Le corpus normalisé
    de ce projet sépare les clitiques avant tokenisation mots entiers (ex.
    "let's" -> "let 's") ; hypothèses ET références héritent de
    cette même convention, donc la comparaison ENTRE modèles/configs de ce
    projet reste valide (même texte de référence des deux côtés) -- mais pas
    la comparaison avec un score BLEU externe calculé sur un texte non retouché.

    Renvoie le score BLEU (0-100, `sacrebleu.corpus_bleu(...).score`).
    """
    resultat = sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True)
    return resultat.score


def _ensure_wordnet() -> bool:
    """Vérifie (et tente de télécharger si besoin) la ressource nltk
    `wordnet`, nécessaire à METEOR. Ne lève JAMAIS : renvoie `False` et
    imprime un avertissement `[etape 10]` si la ressource reste indisponible
    (pas de réseau, échec de téléchargement...)."""
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        pass
    try:
        nltk.download("wordnet", quiet=True)
        nltk.data.find("corpora/wordnet")
        return True
    except Exception as exc:  # noqa: BLE001 -- dégradation volontaire, jamais planter (réseau/téléchargement)
        print(f"[etape 10] METEOR indisponible : ressource nltk 'wordnet' introuvable ({exc}).")
        return False


def corpus_meteor(hypotheses: list[str], references: list[str]) -> float | None:
    """METEOR corpus-level (moyenne des scores phrase par phrase, via
    `nltk.translate.meteor_score.meteor_score`), une référence par phrase.

    `hypotheses`/`references` sont déjà tokenisées par espaces (convention du
    projet, cf. `WordVocabulary.decode`/corpus normalisé) : `str.split()`
    suffit, pas besoin d'un tokeniseur nltk séparé.

    Renvoie `None` (sans lever) si la ressource nltk `wordnet` est
    indisponible -- jamais de plantage pour cette métrique secondaire.
    """
    if not hypotheses:
        return None
    if not _ensure_wordnet():
        return None
    try:
        scores = [
            _meteor_score_fn([reference.split()], hypothesis.split())
            for hypothesis, reference in zip(hypotheses, references)
        ]
    except LookupError as exc:  # pragma: no cover -- ressource retirée en cours de route
        print(f"[etape 10] METEOR indisponible en cours de calcul : {exc}.")
        return None
    return sum(scores) / len(scores)


def corpus_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    num_layers: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict | None:
    """BERTScore corpus-level (moyennes precision/recall/F1), via un modèle
    léger (`distilbert-base-uncased` par défaut -- roberta-large est hors
    d'atteinte en CPU sur ce poste).

    `num_layers` est fixé EXPLICITEMENT (bert-score ne connaît pas la couche
    par défaut de tous les modèles ; 5 est la couche recommandée pour
    distilbert-base-uncased).

    Renvoie `None` (sans lever) si le modèle ne peut pas être chargé/téléchargé
    (pas de réseau, cache HuggingFace absent...) -- jamais de plantage pour
    cette métrique, la plus coûteuse des trois.
    """
    if not hypotheses:
        return None
    try:
        precision, recall, f1 = bert_score.score(
            hypotheses,
            references,
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 -- dégradation volontaire, jamais planter (réseau/téléchargement)
        print(f"[etape 10] BERTScore indisponible : modèle '{model_type}' inaccessible ({exc}).")
        return None
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }


def evaluate_by_length_bin(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    threshold: int = 18,
    compute_bertscore: bool = True,
    bertscore_model_type: str = "distilbert-base-uncased",
) -> pd.DataFrame:
    """Ventile SacreBLEU/METEOR/BERTScore en deux bins de longueur, mesurée
    sur la RÉFÉRENCE EN (`count_words(reference, "en")`, cohérent avec le split
    de l'étape 3) : courtes (`<= threshold` mots) / longues (`> threshold`).

    C'est sur le bin long que l'écart RNN/GRU et l'effet de l'attention
    doivent apparaître (docs/plan-seq2seq.md, étape 10).

    `compute_bertscore=False` permet d'appeler cette fonction sur le test
    complet sans payer le coût de BERTScore (à réserver à l'échantillon
    stratifié ~3000 paires, cf. plan).

    Un bin vide (aucune phrase de cette longueur) donne des `NaN` pour toutes
    ses métriques -- jamais de plantage.
    """
    if not (len(hypotheses) == len(references) == len(sources)):
        raise ValueError("hypotheses, references et sources doivent avoir la même longueur")

    longueurs = [count_words(reference, "en") for reference in references]
    bins = {
        "courte": [i for i, l in enumerate(longueurs) if l <= threshold],
        "longue": [i for i, l in enumerate(longueurs) if l > threshold],
    }

    lignes = []
    for label, indices in bins.items():
        sous_hyp = [hypotheses[i] for i in indices]
        sous_ref = [references[i] for i in indices]

        if not indices:
            lignes.append(
                {
                    "bin": label,
                    "n": 0,
                    "sacrebleu": float("nan"),
                    "meteor": float("nan"),
                    "bertscorePrecision": float("nan"),
                    "bertscoreRecall": float("nan"),
                    "bertscoreF1": float("nan"),
                }
            )
            continue

        bleu = corpus_sacrebleu(sous_hyp, sous_ref)
        meteor = corpus_meteor(sous_hyp, sous_ref)
        bertscore = (
            corpus_bertscore(sous_hyp, sous_ref, model_type=bertscore_model_type)
            if compute_bertscore
            else None
        )

        lignes.append(
            {
                "bin": label,
                "n": len(indices),
                "sacrebleu": bleu,
                "meteor": meteor if meteor is not None else float("nan"),
                "bertscorePrecision": bertscore["precision"] if bertscore else float("nan"),
                "bertscoreRecall": bertscore["recall"] if bertscore else float("nan"),
                "bertscoreF1": bertscore["f1"] if bertscore else float("nan"),
            }
        )

    return pd.DataFrame(lignes)
