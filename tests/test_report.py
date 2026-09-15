"""Tests des analyses de l'étape 10 (`src/evaluation/report.py`) : biais du
teacher forcing (Q14a), échantillon d'exemples par bin (Q14d), désaccords
entre métriques.

`corpus_meteor` (utilisé en interne par `metric_disagreements`) est monkeypatché
au niveau du module `src.evaluation.report` (qui l'importe par `from ... import
corpus_meteor`, donc lie son propre nom) -- aucun de ces tests ne dépend du
réseau ni de la ressource nltk `wordnet`.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.evaluation.report as report_module
from src.evaluation.report import (
    metric_disagreements,
    sample_predictions,
    teacher_forcing_bias,
)


# ---------------------------------------------------------------------------
# teacher_forcing_bias
# ---------------------------------------------------------------------------
class _BiasedFakeModel(nn.Module):
    """Faux `Seq2Seq.forward(src, tgt, teacher_forcing_ratio)` : reproduit
    EXACTEMENT `tgt` en teacher forcing pur (accuracy 100%) ; en génération
    libre, ne reproduit correctement que la position 1 (même entrée `<sos>`
    dans les deux régimes), puis émet systématiquement un token FAUX --
    simule l'exposure bias de façon déterministe, sans dépendre de la
    convergence d'un entraînement réel."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size)
        for t in range(1, tgt_len):
            if teacher_forcing_ratio >= 1.0 or t == 1:
                correct = tgt[:, t]
            else:
                correct = (tgt[:, t] + 1) % self.vocab_size  # toujours faux
            outputs[torch.arange(batch_size), t, correct] = 10.0
        return outputs


def test_teacher_forcing_bias_accuracy_teacher_forcing_superieure_a_generation_libre() -> None:
    vocab_size = 6
    pad_id = 0
    torch.manual_seed(0)
    tgt = torch.randint(1, vocab_size, (8, 5))  # jamais <pad> (0)
    src = torch.zeros(8, 4, dtype=torch.long)
    loader = DataLoader(TensorDataset(src, tgt), batch_size=4)
    model = _BiasedFakeModel(vocab_size)

    resultat = teacher_forcing_bias(model, loader, torch.device("cpu"), pad_id=pad_id)

    assert resultat["teacherForcingAccuracy"] == pytest.approx(1.0)
    assert resultat["freeRunningAccuracy"] < resultat["teacherForcingAccuracy"]
    assert resultat["gap"] > 0.3  # écart net (position 1/4 correcte en génération libre)
    assert resultat["nTokens"] == 8 * 4  # 4 positions cibles (tgt_len=5, position 0 exclue), 8 exemples


def test_teacher_forcing_bias_ignore_le_padding() -> None:
    """Une cible entièrement `<pad>` ne doit pas fausser l'accuracy (ni
    compter dans `nTokens`)."""
    vocab_size = 6
    pad_id = 0
    tgt_reel = torch.tensor([[2, 1, 3, 4]])
    tgt_pad = torch.zeros(1, 4, dtype=torch.long)
    tgt = torch.cat([tgt_reel, tgt_pad], dim=0)  # (2, 4) -- 1 exemple réel, 1 tout <pad>
    src = torch.zeros(2, 3, dtype=torch.long)
    loader = DataLoader(TensorDataset(src, tgt), batch_size=2)
    model = _BiasedFakeModel(vocab_size)

    resultat = teacher_forcing_bias(model, loader, torch.device("cpu"), pad_id=pad_id)

    assert resultat["nTokens"] == 3  # positions 1..3 de l'exemple réel seulement
    assert resultat["teacherForcingAccuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sample_predictions
# ---------------------------------------------------------------------------
def test_sample_predictions_couvre_les_deux_bins_et_bonnes_colonnes() -> None:
    sources = [f"src{i}" for i in range(10)]
    reference_courte = "a b c"  # 3 mots
    reference_longue = " ".join(f"w{i}" for i in range(20))  # 20 mots
    references = [reference_courte] * 5 + [reference_longue] * 5
    hypotheses = [f"hyp{i}" for i in range(10)]

    df = sample_predictions(sources, hypotheses, references, n=6, seed=0, threshold=18)

    assert set(df.columns) == {"bin", "source", "hypothese", "reference", "longueurReference"}
    assert (df["bin"] == "courte").sum() == 3
    assert (df["bin"] == "longue").sum() == 3
    assert len(df) == 6
    assert set(df["source"]).issubset(set(sources))


def test_sample_predictions_reequilibre_si_un_bin_sous_peuple() -> None:
    sources = [f"src{i}" for i in range(5)]
    references = ["a b c"] * 5  # que des courtes -- bin longue vide
    hypotheses = [f"hyp{i}" for i in range(5)]

    df = sample_predictions(sources, hypotheses, references, n=6, seed=1, threshold=18)

    assert (df["bin"] == "longue").sum() == 0
    assert len(df) == 5  # borné par le nombre de phrases courtes disponibles (rééquilibrage)


def test_sample_predictions_erreur_si_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        sample_predictions(["s1", "s2"], ["h1"], ["r1", "r2"])


# ---------------------------------------------------------------------------
# metric_disagreements
# ---------------------------------------------------------------------------
def test_metric_disagreements_detecte_bleu_bas_meteor_haut(monkeypatch) -> None:
    monkeypatch.setattr(report_module, "corpus_meteor", lambda hyps, refs: 0.9)  # accord fort, peu importe la paire

    hypotheses = ["a completely different sentence", "the cat sat on the mat"]
    references = ["the cat sat on the mat", "the cat sat on the mat"]
    sources = ["src1", "src2"]

    df = metric_disagreements(hypotheses, references, sources=sources, bleu_low=20.0, meteor_high=0.5)

    assert len(df) == 1
    assert df.iloc[0]["source"] == "src1"
    assert bool(df.iloc[0]["desaccordMeteor"]) is True
    assert bool(df.iloc[0]["desaccordBertscore"]) is False


def test_metric_disagreements_utilise_bertscore_f1_fourni(monkeypatch) -> None:
    monkeypatch.setattr(report_module, "corpus_meteor", lambda hyps, refs: None)  # meteor indisponible

    hypotheses = ["a completely different sentence"]
    references = ["the cat sat on the mat"]
    sources = ["src1"]

    df = metric_disagreements(
        hypotheses, references, sources=sources, bertscore_f1=[0.95], bleu_low=20.0, bertscore_high=0.85
    )

    assert len(df) == 1
    assert bool(df.iloc[0]["desaccordBertscore"]) is True
    assert bool(df.iloc[0]["desaccordMeteor"]) is False


def test_metric_disagreements_vide_si_aucun_desaccord(monkeypatch) -> None:
    monkeypatch.setattr(report_module, "corpus_meteor", lambda hyps, refs: 0.1)  # accord faible partout

    hypotheses = ["a completely different sentence"]
    references = ["the cat sat on the mat"]

    df = metric_disagreements(hypotheses, references, bleu_low=20.0, meteor_high=0.5)

    assert df.empty


def test_metric_disagreements_erreur_si_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        metric_disagreements(["h1", "h2"], ["r1"])
