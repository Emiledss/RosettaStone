"""Tests des lots F1 et F2 (`src/evaluation`) : génération libre
(`generate.py`, y compris la variante REPRENABLE) et métriques (`metrics.py`
-- SacreBLEU, METEOR, BERTScore, ventilation par bin de longueur).

`generate_translations` est testée avec deux modèles :
- un modèle « factice » (`_FakeSeq2Seq`), décodeur à séquence de tokens
  imposée, pour rendre déterministe (donc non flaky) le test de troncature à
  `<eos>` et le test du `batch_callback` ;
- un vrai `build_model` (graine fixée) pour le test de divergence avec le
  teacher forcing -- seul un modèle réel expose `Seq2Seq.forward`, nécessaire
  pour produire la référence "teacher forcing".

`corpus_meteor`/`corpus_bertscore` sont monkeypatchés au niveau de leurs
fonctions internes (`_ensure_wordnet`, `_meteor_score_fn`, `bert_score.score`)
-- aucun test de ce fichier ne dépend du réseau ni de la ressource nltk
`wordnet` déjà présente (ou non) sur la machine.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.evaluation.metrics as metrics_module
from src.evaluation.generate import (
    generate_translations,
    generate_translations_resumable,
)
from src.evaluation.metrics import (
    corpus_bertscore,
    corpus_meteor,
    corpus_sacrebleu,
    evaluate_by_length_bin,
)
from src.models.seq2seq import build_model
from src.tokenization.vocab_builder import WordVocabulary


# ---------------------------------------------------------------------------
# Vocabulaire jouet, partagé par les tests de génération.
# ---------------------------------------------------------------------------
def _vocabulaire_jouet() -> WordVocabulary:
    textes = pd.Series(["a b c d", "b c d a", "a c d b", "d c b a"])
    return WordVocabulary.fit(textes, lang="en", config_name="test-generate", coverage=1.0)


# ---------------------------------------------------------------------------
# Modèle factice : décodeur à séquence de tokens imposée (indépendante de
# l'entrée), pour un test déterministe de la troncature à `<eos>`.
# ---------------------------------------------------------------------------
class _FakeEncoder(nn.Module):
    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, src_len = src.shape
        hidden = torch.zeros(1, batch_size, 1)
        encoder_outputs = torch.zeros(batch_size, src_len, 1)
        return encoder_outputs, hidden


class _FakeDecoder(nn.Module):
    """Émet, à chaque pas, le token de `sequence[step]` -- même prédiction pour
    tout le batch, indépendante de `input_token`/`hidden`/`encoder_outputs`."""

    def __init__(self, sequence: list[int], vocab_size: int) -> None:
        super().__init__()
        self.sequence = sequence
        self.vocab_size = vocab_size
        self.step = 0

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_token.shape[0]
        token_id = self.sequence[min(self.step, len(self.sequence) - 1)]
        self.step += 1
        prediction = torch.full((batch_size, self.vocab_size), -10.0)
        prediction[:, token_id] = 10.0  # logit dominant -> argmax = token_id
        return prediction, hidden


class _FakeSeq2Seq(nn.Module):
    def __init__(self, decoder_sequence: list[int], vocab_size: int) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder(decoder_sequence, vocab_size)


def _loader_source_jouet(n_exemples: int, batch_size: int, src_len: int = 3, tgt_len: int = 6) -> DataLoader:
    """DataLoader (src, tgt) sans mélange -- seule la FORME de `tgt` est utilisée
    par `generate_translations` (dérivation de `max_len`), jamais son contenu."""
    src = torch.zeros(n_exemples, src_len, dtype=torch.long)
    tgt = torch.zeros(n_exemples, tgt_len, dtype=torch.long)
    return DataLoader(TensorDataset(src, tgt), batch_size=batch_size, shuffle=False)


def test_generation_libre_s_arrete_a_eos_et_ignore_pad_et_ne_contient_jamais_les_tokens_speciaux() -> None:
    """La séquence imposée contient un `<pad>` avant l'`<eos>` -- `<pad>` est
    ignoré (filtré au décodage), et tout ce qui suit l'`<eos>` n'est JAMAIS
    généré (troncature, pas un simple filtrage a posteriori)."""
    en_vocab = _vocabulaire_jouet()
    id_b = en_vocab.encode("b")[0]
    id_c = en_vocab.encode("c")[0]
    id_d = en_vocab.encode("d")[0]  # ne doit jamais apparaître : après l'<eos>

    sequence = [id_b, en_vocab.pad_id, id_c, en_vocab.eos_id, id_d, id_d]
    model = _FakeSeq2Seq(sequence, vocab_size=len(en_vocab))

    loader = _loader_source_jouet(n_exemples=1, batch_size=1, tgt_len=8)
    hypotheses = generate_translations(model, loader, en_vocab, torch.device("cpu"))

    assert hypotheses == ["b c"]
    for token_texte in ("<pad>", "<sos>", "<eos>"):
        assert token_texte not in hypotheses[0]


def test_generation_libre_diverge_du_teacher_forcing() -> None:
    """Sur un modèle réel (`build_model`), la génération libre (jamais nourrie
    du vrai token précédent) diverge de la même passe en teacher forcing pur
    (`teacher_forcing_ratio=1.0`) -- preuve que `generate_translations` ne
    triche pas."""
    torch.manual_seed(7)
    fr_vocab_size, en_vocab_size = 20, 20
    model = build_model(fr_vocab_size, en_vocab_size, emb_dim=8, hidden_dim=16, cell_type="gru", dropout=0.0)
    model.eval()

    en_vocab = _vocabulaire_jouet()  # taille réelle non alignée avec en_vocab_size : ok, decode() gère l'OOV

    torch.manual_seed(0)
    n_exemples, batch_size, src_len, tgt_len = 4, 4, 5, 7
    src = torch.randint(0, fr_vocab_size, (n_exemples, src_len))
    tgt = torch.randint(0, en_vocab_size, (n_exemples, tgt_len))
    tgt[:, 0] = 2  # <sos> en position 0, convention numerize.py

    with torch.no_grad():
        sortie_teacher_forcing = model(src, tgt, teacher_forcing_ratio=1.0)
    hypotheses_teacher_forcing = [
        en_vocab.decode(sortie_teacher_forcing[i, 1:, :].argmax(dim=-1).tolist()) for i in range(n_exemples)
    ]

    loader = DataLoader(TensorDataset(src, tgt), batch_size=batch_size, shuffle=False)
    hypotheses_generation_libre = generate_translations(model, loader, en_vocab, torch.device("cpu"))

    assert hypotheses_generation_libre != hypotheses_teacher_forcing


def test_batch_callback_appele_une_fois_par_batch_avec_les_bons_indices() -> None:
    """`batch_callback(index_debut, hypotheses_du_batch)` est appelé une fois
    par batch, avec l'indice de la première phrase du batch dans la sortie
    globale et les hypothèses effectivement produites pour ce batch."""
    en_vocab = _vocabulaire_jouet()
    id_b = en_vocab.encode("b")[0]
    model = _FakeSeq2Seq([id_b], vocab_size=len(en_vocab))  # une seule sortie stable : "b"

    n_exemples, batch_size = 5, 2  # 3 batches : tailles 2, 2, 1
    loader = _loader_source_jouet(n_exemples=n_exemples, batch_size=batch_size, tgt_len=2)  # 1 pas généré

    appels: list[tuple[int, list[str]]] = []
    hypotheses = generate_translations(
        model, loader, en_vocab, torch.device("cpu"), batch_callback=lambda i, h: appels.append((i, h))
    )

    assert [i for i, _ in appels] == [0, 2, 4]
    assert [len(h) for _, h in appels] == [2, 2, 1]
    assert [h for _, batch_hyps in appels for h in batch_hyps] == hypotheses
    assert hypotheses == ["b"] * n_exemples


# ---------------------------------------------------------------------------
# SacreBLEU
# ---------------------------------------------------------------------------
def test_corpus_sacrebleu_score_maximal_quand_hypotheses_egalent_references() -> None:
    references = [
        "the cat sat on the mat",
        "she sells sea shells",
        "let 's try something else",
    ]
    score = corpus_sacrebleu(list(references), references)
    assert score > 99.9


def test_corpus_sacrebleu_score_strictement_plus_bas_si_hypotheses_differentes() -> None:
    references = [
        "the cat sat on the mat",
        "she sells sea shells",
        "let 's try something else",
    ]
    hypotheses = [
        "a dog slept under a table",
        "he buys sea shells today",
        "we could not do it",
    ]
    score_egal = corpus_sacrebleu(list(references), references)
    score_different = corpus_sacrebleu(hypotheses, references)
    assert score_different < score_egal


# ---------------------------------------------------------------------------
# generate_translations_resumable (étape 10, reprise) : modèle factice dont
# l'hypothèse dépend du CONTENU de `src` (pas juste de l'ordre d'appel comme
# `_FakeSeq2Seq` plus haut) -- indispensable pour vérifier que la reprise
# (sous-échantillonnage via `Subset`) associe bien chaque hypothèse à la
# BONNE phrase source, dans le BON ordre.
# ---------------------------------------------------------------------------
class _EchoEncoder(nn.Module):
    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, src_len = src.shape
        hidden = src[:, 0].unsqueeze(0).unsqueeze(-1).float()  # (1, batch, 1) -- porte l'id à renvoyer
        encoder_outputs = torch.zeros(batch_size, src_len, 1)
        return encoder_outputs, hidden


class _EchoDecoder(nn.Module):
    """Émet, en UN seul pas, le token dont l'id est porté par `hidden`
    (cf. `_EchoEncoder`) -- déterministe et dépendant de `src`."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_token.shape[0]
        token_ids = hidden[0, :, 0].round().long().clamp(0, self.vocab_size - 1)
        prediction = torch.full((batch_size, self.vocab_size), -10.0)
        prediction[torch.arange(batch_size), token_ids] = 10.0
        return prediction, hidden


class _EchoSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.encoder = _EchoEncoder()
        self.decoder = _EchoDecoder(vocab_size)


class _ExplodingSeq2Seq(nn.Module):
    """Modèle qui plante dès que l'encodeur est appelé -- preuve qu'une
    génération déjà complète ne relance AUCUN calcul."""

    def encoder(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("le modèle n'aurait jamais dû être appelé (génération déjà complète)")


def _echo_dataset_et_vocab(n_exemples: int) -> tuple[TensorDataset, WordVocabulary, list[str]]:
    """Dataset jouet où l'hypothèse attendue pour l'exemple `i` est le mot
    `lettres[i]` (cycle a/b/c/d), et `en_vocab` construit sur ces 4 mots."""
    en_vocab = _vocabulaire_jouet()
    ids = {mot: en_vocab.encode(mot)[0] for mot in "abcd"}
    lettres = [("a", "b", "c", "d")[i % 4] for i in range(n_exemples)]
    src = torch.tensor([[ids[lettre], 0, 0] for lettre in lettres], dtype=torch.long)
    tgt = torch.zeros(n_exemples, 2, dtype=torch.long)  # forme seulement -- max_len passé explicitement
    return TensorDataset(src, tgt), en_vocab, lettres


def test_generate_translations_resumable_ecrit_toutes_les_hypotheses_dans_l_ordre(tmp_path) -> None:
    dataset, en_vocab, lettres = _echo_dataset_et_vocab(n_exemples=12)
    model = _EchoSeq2Seq(vocab_size=len(en_vocab))
    hyp_path = tmp_path / "hypotheses.jsonl"

    hypotheses = generate_translations_resumable(
        model, dataset, en_vocab, torch.device("cpu"), hyp_path, batch_size=4, max_len=2
    )

    assert hypotheses == lettres
    lignes = hyp_path.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == len(lettres)
    for i, ligne in enumerate(lignes):
        assert json.loads(ligne) == {"index": i, "hypothesis": lettres[i]}


def test_generate_translations_resumable_reprend_sans_regenerer_le_deja_fait(tmp_path) -> None:
    dataset, en_vocab, lettres = _echo_dataset_et_vocab(n_exemples=12)
    n = len(lettres)
    hyp_path = tmp_path / "hypotheses.jsonl"

    # Simule une génération INTERROMPUE : les 5 premières lignes existent déjà,
    # avec un contenu volontairement DIFFÉRENT de ce que le modèle produirait --
    # si elles étaient régénérées, ce contenu disparaîtrait.
    n_deja = 5
    placeholders = [f"PLACEHOLDER_{i}" for i in range(n_deja)]
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"index": i, "hypothesis": texte}) + "\n" for i, texte in enumerate(placeholders))

    model = _EchoSeq2Seq(vocab_size=len(en_vocab))
    hypotheses = generate_translations_resumable(
        model, dataset, en_vocab, torch.device("cpu"), hyp_path, batch_size=4, max_len=2
    )

    assert hypotheses[:n_deja] == placeholders  # PAS régénérées
    assert hypotheses[n_deja:] == lettres[n_deja:]  # le reste généré normalement
    assert len(hypotheses) == n

    lignes = hyp_path.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == n
    for i, ligne in enumerate(lignes):
        assert json.loads(ligne)["index"] == i  # ordre et longueur du fichier corrects


def test_generate_translations_resumable_purge_une_ligne_corrompue_avant_de_reprendre(tmp_path) -> None:
    dataset, en_vocab, lettres = _echo_dataset_et_vocab(n_exemples=8)
    n = len(lettres)
    hyp_path = tmp_path / "hypotheses.jsonl"

    n_valides = 3
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"index": i, "hypothesis": lettres[i]}) + "\n" for i in range(n_valides))
        f.write('{"index": 3, "hypothes')  # ligne tronquée, sans retour à la ligne -- écriture interrompue

    model = _EchoSeq2Seq(vocab_size=len(en_vocab))
    hypotheses = generate_translations_resumable(
        model, dataset, en_vocab, torch.device("cpu"), hyp_path, batch_size=4, max_len=2
    )

    assert hypotheses == lettres  # la queue corrompue est purgée puis régénérée
    lignes = hyp_path.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == n
    for i, ligne in enumerate(lignes):
        enregistrement = json.loads(ligne)
        assert enregistrement == {"index": i, "hypothesis": lettres[i]}


def test_generate_translations_resumable_deja_complet_ne_relance_aucun_calcul(tmp_path) -> None:
    dataset, en_vocab, lettres = _echo_dataset_et_vocab(n_exemples=6)
    hyp_path = tmp_path / "hypotheses.jsonl"
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"index": i, "hypothesis": lettre}) + "\n" for i, lettre in enumerate(lettres))

    model = _ExplodingSeq2Seq()  # planterait si le moindre calcul était relancé
    hypotheses = generate_translations_resumable(
        model, dataset, en_vocab, torch.device("cpu"), hyp_path, batch_size=4, max_len=2
    )

    assert hypotheses == lettres


# ---------------------------------------------------------------------------
# METEOR (corpus_meteor) -- monkeypatché, jamais de réseau dans les tests.
# ---------------------------------------------------------------------------
def test_corpus_meteor_renvoie_none_si_wordnet_indisponible(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "_ensure_wordnet", lambda: False)
    assert corpus_meteor(["a b"], ["a b"]) is None


def test_corpus_meteor_moyenne_les_scores_par_phrase(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "_ensure_wordnet", lambda: True)
    scores_successifs = iter([0.2, 0.6])
    monkeypatch.setattr(metrics_module, "_meteor_score_fn", lambda refs, hyp: next(scores_successifs))

    score = corpus_meteor(["a b", "c d"], ["a b", "c d"])

    assert score == pytest.approx(0.4)  # moyenne de 0.2 et 0.6


def test_corpus_meteor_liste_vide_renvoie_none() -> None:
    assert corpus_meteor([], []) is None


# ---------------------------------------------------------------------------
# BERTScore (corpus_bertscore) -- monkeypatché, jamais de téléchargement de
# modèle dans les tests.
# ---------------------------------------------------------------------------
def test_corpus_bertscore_renvoie_none_si_modele_indisponible(monkeypatch) -> None:
    def _bert_score_qui_plante(*args, **kwargs):
        raise OSError("pas de réseau")

    monkeypatch.setattr(metrics_module.bert_score, "score", _bert_score_qui_plante)
    assert corpus_bertscore(["a"], ["a"]) is None


def test_corpus_bertscore_calcule_les_moyennes(monkeypatch) -> None:
    def _faux_bert_score(cands, refs, **kwargs):
        return (
            torch.tensor([0.8, 0.6]),
            torch.tensor([0.7, 0.5]),
            torch.tensor([0.75, 0.55]),
        )

    monkeypatch.setattr(metrics_module.bert_score, "score", _faux_bert_score)
    resultat = corpus_bertscore(["a", "b"], ["a", "b"])

    assert resultat == pytest.approx({"precision": 0.7, "recall": 0.6, "f1": 0.65})


def test_corpus_bertscore_liste_vide_renvoie_none() -> None:
    assert corpus_bertscore([], []) is None


# ---------------------------------------------------------------------------
# evaluate_by_length_bin
# ---------------------------------------------------------------------------
def test_evaluate_by_length_bin_ventile_correctement(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "_ensure_wordnet", lambda: True)
    monkeypatch.setattr(metrics_module, "_meteor_score_fn", lambda refs, hyp: 1.0)

    reference_courte = "the cat sat on the mat"  # 6 mots -- assez long pour un BLEU (4-grammes) non nul
    reference_longue = " ".join(f"w{i}" for i in range(20))  # 20 mots
    references = [reference_courte, reference_longue]
    hypotheses = list(references)  # score parfait des deux côtés
    sources = ["src1", "src2"]

    df = evaluate_by_length_bin(hypotheses, references, sources, threshold=18, compute_bertscore=False)

    assert set(df["bin"]) == {"courte", "longue"}
    ligne_courte = df.loc[df["bin"] == "courte"].iloc[0]
    ligne_longue = df.loc[df["bin"] == "longue"].iloc[0]
    assert ligne_courte["n"] == 1 and ligne_longue["n"] == 1
    assert ligne_courte["sacrebleu"] > 99.9
    assert ligne_courte["meteor"] == pytest.approx(1.0)
    assert math.isnan(ligne_courte["bertscorePrecision"])  # compute_bertscore=False


def test_evaluate_by_length_bin_bin_vide_donne_des_nan(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "_ensure_wordnet", lambda: True)
    monkeypatch.setattr(metrics_module, "_meteor_score_fn", lambda refs, hyp: 1.0)

    references = ["a b c"]  # une seule phrase courte -> bin "longue" vide
    hypotheses = ["a b c"]
    sources = ["src"]

    df = evaluate_by_length_bin(hypotheses, references, sources, threshold=18, compute_bertscore=False)

    ligne_longue = df.loc[df["bin"] == "longue"].iloc[0]
    assert ligne_longue["n"] == 0
    assert math.isnan(ligne_longue["sacrebleu"])
    assert math.isnan(ligne_longue["meteor"])
    assert math.isnan(ligne_longue["bertscoreF1"])


def test_evaluate_by_length_bin_appelle_bertscore_quand_demande(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "_ensure_wordnet", lambda: True)
    monkeypatch.setattr(metrics_module, "_meteor_score_fn", lambda refs, hyp: 1.0)
    monkeypatch.setattr(
        metrics_module,
        "corpus_bertscore",
        lambda hyps, refs, model_type=None: {"precision": 0.9, "recall": 0.8, "f1": 0.85},
    )

    references = ["a b c"]
    hypotheses = ["a b c"]
    sources = ["src"]

    df = evaluate_by_length_bin(hypotheses, references, sources, threshold=18, compute_bertscore=True)

    ligne_courte = df.loc[df["bin"] == "courte"].iloc[0]
    assert ligne_courte["bertscorePrecision"] == pytest.approx(0.9)
    assert ligne_courte["bertscoreF1"] == pytest.approx(0.85)


def test_evaluate_by_length_bin_erreur_si_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        evaluate_by_length_bin(["h1", "h2"], ["r1"], ["s1", "s2"])
