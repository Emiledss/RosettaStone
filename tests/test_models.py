"""Tests de l'étape 6 (`src/models/seq2seq.py`) : encodeur-décodeur RNN|GRU,
attention additive de Bahdanau, teacher forcing.

Mini-modèles (dimensions minuscules, 1-2 epochs ailleurs dans
`test_training.py`) pour garder le gate rapide.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.evaluation.generate import generate_translations
from src.evaluation.metrics import corpus_sacrebleu
from src.models.seq2seq import BahdanauAttention, Encoder, build_model
from src.tokenization.numerize import make_dataloaders
from src.tokenization.vocab_builder import WordVocabulary
from src.training.loop import train_and_validate

FR_VOCAB_SIZE = 12
EN_VOCAB_SIZE = 15
EMB_DIM = 4
HIDDEN_DIM = 6
BATCH_SIZE = 2
SRC_LEN = 5
TGT_LEN = 7

JOINT_VOCAB_SIZE = 20  # fr_vocab_size == en_vocab_size, requis par tie_embeddings=True


def _batch_jouet() -> tuple[torch.Tensor, torch.Tensor]:
    """Batch jouet d'indices entiers (`<pad>`=0 possible, comme le vrai pipeline)."""
    torch.manual_seed(0)
    src = torch.randint(0, FR_VOCAB_SIZE, (BATCH_SIZE, SRC_LEN))
    tgt = torch.randint(0, EN_VOCAB_SIZE, (BATCH_SIZE, TGT_LEN))
    return src, tgt


def test_encoder_renvoie_tous_les_etats_caches() -> None:
    """L'encodeur renvoie TOUS les états cachés (dimension temporelle =
    src_len), pas seulement le dernier -- prérequis pour greffer l'attention."""
    src, _tgt = _batch_jouet()
    encoder = Encoder(FR_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, cell_type="gru")

    outputs, hidden = encoder(src)

    assert outputs.shape == (BATCH_SIZE, SRC_LEN, HIDDEN_DIM)
    assert hidden.shape == (1, BATCH_SIZE, HIDDEN_DIM)


def test_forward_rnn_et_gru_forme_de_sortie_exacte() -> None:
    """Bascule RNN<->GRU = 1 paramètre (`cell_type`) ; `forward` renvoie
    `[batch, seq_len, vocab_size]`, batch-first, sans erreur de dimension."""
    src, tgt = _batch_jouet()
    for cell_type in ("rnn", "gru"):
        model = build_model(
            FR_VOCAB_SIZE, EN_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, cell_type, dropout=0.0
        )
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        assert output.shape == (BATCH_SIZE, TGT_LEN, EN_VOCAB_SIZE)


def test_attention_on_off_passent_et_produisent_des_sorties_differentes() -> None:
    """Bascule attention on/off = 1 flag (`use_attention`) ; les deux passent
    sans erreur de dimension et produisent des sorties différentes."""
    src, tgt = _batch_jouet()

    torch.manual_seed(1)
    model_sans_attention = build_model(
        FR_VOCAB_SIZE, EN_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, "gru", dropout=0.0, use_attention=False
    )
    torch.manual_seed(1)
    model_avec_attention = build_model(
        FR_VOCAB_SIZE, EN_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, "gru", dropout=0.0, use_attention=True
    )

    output_sans = model_sans_attention(src, tgt, teacher_forcing_ratio=1.0)
    output_avec = model_avec_attention(src, tgt, teacher_forcing_ratio=1.0)

    assert output_sans.shape == output_avec.shape == (BATCH_SIZE, TGT_LEN, EN_VOCAB_SIZE)
    assert not torch.allclose(output_sans, output_avec)


def test_teacher_forcing_ratio_0_et_1_comportements_distincts() -> None:
    """`teacher_forcing_ratio=1.0` (toujours le vrai token) et `=0.0` (toujours
    la prédiction du modèle) donnent des sorties vérifiablement distinctes, à
    poids et entrées fixés."""
    src, tgt = _batch_jouet()
    torch.manual_seed(2)
    model = build_model(FR_VOCAB_SIZE, EN_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, "rnn", dropout=0.0)
    model.eval()  # pas de dropout : seule la politique de teacher forcing doit varier

    with torch.no_grad():
        output_teacher_forcing_pur = model(src, tgt, teacher_forcing_ratio=1.0)
        output_generation_libre = model(src, tgt, teacher_forcing_ratio=0.0)

    # la position 0 (<sos> en entrée, jamais prédite) reste à zéro dans les deux cas
    assert torch.all(output_teacher_forcing_pur[:, 0, :] == 0)
    assert torch.all(output_generation_libre[:, 0, :] == 0)
    # au-delà de la position 0, les deux politiques divergent (nourrissage différent)
    assert not torch.allclose(
        output_teacher_forcing_pur[:, 1:, :], output_generation_libre[:, 1:, :]
    )


# ---------------------------------------------------------------------------
# tie_embeddings : garde-fou, piège de dimensions, comptage de paramètres.
# ---------------------------------------------------------------------------
def test_tie_embeddings_avec_vocabulaires_differents_leve_value_error() -> None:
    """Lier n'a de sens que si les deux vocabulaires sont identiques -- garde-fou
    obligatoire pour les configs à vocabulaires séparés (full, words95)."""
    with pytest.raises(ValueError):
        build_model(
            FR_VOCAB_SIZE, EN_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, "gru", dropout=0.0, tie_embeddings=True
        )


def test_tie_embeddings_avec_emb_dim_different_de_hidden_dim_fonctionne() -> None:
    """Piège de dimensions : Optuna tire embDim et hiddenDim indépendamment,
    ils diffèrent la plupart du temps -- la projection hidden_dim->emb_dim
    doit rendre ce cas fonctionnel (pas de plantage, pas de flag ignoré)."""
    emb_dim, hidden_dim = 4, 6
    assert emb_dim != hidden_dim
    src, tgt = _batch_jouet()
    src = src % JOINT_VOCAB_SIZE
    tgt = tgt % JOINT_VOCAB_SIZE

    model = build_model(
        JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE, emb_dim, hidden_dim, "gru", dropout=0.0, tie_embeddings=True
    )
    output = model(src, tgt, teacher_forcing_ratio=0.5)
    assert output.shape == (BATCH_SIZE, TGT_LEN, JOINT_VOCAB_SIZE)


def test_tie_embeddings_avec_emb_dim_egal_hidden_dim_fonctionne() -> None:
    """Cas où emb_dim == hidden_dim : la projection intermédiaire devient une
    identité, la liaison directe (au sens classique) reste opérante."""
    dim = 6
    src, tgt = _batch_jouet()
    src = src % JOINT_VOCAB_SIZE
    tgt = tgt % JOINT_VOCAB_SIZE

    model = build_model(
        JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE, dim, dim, "rnn", dropout=0.0, tie_embeddings=True
    )
    output = model(src, tgt, teacher_forcing_ratio=0.5)
    assert output.shape == (BATCH_SIZE, TGT_LEN, JOINT_VOCAB_SIZE)


def test_modele_lie_a_strictement_moins_de_parametres_que_non_lie() -> None:
    """C'est le garde-fou de fond : lier n'a d'intérêt que si ça réduit
    effectivement le nombre de paramètres, à hyperparamètres égaux."""
    emb_dim, hidden_dim = 4, 6  # dimensions différentes, cas piégeux inclus

    model_lie = build_model(
        JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE, emb_dim, hidden_dim, "gru", dropout=0.0, tie_embeddings=True
    )
    model_non_lie = build_model(
        JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE, emb_dim, hidden_dim, "gru", dropout=0.0, tie_embeddings=False
    )

    n_parametres_lie = sum(p.numel() for p in model_lie.parameters())
    n_parametres_non_lie = sum(p.numel() for p in model_non_lie.parameters())
    assert n_parametres_lie < n_parametres_non_lie


# ---------------------------------------------------------------------------
# Lot O : encodeur insensible au padding, attention masquée. Avant correctif,
# `Encoder.forward` renvoyait l'état après les pas de `<pad>` -- le contexte
# était effacé sur des batchs paddés à des longueurs très supérieures à la
# phrase réelle (RNN sans portes en particulier, incapable d'y résister).
# ---------------------------------------------------------------------------
PAD_TOY_VOCAB_SIZE = 30


@pytest.mark.parametrize("cell_type", ["rnn", "gru"])
def test_encoder_invariant_au_padding(cell_type: str) -> None:
    """État final de l'encodeur identique pour une phrase de 8 tokens et la
    même phrase suivie de 20 `<pad>` -- et `encoder_outputs` identiques sur
    les 8 premières positions dans les deux cas (`pack_padded_sequence`/
    `pad_packed_sequence`, lot O)."""
    torch.manual_seed(0)
    emb_dim, hidden_dim = 8, 16
    encoder = Encoder(PAD_TOY_VOCAB_SIZE, emb_dim, hidden_dim, cell_type=cell_type)
    encoder.eval()

    src = torch.randint(1, PAD_TOY_VOCAB_SIZE, (1, 8))  # jamais 0 (<pad>) dans le contenu réel
    src_avec_pad = torch.cat([src, torch.zeros(1, 20, dtype=torch.long)], dim=1)

    with torch.no_grad():
        outputs, hidden = encoder(src)
        outputs_avec_pad, hidden_avec_pad = encoder(src_avec_pad)

    assert torch.allclose(hidden, hidden_avec_pad, atol=1e-5)
    assert torch.allclose(outputs, outputs_avec_pad[:, :8], atol=1e-5)


def test_attention_masque_les_positions_de_padding() -> None:
    """Avec `src_mask`, les poids d'attention sur les positions de padding
    valent exactement 0 et la somme sur les vraies positions vaut 1."""
    torch.manual_seed(0)
    hidden_dim = 6
    attention = BahdanauAttention(hidden_dim)
    batch_size, src_len = 2, 5
    decoder_hidden = torch.randn(batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
    src_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )

    _context, weights = attention(decoder_hidden, encoder_outputs, src_mask)

    assert torch.all(weights[~src_mask] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-6)


def _corpus_jouet_surapprentissage() -> pd.DataFrame:
    """64 paires FR->EN synthétiques (4 sujets x 4 verbes x 4 objets), assez
    petites et régulières pour qu'un modèle de 128 unités cachées les
    mémorise en quelques dizaines d'epochs -- pas le vrai corpus Rosetta."""
    sujets = [("le chat", "the cat"), ("le chien", "the dog"), ("la souris", "the mouse"), ("le lion", "the lion")]
    verbes = [("mange", "eats"), ("regarde", "watches"), ("aime", "likes"), ("chasse", "chases")]
    objets = [("la pomme", "the apple"), ("le pain", "the bread"), ("la balle", "the ball"), ("la lune", "the moon")]

    paires = [
        (f"{sujet_fr} {verbe_fr} {objet_fr}", f"{sujet_en} {verbe_en} {objet_en}")
        for sujet_fr, sujet_en in sujets
        for verbe_fr, verbe_en in verbes
        for objet_fr, objet_en in objets
    ]
    return pd.DataFrame(paires, columns=["fr", "en"])


def test_rnn_surapprend_le_corpus_jouet_apres_correctif_encodeur(tmp_path) -> None:
    """Le test qui manquait avant le lot O : sur un corpus jouet minuscule,
    teacher forcing pur, le RNN doit apprendre à générer des traductions
    VARIÉES (pas une sortie constante -- le symptôme du bug de padding) et
    obtenir un BLEU correct en génération libre. Avant correctif, ce test
    échouait (BLEU 1,0, sortie constante) : l'encodeur perdait le contexte
    sur les positions paddées."""
    torch.manual_seed(0)
    df = _corpus_jouet_surapprentissage()

    fr_vocab = WordVocabulary.fit(df["fr"], lang="fr", config_name="toy-fr", coverage=1.0)
    en_vocab = WordVocabulary.fit(df["en"], lang="en", config_name="toy-en", coverage=1.0)

    # batch_size=8 (au lieu du batch unique de 64) -- plus de pas de gradient par
    # epoch, nécessaire pour mémoriser 64 phrases sans attention en ~2 minutes.
    loaders = make_dataloaders(
        {"train": df, "val": df},
        fr_vocab,
        en_vocab,
        batch_size=8,
        bucket_by_length=False,
    )

    model = build_model(
        len(fr_vocab), len(en_vocab), emb_dim=32, hidden_dim=128, cell_type="rnn", dropout=0.0
    )

    train_and_validate(
        model,
        loaders,
        max_epochs=120,
        lr=1e-3,
        grad_clip=1.0,
        teacher_forcing_ratio=1.0,
        patience=120,  # pas d'arrêt anticipé sur ce petit jeu jouet
        device=torch.device("cpu"),
        pad_id=0,
        checkpoint_path=str(tmp_path / "toy_rnn.pt"),
    )

    references = list(df["en"])
    hypotheses = generate_translations(model, loaders["val"], en_vocab, torch.device("cpu"))

    # fini la sortie constante : au moins la moitié des hypothèses diffèrent entre elles
    assert len(set(hypotheses)) >= len(hypotheses) // 2

    bleu = corpus_sacrebleu(hypotheses, references)
    assert bleu > 20
