"""Tests de l'étape 6 (`src/models/seq2seq.py`) : encodeur-décodeur RNN|GRU,
attention additive de Bahdanau, teacher forcing.

Mini-modèles (dimensions minuscules, 1-2 epochs ailleurs dans
`test_training.py`) pour garder le gate rapide.
"""

from __future__ import annotations

import torch

from src.models.seq2seq import Encoder, build_model

FR_VOCAB_SIZE = 12
EN_VOCAB_SIZE = 15
EMB_DIM = 4
HIDDEN_DIM = 6
BATCH_SIZE = 2
SRC_LEN = 5
TGT_LEN = 7


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
