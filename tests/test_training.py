"""Tests de l'étape 7 (`src/training/loop.py`) : loss ignorant le padding,
gradient clipping, callback de fin d'epoch, early stopping.

Mini-jeu et mini-modèle (2 vocabulaires de 8 tokens, quelques epochs) pour
garder le gate rapide.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.seq2seq import build_model
from src.training.loop import train_and_validate

VOCAB_SIZE = 8
SEQ_LEN = 6
N_EXEMPLES = 16
BATCH_SIZE = 4


def _mini_loaders() -> dict[str, DataLoader]:
    torch.manual_seed(0)
    src = torch.randint(1, VOCAB_SIZE, (N_EXEMPLES, SEQ_LEN))
    tgt = torch.randint(1, VOCAB_SIZE, (N_EXEMPLES, SEQ_LEN))
    train_ds = TensorDataset(src, tgt)
    val_ds = TensorDataset(src[:8], tgt[:8])
    return {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE),
    }


def _mini_model(cell_type: str = "gru") -> nn.Module:
    return build_model(VOCAB_SIZE, VOCAB_SIZE, emb_dim=8, hidden_dim=16, cell_type=cell_type, dropout=0.0)


def test_train_and_validate_loss_descend_et_best_val_loss_finie() -> None:
    """La train loss descend sur quelques epochs (pas de bug de pipeline), et
    `bestValLoss` est finie."""
    torch.manual_seed(0)
    loaders = _mini_loaders()
    model = _mini_model()

    result = train_and_validate(
        model,
        loaders,
        max_epochs=8,
        lr=0.01,
        grad_clip=1.0,
        teacher_forcing_ratio=0.5,
        patience=8,
        device=torch.device("cpu"),
        pad_id=0,
    )

    assert "bestValLoss" in result and "history" in result
    history = result["history"]
    assert len(history["trainLoss"]) == 8
    assert history["trainLoss"][-1] < history["trainLoss"][0]
    assert math.isfinite(result["bestValLoss"])


def test_on_epoch_end_appele_une_fois_par_epoch_et_interrompt_si_true() -> None:
    """`on_epoch_end(epoch, valLoss)` est appelé exactement une fois par epoch
    réalisée ; s'il renvoie `True`, l'entraînement s'arrête aussitôt."""
    torch.manual_seed(0)
    loaders = _mini_loaders()
    model = _mini_model()

    appels: list[tuple[int, float]] = []

    def on_epoch_end(epoch: int, val_loss: float) -> bool:
        appels.append((epoch, val_loss))
        return epoch >= 2  # arrêt volontaire dès la 2e epoch

    result = train_and_validate(
        model,
        loaders,
        max_epochs=8,
        lr=0.01,
        grad_clip=1.0,
        teacher_forcing_ratio=0.5,
        patience=8,
        device=torch.device("cpu"),
        on_epoch_end=on_epoch_end,
        pad_id=0,
    )

    assert appels == [(1, appels[0][1]), (2, appels[1][1])]
    assert len(result["history"]["trainLoss"]) == 2


def test_on_epoch_end_exception_pruning_optuna_non_interceptee() -> None:
    """Une exception levée depuis `on_epoch_end` (pruning Optuna) doit se
    propager -- `train_and_validate` ne doit JAMAIS l'intercepter."""
    torch.manual_seed(0)
    loaders = _mini_loaders()
    model = _mini_model()

    class PruningFactice(Exception):
        pass

    def on_epoch_end(epoch: int, val_loss: float) -> bool:
        raise PruningFactice("essai élagué")

    try:
        train_and_validate(
            model,
            loaders,
            max_epochs=8,
            lr=0.01,
            grad_clip=1.0,
            teacher_forcing_ratio=0.5,
            patience=8,
            device=torch.device("cpu"),
            on_epoch_end=on_epoch_end,
            pad_id=0,
        )
        raise AssertionError("PruningFactice aurait dû se propager")
    except PruningFactice:
        pass


def test_gradient_clipping_evite_l_explosion_sans_nan() -> None:
    """Avec un `grad_clip` faible, l'entraînement reste stable (pas de NaN),
    conformément au point de contrôle de l'étape 7."""
    torch.manual_seed(0)
    loaders = _mini_loaders()
    model = _mini_model(cell_type="rnn")

    result = train_and_validate(
        model,
        loaders,
        max_epochs=5,
        lr=0.05,
        grad_clip=0.5,
        teacher_forcing_ratio=0.5,
        patience=5,
        device=torch.device("cpu"),
        pad_id=0,
    )

    assert math.isfinite(result["bestValLoss"])
    assert all(math.isfinite(v) for v in result["history"]["trainLoss"])
    assert all(math.isfinite(v) for v in result["history"]["valLoss"])


def test_checkpoint_ecrit_seulement_si_checkpoint_path_fourni(tmp_path) -> None:
    """Aucun fichier écrit si `checkpoint_path=None` (cas Optuna) ; un fichier
    écrit si `checkpoint_path` est fourni."""
    torch.manual_seed(0)
    loaders = _mini_loaders()

    model_sans_checkpoint = _mini_model()
    train_and_validate(
        model_sans_checkpoint,
        loaders,
        max_epochs=2,
        lr=0.01,
        grad_clip=1.0,
        teacher_forcing_ratio=0.5,
        patience=2,
        device=torch.device("cpu"),
        pad_id=0,
        checkpoint_path=None,
    )
    assert list(tmp_path.iterdir()) == []

    chemin = tmp_path / "best_model.pt"
    model_avec_checkpoint = _mini_model()
    train_and_validate(
        model_avec_checkpoint,
        loaders,
        max_epochs=2,
        lr=0.01,
        grad_clip=1.0,
        teacher_forcing_ratio=0.5,
        patience=2,
        device=torch.device("cpu"),
        pad_id=0,
        checkpoint_path=str(chemin),
    )
    assert chemin.exists()


def test_loss_ignore_le_padding() -> None:
    """`CrossEntropyLoss(ignore_index=pad_id)` (mécanisme utilisé par
    `train_and_validate`) : une cible entièrement `<pad>` ne contribue pas --
    la loss ne dépend alors plus des logits, même absurdes, sur ces positions."""
    pad_id = 0
    vocab_size = 6
    torch.manual_seed(0)

    logits_reels = torch.randn(2, 3, vocab_size)
    cibles_reelles = torch.randint(1, vocab_size, (2, 3))
    logits_padding_absurdes = torch.randn(2, 4, vocab_size) * 1000.0
    cibles_padding = torch.zeros(2, 4, dtype=torch.long)  # entièrement <pad>

    logits_complets = torch.cat([logits_reels, logits_padding_absurdes], dim=1)
    cibles_completes = torch.cat([cibles_reelles, cibles_padding], dim=1)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss_avec_padding = criterion(logits_complets.reshape(-1, vocab_size), cibles_completes.reshape(-1))
    loss_sans_padding = criterion(logits_reels.reshape(-1, vocab_size), cibles_reelles.reshape(-1))

    assert torch.allclose(loss_avec_padding, loss_sans_padding, atol=1e-5)


