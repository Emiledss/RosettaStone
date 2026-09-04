"""Boucle d'entraînement (étape 7) : loss ignorant le padding, Adam, gradient
clipping, suivi train/val loss + perplexité, early stopping, callback de fin
d'epoch (point d'accroche du pruning Optuna).

Contrat imposé par `notebooks/Rosetta_Optuna.ipynb` (`makeObjective` /
`runStudy`) :

- `train_and_validate(model, loaders, max_epochs, lr, grad_clip,
  teacher_forcing_ratio, patience, device, on_epoch_end=None, pad_id=0,
  checkpoint_path=None)` -- appelée positionnellement jusqu'à `device`, puis
  `on_epoch_end` en mot-clé.
- Renvoie un `dict` contenant au minimum les clés `bestValLoss` et `history`
  (`makeObjective` lit `result["bestValLoss"]`).
- `on_epoch_end(epoch, val_loss)` est appelé à chaque fin d'epoch ; s'il
  renvoie `True`, l'entraînement s'arrête. Il peut aussi lever une exception
  (pruning Optuna) -- ne JAMAIS l'intercepter ici.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_and_validate(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    max_epochs: int,
    lr: float,
    grad_clip: float,
    teacher_forcing_ratio: float,
    patience: int,
    device: torch.device,
    on_epoch_end: Callable[[int, float], bool] | None = None,
    pad_id: int = 0,
    checkpoint_path: str | None = None,
) -> dict:
    """Entraîne `model` et valide à chaque epoch, avec early stopping.

    - Loss : `CrossEntropyLoss(ignore_index=pad_id)` -- le padding ne compte pas.
    - Optimiseur : Adam(`lr`).
    - Gradient clipping (`clip_grad_norm_(model.parameters(), grad_clip)`).
    - Validation en génération libre (`teacher_forcing_ratio=0.0`) -- pas de
      triche : la val loss doit refléter l'inférence, pas l'entraînement.
    - Early stopping sur `patience` epochs sans amélioration de la val loss.
    - `checkpoint_path` : si fourni, sauvegarde le meilleur modèle (val loss)
      à ce chemin ; si `None` (cas Optuna, un essai par trial), aucun fichier
      n'est écrit -- écrire un checkpoint par essai serait du gaspillage.

    Renvoie `{"history": {...}, "bestValLoss": float}` -- `history` contient
    les listes `trainLoss`, `valLoss`, `valPerplexity`, une entrée par epoch
    effectivement réalisée.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"trainLoss": [], "valLoss": [], "valPerplexity": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, n_train_batches = 0.0, 0
        for src, tgt in loaders["train"]:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:, :].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_sum += loss.item()
            n_train_batches += 1
        train_loss = train_loss_sum / max(n_train_batches, 1)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for src, tgt in loaders["val"]:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, 0.0)  # pas de triche en validation
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:, :].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
                val_loss_sum += loss.item()
                n_val_batches += 1
        val_loss = val_loss_sum / max(n_val_batches, 1)
        val_perplexity = math.exp(min(val_loss, 20)) if math.isfinite(val_loss) else float("inf")

        history["trainLoss"].append(train_loss)
        history["valLoss"].append(val_loss)
        history["valPerplexity"].append(val_perplexity)

        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Le callback peut lever une exception (pruning Optuna) -- ne pas l'intercepter.
        should_stop = bool(on_epoch_end(epoch, val_loss)) if on_epoch_end is not None else False
        if should_stop or epochs_without_improvement >= patience:
            break

    return {"history": history, "bestValLoss": best_val_loss}
