"""Boucle d'entraînement (étape 7) : loss ignorant le padding, Adam, gradient
clipping, suivi train/val loss + perplexité, early stopping, callback de fin
d'epoch (point d'accroche du pruning Optuna), scheduled sampling et objectif
optionnel piloté par une métrique externe (SacreBLEU).

Contrat imposé par `notebooks/Rosetta_Modelisation.ipynb` (`makeObjective` /
`runStudy`) :

- `train_and_validate(model, loaders, max_epochs, lr, grad_clip,
  teacher_forcing_ratio, patience, device, on_epoch_end=None, pad_id=0,
  checkpoint_path=None, teacher_forcing_decay=1.0, objective_fn=None,
  direction="minimize")` -- les 8 premiers arguments (jusqu'à `device`) sont
  appelés positionnellement, tous les suivants restent des mots-clés à défaut
  (rétrocompatible avec les appels existants).
- Renvoie un `dict` contenant au minimum les clés `bestValLoss`,
  `bestObjective` et `history`. `bestValLoss` reste la meilleure val loss
  observée, quels que soient `objective_fn`/`direction`. `bestObjective` est
  la meilleure valeur de la métrique qui pilote réellement les décisions
  (checkpoint, early stopping, `on_epoch_end`) : la val loss par défaut
  (`objective_fn=None`, `direction="minimize"` -- alors `bestObjective ==
  bestValLoss`), ou `objective_fn(model)` sinon (typiquement le SacreBLEU en
  génération libre sur un sous-échantillon de val, `direction="maximize"`).
- `on_epoch_end(epoch, metric_value)` est appelé à chaque fin d'epoch avec
  cette même métrique (val loss par défaut, `objective_fn(model)` sinon) ; s'il
  renvoie `True`, l'entraînement s'arrête. Il peut aussi lever une exception
  (pruning Optuna) -- ne JAMAIS l'intercepter ici.
- `teacher_forcing_decay` : le ratio de teacher forcing décroît
  **exponentiellement** depuis `teacher_forcing_ratio` (epoch 1), selon
  `ratio(e) = teacher_forcing_ratio * teacher_forcing_decay ** (e - 1)`.
  `teacher_forcing_decay == 1.0` désactive le scheduled sampling (ratio
  constant, comportement inchangé). La formule ne dépend **pas** de
  `max_epochs` -- voir `_scheduled_teacher_forcing_ratio` pour le pourquoi.
  Le ratio effectif de chaque epoch est enregistré dans
  `history["teacherForcingRatio"]`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def _scheduled_teacher_forcing_ratio(epoch: int, start: float, decay: float) -> float:
    """Décroissance **exponentielle** : `start * decay ** (epoch - 1)`.

    `decay == 1.0` -> ratio constant égal à `start` (scheduled sampling
    désactivé, comportement d'avant ce paramètre).

    La formule ne dépend PAS de `max_epochs`, et c'est le point du
    reparamétrage :

    - **transférabilité** : la valeur retenue par la recherche Optuna
      (8 epochs) signifie exactement la même chose à l'entraînement final
      (~15 epochs). Une décroissance normalisée sur `max_epochs` changeait de
      vitesse d'un régime à l'autre, rendant l'hyperparamètre intransférable ;
    - **observabilité sous pruning** : `MedianPruner` coupe une part
      importante des essais après quelques epochs. Le ratio ayant déjà bougé à
      ce moment-là, un essai élagué tôt porte l'information du paramètre ; avec
      une décroissance normalisée il n'approchait jamais son plancher et Optuna
      scorait un effet qui ne s'était pas produit.

    Le plancher vaut 0 par construction (limite de `decay ** (epoch - 1)`), ce
    qui correspond aux conditions réelles d'inférence -- étant entendu que
    l'inférence n'a jamais eu besoin de teacher forcing : `generate_translations`
    fait de la génération libre pure. Ce paramètre ne règle que la préparation
    du modèle au biais d'exposition.
    """
    return start * (decay ** (epoch - 1))


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
    teacher_forcing_decay: float = 1.0,
    objective_fn: Callable[[nn.Module], float] | None = None,
    direction: str = "minimize",
) -> dict:
    """Entraîne `model` et valide à chaque epoch, avec early stopping.

    - Loss : `CrossEntropyLoss(ignore_index=pad_id)` -- le padding ne compte pas.
    - Optimiseur : Adam(`lr`).
    - Gradient clipping (`clip_grad_norm_(model.parameters(), grad_clip)`).
    - Teacher forcing en entraînement : décroît exponentiellement depuis
      `teacher_forcing_ratio` au taux `teacher_forcing_decay` par epoch,
      indépendamment de `max_epochs` (cf. `_scheduled_teacher_forcing_ratio`).
    - Validation en génération libre (`teacher_forcing_ratio=0.0`) -- pas de
      triche : la val loss doit refléter l'inférence, pas l'entraînement.
    - `objective_fn(model)`, si fourni, est appelé à chaque fin d'epoch et
      REMPLACE la val loss comme métrique pilotant `on_epoch_end`, le
      checkpoint et l'early stopping -- typiquement le SacreBLEU calculé en
      génération libre sur un sous-échantillon de validation
      (`direction="maximize"` dans ce cas). La val loss et la perplexité
      restent calculées et enregistrées dans `history` dans tous les cas : on
      ne perd pas l'information, on change seulement ce qui pilote la décision.
    - `direction` : `"minimize"` (val loss, par défaut) ou `"maximize"` (ex.
      SacreBLEU) -- gouverne le sens de « amélioration » pour le checkpoint et
      l'early stopping (pas de martingale implicite sur le signe de la métrique).
    - Early stopping sur `patience` epochs sans amélioration (au sens de
      `direction`) de la métrique pilotée (val loss, ou `objective_fn` si fourni).
    - `checkpoint_path` : si fourni, sauvegarde le meilleur modèle (au sens de
      `direction`) à ce chemin ; si `None` (cas Optuna, un essai par trial),
      aucun fichier n'est écrit -- écrire un checkpoint par essai serait du
      gaspillage.

    Renvoie `{"history": {...}, "bestValLoss": float, "bestObjective": float}`
    -- `history` contient les listes `trainLoss`, `valLoss`, `valPerplexity`,
    `teacherForcingRatio`, une entrée par epoch effectivement réalisée.
    `bestValLoss` reste la meilleure val loss observée (indépendamment de
    `objective_fn`) ; `bestObjective` est la meilleure valeur de la métrique
    qui a réellement piloté les décisions (identique à `bestValLoss` quand
    `objective_fn=None`).
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError(f"direction doit être 'minimize' ou 'maximize', reçu {direction!r}")

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "trainLoss": [],
        "valLoss": [],
        "valPerplexity": [],
        "teacherForcingRatio": [],
    }
    best_val_loss = float("inf")
    best_metric = float("inf") if direction == "minimize" else float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        epoch_teacher_forcing_ratio = _scheduled_teacher_forcing_ratio(
            epoch, teacher_forcing_ratio, teacher_forcing_decay
        )

        model.train()
        train_loss_sum, n_train_batches = 0.0, 0
        for src, tgt in loaders["train"]:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt, epoch_teacher_forcing_ratio)
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
        history["teacherForcingRatio"].append(epoch_teacher_forcing_ratio)

        best_val_loss = min(best_val_loss, val_loss)  # toujours suivie, indépendamment de objective_fn/direction

        metric_value = objective_fn(model) if objective_fn is not None else val_loss
        if direction == "minimize":
            improved = metric_value < best_metric - 1e-4
        else:
            improved = metric_value > best_metric + 1e-4

        if improved:
            best_metric = metric_value
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Le callback peut lever une exception (pruning Optuna) -- ne pas l'intercepter.
        should_stop = bool(on_epoch_end(epoch, metric_value)) if on_epoch_end is not None else False
        if should_stop or epochs_without_improvement >= patience:
            break

    return {"history": history, "bestValLoss": best_val_loss, "bestObjective": best_metric}
