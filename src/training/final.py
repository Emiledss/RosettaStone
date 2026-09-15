"""Entraînement final REPRENABLE (étape 9), pour tourner des heures sur le
corpus complet sans craindre une interruption -- contrairement à
`src.training.loop.train_and_validate` (un essai Optuna, jetable).

Architecture par run, sous `run_dir` (typiquement `reports/runs/<run_id>/`) :
- `state.json` -- statut de chaque étape (`train`, `generate`, `metrics`),
  SEULE source de vérité pour savoir quoi reprendre.
- `checkpoint.pt` -- modèle + optimiseur + historique + état du RNG, écrit à
  CHAQUE fin d'epoch (pas seulement au meilleur) pour permettre la reprise.
- `best.pt` -- SEULEMENT le meilleur modèle (au sens de `direction`).
- `history.json` -- courbes (train loss, val loss, perplexité, teacher forcing).

Écriture ATOMIQUE partout : fichier temporaire `<nom>.tmp` puis `os.replace`
(atomique sur POSIX ET Windows) -- une interruption pendant l'écriture laisse
soit l'ancien fichier intact, soit le nouveau complet, jamais un fichier
tronqué/corrompu.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.loop import _scheduled_teacher_forcing_ratio

DEFAULT_STEPS = ("train", "generate", "metrics")


# ---------------------------------------------------------------------------
# state.json -- source de vérité de la reprise, partagée par les 3 étapes.
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomique (POSIX rename / Windows MoveFileEx)
    except BaseException:
        tmp.unlink(missing_ok=True)  # pas de .tmp résiduel après un échec
        raise


def _atomic_torch_save(path: Path, obj: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # remplace SEULEMENT si l'écriture a réussi -- l'ancien
        # fichier (s'il existe) reste intact et lisible tant que ce point n'est pas atteint
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_state(run_dir: str | Path) -> dict:
    """Charge `state.json` s'il existe, sinon un état initial (toutes les
    étapes `"pending"`). Ne lève jamais pour un `run_dir` neuf."""
    path = Path(run_dir) / "state.json"
    if not path.exists():
        return {"steps": {step: "pending" for step in DEFAULT_STEPS}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(run_dir: str | Path, state: dict) -> None:
    """Écrit `state.json` de façon atomique."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "state.json", state)


def mark_step_done(run_dir: str | Path, step: str, **extra: object) -> dict:
    """Marque `step` (`"train"`/`"generate"`/`"metrics"`) comme `"done"` dans
    `state.json`, avec des champs additionnels optionnels (`**extra`) fusionnés
    au niveau racine de l'état. Réutilisable par l'orchestration des étapes
    `generate`/`metrics` (notebook), hors du périmètre de `train_final_model`."""
    state = load_state(run_dir)
    state.setdefault("steps", {})[step] = "done"
    state.update(extra)
    save_state(run_dir, state)
    return state


# ---------------------------------------------------------------------------
# Comparaison d'hyperparamètres -- un dossier de run est lié à SES
# hyperparamètres. Réutilisée pour le refus explicite ci-dessous ET par le
# notebook (avertissement de divergence à l'étape 10, `loadBestModel`).
# ---------------------------------------------------------------------------
def params_differences(old: dict, new: dict) -> list[str]:
    """Compare deux dicts d'hyperparamètres, strict sur l'ensemble des clés et
    sur `tokenizationConfig` (égalité exacte), tolérant sur les valeurs
    numériques (`math.isclose`, pour absorber les arrondis flottants du type
    `0.30000000000000004` vs `0.3`). Renvoie la liste des différences
    lisibles, vide si `old`/`new` sont équivalents."""
    diffs: list[str] = []
    for key in sorted(set(old) | set(new)):
        if key not in old:
            diffs.append(f"'{key}' absent (run existant) vs {new[key]!r} (demandé)")
            continue
        if key not in new:
            diffs.append(f"'{key}'={old[key]!r} (run existant) vs absent (demandé)")
            continue
        old_val, new_val = old[key], new[key]
        if isinstance(old_val, bool) or isinstance(new_val, bool):
            equal = old_val == new_val
        elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            equal = math.isclose(float(old_val), float(new_val), rel_tol=1e-6, abs_tol=1e-9)
        else:
            equal = old_val == new_val
        if not equal:
            diffs.append(f"'{key}'={old_val!r} (run existant) vs {new_val!r} (demandé)")
    return diffs


# ---------------------------------------------------------------------------
# Entraînement final reprenable.
# ---------------------------------------------------------------------------
def train_final_model(
    run_id: str,
    model_factory: Callable[[], nn.Module],
    loaders: dict[str, DataLoader],
    params: dict,
    run_dir: str | Path,
    max_epochs: int,
    device: torch.device,
    grad_clip: float = 1.0,
    patience: int = 5,
    pad_id: int = 0,
    teacher_forcing_ratio: float = 1.0,
    objective_fn: Callable[[nn.Module], float] | None = None,
    direction: str = "minimize",
    seed: int | None = None,
    on_epoch_end: Callable[[int, float], bool] | None = None,
) -> dict:
    """Entraîne `model_factory()` jusqu'à `max_epochs` (ou early stopping), en
    reprenant AUTOMATIQUEMENT depuis `run_dir` si un `checkpoint.pt` ou un
    `state.json` marquant l'étape "train" terminée y existent déjà :

    - étape "train" déjà `"done"` : sautée entièrement, `history.json` relu tel quel ;
    - `checkpoint.pt` présent : modèle/optimiseur/historique/RNG rechargés, la
      boucle repart à `epoch + 1` ;
    - sinon : entraînement neuf, `seed` (si fourni) initialise les RNG.

    `params` doit contenir `"learningRate"`, optionnellement
    `"teacherForcingDecay"` (défaut 1.0). `on_epoch_end(epoch, metric_value)`,
    si fourni, est appelé APRÈS l'écriture du checkpoint de chaque epoch --
    une exception s'y propage sans être interceptée (point d'accroche pour
    simuler une interruption : l'epoch reste sauvegardée mais "train" n'est
    pas marqué terminé) ; un retour `True` arrête proprement l'entraînement.

    `direction` ("minimize"/"maximize") gouverne le sens d'amélioration de la
    métrique pilotée (`objective_fn(model)` si fourni, sinon val loss) et est
    stockée dans `state.json` : reprendre avec l'AUTRE direction lève une
    `ValueError` plutôt que de comparer silencieusement `bestMetric` dans le
    mauvais sens.

    Garde-fou (n'interrompt rien) : si `objective_fn` est `None` et que la val
    loss en génération libre croît sans interruption sur au moins 3 epochs, un
    avertissement signale qu'elle n'est plus fiable pour l'early stopping
    (biais d'exposition) alors qu'une métrique comme le SacreBLEU continue de
    s'améliorer.

    Renvoie `{"history", "bestValLoss", "bestObjective", "resumed", "startEpoch"}`.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError(f"direction doit être 'minimize' ou 'maximize', reçu {direction!r}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"

    state = load_state(run_dir)
    state.setdefault("steps", {step: "pending" for step in DEFAULT_STEPS})

    # Un dossier de run est lié à SES hyperparamètres :
    # si ce run_dir a déjà des `params` enregistrés (reprise partielle OU run
    # "done") et qu'ils diffèrent de ceux demandés ici, on REFUSE -- jamais de
    # reprise sur un checkpoint incompatible, jamais de "déjà terminé" silencieux
    # sur un modèle qui n'est pas celui demandé. Vérifié AVANT le fast-path
    # "train done" ci-dessous, qui ne doit jamais être atteint dans ce cas.
    params_existants = state.get("params")
    if params_existants is not None:
        diffs = params_differences(params_existants, params)
        if diffs:
            raise ValueError(
                f"run '{run_id}' déjà entraîné avec d'autres hyperparamètres "
                f"({'; '.join(diffs)}) -- utilise un autre run_id ou un autre tag."
            )

    # Même logique que `params_existants` ci-dessus, pour `direction` : un
    # `run_dir` déjà entraîné (au moins partiellement -- `params` déjà
    # enregistrés) l'a été dans UN sens d'amélioration de `bestMetric` ; le
    # reprendre avec l'autre sens comparerait `checkpoint["bestMetric"]`
    # (ex. une val loss, `-inf` initial jamais dépassé) comme s'il s'agissait
    # d'un SacreBLEU, ou l'inverse -- silencieusement faux. Un `run_dir`
    # antérieur à ce garde-fou n'a jamais eu de `direction` stockée : il a
    # nécessairement tourné avec l'ancien défaut `"minimize"` (aucun autre
    # chemin de code n'existait), donc `None` vaut `"minimize"` ici.
    if params_existants is not None:
        direction_existante = state.get("direction") or "minimize"
        if direction_existante != direction:
            raise ValueError(
                f"run '{run_id}' déjà entraîné avec direction='{direction_existante}' "
                f"({'stockée dans state.json' if state.get('direction') else 'implicite, run antérieur au garde-fou de direction'}) "
                f"mais direction='{direction}' demandée maintenant -- incompatible (le sens "
                "d'amélioration de bestMetric ne correspond plus) -- utilise un autre run_id/tag "
                "ou repars de zéro (supprime ce run_dir)."
            )

    state["runId"] = run_id
    state["params"] = params
    state["direction"] = direction

    if state["steps"].get("train") == "done":
        print(f"[etape 9] run '{run_id}': étape 'train' déjà terminée (state.json) -- reprise du résultat sauvegardé, aucun calcul.")
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        return {
            "history": history,
            "bestValLoss": state.get("bestValLoss", float("nan")),
            "bestObjective": state.get("bestObjective", float("nan")),
            "resumed": True,
            "startEpoch": state.get("lastEpoch", 0) + 1,
        }

    teacher_forcing_decay = float(params.get("teacherForcingDecay", 1.0))
    lr = float(params.get("learningRate", params.get("lr", 1e-3)))

    # `seed` doit couvrir l'initialisation du modèle ET la boucle d'entraînement
    # (scheduled sampling, shuffle du DataLoader...) -- appliqué AVANT
    # `model_factory()` pour que les DEUX soient déterministes à partir d'un
    # même `seed`. Sans objet en cas de reprise : `set_rng_state` ci-dessous
    # écrase de toute façon cet état par celui sauvegardé en fin d'epoch
    # précédente (et les poids par `load_state_dict`).
    if not checkpoint_path.exists() and seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    model = model_factory()
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "trainLoss": [],
        "valLoss": [],
        "valPerplexity": [],
        "teacherForcingRatio": [],
        "bleu": [],
    }
    best_val_loss = float("inf")
    best_metric = float("inf") if direction == "minimize" else float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1
    resumed = False
    val_loss_croissante_avertie = False

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["modelStateDict"])
        optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        history = checkpoint["history"]
        best_val_loss = checkpoint["bestValLoss"]
        best_metric = checkpoint["bestMetric"]
        epochs_without_improvement = checkpoint["epochsWithoutImprovement"]
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["rngState"]["torch"])
        np.random.set_state(checkpoint["rngState"]["numpy"])
        random.setstate(checkpoint["rngState"]["python"])
        resumed = True
        print(
            f"[etape 9] run '{run_id}': reprise depuis checkpoint.pt (dernière epoch terminée : "
            f"{checkpoint['epoch']}) -> repart à l'epoch {start_epoch}/{max_epochs}."
        )

    if start_epoch > max_epochs:
        print(f"[etape 9] run '{run_id}': max_epochs ({max_epochs}) déjà atteint par le checkpoint, rien à ré-entraîner.")

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_tf_ratio = _scheduled_teacher_forcing_ratio(
            epoch, teacher_forcing_ratio, teacher_forcing_decay
        )

        model.train()
        train_loss_sum, n_train_batches = 0.0, 0
        for src, tgt in loaders["train"]:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt, epoch_tf_ratio)
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
        history["teacherForcingRatio"].append(epoch_tf_ratio)

        best_val_loss = min(best_val_loss, val_loss)

        metric_value = objective_fn(model) if objective_fn is not None else val_loss
        # `setdefault` : un `history.json` antérieur à ce correctif (repris depuis un
        # `checkpoint.pt` plus ancien) n'a pas encore la clé "bleu".
        history.setdefault("bleu", []).append(metric_value if objective_fn is not None else None)
        improved = (
            metric_value < best_metric - 1e-4
            if direction == "minimize"
            else metric_value > best_metric + 1e-4
        )

        # Garde-fou (n'interrompt rien) : sans `objective_fn`, l'early stopping
        # est piloté par la val loss en génération libre -- si elle croît sans
        # interruption depuis l'epoch 1 sur au moins 3 epochs, le meilleur
        # modèle qu'elle désignera sera systématiquement sous-entraîné (biais
        # d'exposition : la CE en génération libre explose dès que la séquence
        # générée s'écarte de la référence). Affiché une seule fois par run.
        if (
            objective_fn is None
            and not val_loss_croissante_avertie
            and len(history["valLoss"]) >= 3
            and all(
                history["valLoss"][i] < history["valLoss"][i + 1]
                for i in range(len(history["valLoss"]) - 1)
            )
        ):
            print(
                f"[etape 9] run '{run_id}' ATTENTION : la val loss en génération libre croît "
                "depuis l'epoch 1 -- l'early stopping sur cette métrique sélectionnera un "
                "modèle sous-entraîné ; passer un objective_fn (BLEU)."
            )
            val_loss_croissante_avertie = True

        if improved:
            best_metric = metric_value
            epochs_without_improvement = 0
            _atomic_torch_save(best_path, model.state_dict())
        else:
            epochs_without_improvement += 1

        # Checkpoint de REPRISE écrit à CHAQUE fin d'epoch, pas seulement au
        # meilleur -- c'est ce qui permet de repartir à `epoch + 1` après une
        # interruption, quelle que soit la qualité de cette epoch.
        _atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "modelStateDict": model.state_dict(),
                "optimizerStateDict": optimizer.state_dict(),
                "history": history,
                "bestValLoss": best_val_loss,
                "bestMetric": best_metric,
                "epochsWithoutImprovement": epochs_without_improvement,
                "rngState": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
        )
        _atomic_write_json(history_path, history)

        state["lastEpoch"] = epoch
        state["bestValLoss"] = best_val_loss
        state["bestObjective"] = best_metric
        save_state(run_dir, state)

        bleu_suffix = f" bleu={metric_value:.4f}" if objective_fn is not None else ""
        print(
            f"[etape 9] run '{run_id}' epoch {epoch}/{max_epochs}: trainLoss={train_loss:.4f} "
            f"valLoss={val_loss:.4f} valPpl={val_perplexity:.2f}{bleu_suffix} tfRatio={epoch_tf_ratio:.2f}"
        )

        # Appelé APRÈS l'écriture du checkpoint (déjà repris au redémarrage) --
        # une exception ici (pruning, interruption simulée) se propage SANS
        # être interceptée, `state.json` ne marque alors PAS "train" comme
        # terminé : la prochaine reprise repartira à `epoch + 1`.
        should_stop = bool(on_epoch_end(epoch, metric_value)) if on_epoch_end is not None else False
        if should_stop:
            print(f"[etape 9] run '{run_id}': arrêt demandé par on_epoch_end() après l'epoch {epoch}.")
            break

        if epochs_without_improvement >= patience:
            print(f"[etape 9] run '{run_id}': early stopping (patience={patience}) après l'epoch {epoch}.")
            break

    state["steps"]["train"] = "done"
    save_state(run_dir, state)

    return {
        "history": history,
        "bestValLoss": best_val_loss,
        "bestObjective": best_metric,
        "resumed": resumed,
        "startEpoch": start_epoch,
    }
