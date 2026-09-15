"""Tests de l'étape 9 (`src/training/final.py`) : entraînement final
REPRENABLE -- checkpoint + optimiseur + historique + RNG écrits à chaque
epoch, `state.json` comme source de vérité, écriture atomique.

Mini-jeu et mini-modèle (mêmes dimensions que `tests/test_training.py`) pour
garder le gate rapide -- aucune interruption n'est simulée en tuant un
processus (impossible dans pytest) : `on_epoch_end` qui lève une exception
joue ce rôle, exactement comme le pruning Optuna dans `train_and_validate`.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.training.final as final_module
from src.models.seq2seq import build_model
from src.training.final import (
    load_state,
    params_differences,
    train_final_model,
)

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


def _mini_model() -> nn.Module:
    return build_model(VOCAB_SIZE, VOCAB_SIZE, emb_dim=8, hidden_dim=16, cell_type="gru", dropout=0.0)


class _InterruptionSimulee(Exception):
    """Simule une coupure (process tué, exception non gérée) après une epoch."""


class _CroissanceForceeModel(nn.Module):
    """Modèle factice dont la CE croît de façon strictement monotone à CHAQUE appel
    forward (train et val confondus) -- force une val loss croissante de façon
    déterministe pour tester le garde-fou (lot correctif étape 9), sans dépendre du
    chaos d'un vrai entraînement divergent (non reproductible de façon fiable)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.call_count = 0
        self._dummy = nn.Parameter(torch.zeros(1))  # un seul paramètre, requis par l'optimiseur

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        self.call_count += 1
        cible_onehot = nn.functional.one_hot(
            tgt.clamp(min=0, max=self.vocab_size - 1), num_classes=self.vocab_size
        ).float()
        # logit de la classe cible de plus en plus négatif -> CE croissante, quel que
        # soit le contenu réel de src/tgt -- +0*self._dummy garde le graphe autograd.
        return -cible_onehot * (self.call_count * 2.0) + self._dummy * 0.0


# ---------------------------------------------------------------------------
# state.json -- helpers.
# ---------------------------------------------------------------------------
def test_load_state_sur_run_dir_neuf_renvoie_les_etapes_pending(tmp_path) -> None:
    state = load_state(tmp_path / "run-inexistant")
    assert state["steps"] == {"train": "pending", "generate": "pending", "metrics": "pending"}


def test_mark_step_done_ecrit_et_relit(tmp_path) -> None:
    run_dir = tmp_path / "run"
    final_module.mark_step_done(run_dir, "generate", nLignes=123)
    state = load_state(run_dir)
    assert state["steps"]["generate"] == "done"
    assert state["steps"]["train"] == "pending"  # pas touché
    assert state["nLignes"] == 123


# ---------------------------------------------------------------------------
# Écriture atomique.
# ---------------------------------------------------------------------------
def test_checkpoint_reste_lisible_si_l_ecriture_est_interrompue(tmp_path, monkeypatch) -> None:
    """Une écriture qui échoue en cours de route (`torch.save` qui lève) ne
    doit JAMAIS corrompre/remplacer le `checkpoint.pt` déjà présent."""
    chemin = tmp_path / "checkpoint.pt"
    final_module._atomic_torch_save(chemin, {"valeur": 1})
    assert chemin.exists()
    assert torch.load(chemin, weights_only=False) == {"valeur": 1}

    def torch_save_qui_plante(*args, **kwargs) -> None:
        raise OSError("coupure simulée pendant l'écriture")

    monkeypatch.setattr(final_module.torch, "save", torch_save_qui_plante)

    with pytest.raises(OSError):
        final_module._atomic_torch_save(chemin, {"valeur": 2})

    # le fichier tmp ne doit pas traîner, et le fichier final reste l'ANCIEN contenu
    assert not chemin.with_name(chemin.name + ".tmp").exists()
    assert torch.load(chemin, weights_only=False) == {"valeur": 1}


def test_state_json_reste_lisible_si_l_ecriture_est_interrompue(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    final_module.save_state(run_dir, {"steps": {"train": "done"}})

    reel_open = open

    def open_qui_plante(path, mode="r", *args, **kwargs):
        if str(path).endswith(".tmp") and "w" in mode:
            raise OSError("coupure simulée pendant l'écriture")
        return reel_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(final_module, "open", open_qui_plante, raising=False)

    with pytest.raises(OSError):
        final_module.save_state(run_dir, {"steps": {"train": "pending"}})

    state = load_state(run_dir)
    assert state["steps"]["train"] == "done"  # inchangé


# ---------------------------------------------------------------------------
# Reprise après interruption -- le coeur de la demande.
# ---------------------------------------------------------------------------
def test_reprise_apres_interruption_simulee_repart_a_la_bonne_epoch(tmp_path) -> None:
    loaders = _mini_loaders()
    run_dir = tmp_path / "run-interruption"
    params = {"learningRate": 0.01}

    def on_epoch_end_qui_plante_a_l_epoch_3(epoch: int, metric_value: float) -> bool:
        # Le checkpoint de l'epoch 3 est déjà écrit AVANT cet appel (voir
        # `train_final_model`) -- la coupure simulée ici correspond à un
        # process tué juste après l'epoch 3, avant le début de l'epoch 4.
        if epoch == 3:
            raise _InterruptionSimulee("coupure simulée après l'epoch 3")
        return False

    with pytest.raises(_InterruptionSimulee):
        train_final_model(
            run_id="test-run",
            model_factory=_mini_model,
            loaders=loaders,
            params=params,
            run_dir=run_dir,
            max_epochs=4,
            device=torch.device("cpu"),
            patience=10,
            seed=0,
            on_epoch_end=on_epoch_end_qui_plante_a_l_epoch_3,
        )

    # --- état juste après le "crash" ---------------------------------------
    assert (run_dir / "checkpoint.pt").exists()
    etat_apres_crash = load_state(run_dir)
    assert etat_apres_crash["steps"]["train"] == "pending"  # PAS marqué terminé
    assert etat_apres_crash["lastEpoch"] == 3
    with open(run_dir / "history.json", encoding="utf-8") as f:
        historique_apres_crash = json.load(f)
    assert len(historique_apres_crash["trainLoss"]) == 3

    # --- reprise : doit repartir à l'epoch 4, pas de zéro -------------------
    resultat = train_final_model(
        run_id="test-run",
        model_factory=_mini_model,
        loaders=loaders,
        params=params,
        run_dir=run_dir,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    assert resultat["resumed"] is True
    assert resultat["startEpoch"] == 4
    assert len(resultat["history"]["trainLoss"]) == 4
    # les 3 premières epochs (déjà réalisées avant le crash) sont préservées à l'identique
    assert resultat["history"]["trainLoss"][:3] == pytest.approx(historique_apres_crash["trainLoss"])
    assert resultat["history"]["valLoss"][:3] == pytest.approx(historique_apres_crash["valLoss"])

    assert load_state(run_dir)["steps"]["train"] == "done"

    # --- ré-appel : étape déjà terminée -> sautée entièrement, aucun recalcul
    resultat_saute = train_final_model(
        run_id="test-run",
        model_factory=_mini_model,
        loaders=loaders,
        params=params,
        run_dir=run_dir,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
    )
    assert resultat_saute["resumed"] is True
    assert resultat_saute["history"]["trainLoss"] == resultat["history"]["trainLoss"]


def test_reprise_produit_le_meme_resultat_qu_un_entrainement_non_interrompu(tmp_path) -> None:
    """Reprendre après une "interruption" à l'epoch 3 doit produire EXACTEMENT
    le même historique (4 décimales) qu'un entraînement de 4 epochs jamais
    interrompu, à seed égale -- preuve que la reprise (modèle + optimiseur +
    RNG restaurés) est une continuation transparente, pas une approximation."""
    params = {"learningRate": 0.01}

    # --- run A : jamais interrompu ------------------------------------------
    run_dir_a = tmp_path / "run-continu"
    resultat_continu = train_final_model(
        run_id="continu",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir_a,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    # --- run B : interrompu après l'epoch 3, puis repris --------------------
    run_dir_b = tmp_path / "run-interrompu"

    def on_epoch_end_qui_plante_a_l_epoch_3(epoch: int, metric_value: float) -> bool:
        if epoch == 3:
            raise _InterruptionSimulee()
        return False

    with pytest.raises(_InterruptionSimulee):
        train_final_model(
            run_id="interrompu",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=params,
            run_dir=run_dir_b,
            max_epochs=4,
            device=torch.device("cpu"),
            patience=10,
            seed=0,
            on_epoch_end=on_epoch_end_qui_plante_a_l_epoch_3,
        )
    resultat_repris = train_final_model(
        run_id="interrompu",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir_b,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
    )

    assert resultat_repris["history"]["trainLoss"] == pytest.approx(
        resultat_continu["history"]["trainLoss"], abs=1e-6
    )
    assert resultat_repris["history"]["valLoss"] == pytest.approx(
        resultat_continu["history"]["valLoss"], abs=1e-6
    )


def test_on_epoch_end_qui_renvoie_true_arrete_proprement_et_marque_termine(tmp_path) -> None:
    """Contrairement à une exception, un `on_epoch_end` qui renvoie `True` est
    un arrêt VOLONTAIRE (comme l'early stopping) : `state.json` marque bien
    "train" comme terminé, pas de reprise nécessaire."""
    run_dir = tmp_path / "run-stop-volontaire"
    appels = []

    def on_epoch_end(epoch: int, metric_value: float) -> bool:
        appels.append(epoch)
        return epoch >= 2

    resultat = train_final_model(
        run_id="stop-volontaire",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params={"learningRate": 0.01},
        run_dir=run_dir,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
        on_epoch_end=on_epoch_end,
    )

    assert appels == [1, 2]
    assert len(resultat["history"]["trainLoss"]) == 2
    assert load_state(run_dir)["steps"]["train"] == "done"


# ---------------------------------------------------------------------------
# Lot L -- un dossier de run est lié à SES hyperparamètres : refus explicite
# sur incohérence, jamais de reprise/« déjà terminé » silencieux sur un
# checkpoint entraîné avec d'autres hyperparamètres.
# ---------------------------------------------------------------------------
def test_params_differences_detecte_les_ecarts_et_tolere_les_flottants() -> None:
    ancien = {"tokenizationConfig": "bpe4k", "hiddenDim": 256, "dropout": 0.30000000000000004}

    # identique à tolérance flottante près -- aucune différence
    assert params_differences(ancien, {"tokenizationConfig": "bpe4k", "hiddenDim": 256, "dropout": 0.3}) == []

    diffs = params_differences(ancien, {"tokenizationConfig": "unigram8k", "hiddenDim": 1024, "dropout": 0.3})
    assert any("tokenizationConfig" in d for d in diffs)
    assert any("hiddenDim" in d for d in diffs)


def test_reprise_avec_params_identiques_fonctionne(tmp_path) -> None:
    """Non-régression : reprendre un run existant avec EXACTEMENT les mêmes
    hyperparamètres (nouvel objet dict, mêmes valeurs) doit toujours marcher."""
    run_dir = tmp_path / "run-params-identiques"
    params = {"learningRate": 0.01, "hiddenDim": 16, "tokenizationConfig": "bpe4k"}

    train_final_model(
        run_id="params-identiques",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    resultat = train_final_model(
        run_id="params-identiques",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=dict(params),  # nouvel objet, mêmes valeurs
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
    )
    assert resultat["resumed"] is True


def test_reprise_avec_hidden_dim_different_leve_une_erreur_explicite(tmp_path) -> None:
    run_dir = tmp_path / "run-hidden-dim-different"
    params = {"learningRate": 0.01, "hiddenDim": 16, "tokenizationConfig": "bpe4k"}

    train_final_model(
        run_id="hidden-dim-different",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    with pytest.raises(ValueError, match="hiddenDim"):
        train_final_model(
            run_id="hidden-dim-different",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=dict(params, hiddenDim=256),
            run_dir=run_dir,
            max_epochs=1,
            device=torch.device("cpu"),
            patience=10,
        )


def test_reprise_avec_tokenization_config_different_leve_une_erreur_explicite(tmp_path) -> None:
    run_dir = tmp_path / "run-tokenization-config-different"
    params = {"learningRate": 0.01, "hiddenDim": 16, "tokenizationConfig": "bpe4k"}

    train_final_model(
        run_id="tokenization-config-different",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    with pytest.raises(ValueError, match="tokenizationConfig"):
        train_final_model(
            run_id="tokenization-config-different",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=dict(params, tokenizationConfig="unigram8k"),
            run_dir=run_dir,
            max_epochs=1,
            device=torch.device("cpu"),
            patience=10,
        )


def test_run_marque_train_done_avec_params_differents_leve_une_erreur_pas_de_saut_silencieux(tmp_path) -> None:
    """Le cas le plus dangereux : un run marqué "train":
    "done" avec des params différents ne doit JAMAIS être silencieusement
    considéré comme "déjà terminé" -- il doit lever AVANT ce fast-path."""
    run_dir = tmp_path / "run-done-params-differents"
    params = {"learningRate": 0.01, "hiddenDim": 16, "tokenizationConfig": "bpe4k"}

    train_final_model(
        run_id="done-params-differents",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )
    assert load_state(run_dir)["steps"]["train"] == "done"

    with pytest.raises(ValueError):
        train_final_model(
            run_id="done-params-differents",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=dict(params, hiddenDim=999),
            run_dir=run_dir,
            max_epochs=1,
            device=torch.device("cpu"),
            patience=10,
        )


def test_comparaison_params_tolere_les_flottants_quasi_egaux_dans_train_final_model(tmp_path) -> None:
    run_dir = tmp_path / "run-tolerance-flottants"
    params = {"learningRate": 0.01, "dropout": 0.30000000000000004}

    train_final_model(
        run_id="tolerance-flottants",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )

    resultat = train_final_model(
        run_id="tolerance-flottants",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=dict(params, dropout=0.3),  # égal à tolérance flottante près
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
    )
    assert resultat["resumed"] is True


# ---------------------------------------------------------------------------
# Correctif étape 9 : `objective_fn`/`direction` pilotent best.pt/l'early
# stopping (pas seulement la val loss), garde-fou de val loss croissante,
# refus explicite d'une reprise à direction incompatible.
# ---------------------------------------------------------------------------
def test_objective_fn_pilote_best_pt_sur_l_epoch_de_meilleur_objectif_pas_de_meilleure_val_loss(tmp_path) -> None:
    """`objective_fn` remplace la val loss pour piloter `best.pt` -- cas construit où
    les deux divergent : l'objectif (artificiel, indépendant de la qualité réelle du
    modèle) culmine à l'epoch 2, alors que la val loss (calculée normalement, sur le
    même entraînement) ne s'améliore pas forcément à cette même epoch -- `best.pt`
    doit correspondre EXACTEMENT aux poids de l'epoch 2, pas à ceux de la meilleure
    val loss ni à ceux de la dernière epoch."""
    objectif_par_epoch = {1: 0.1, 2: 0.9, 3: 0.4, 4: 0.2}  # pic à l'epoch 2
    appels: list[int] = []
    etats_captures: dict[int, dict] = {}

    def objective_fn(model: nn.Module) -> float:
        epoch = len(appels) + 1
        appels.append(epoch)
        etats_captures[epoch] = {k: v.clone() for k, v in model.state_dict().items()}
        return objectif_par_epoch[epoch]

    run_dir = tmp_path / "run-objectif-diverge"
    resultat = train_final_model(
        run_id="objectif-diverge",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params={"learningRate": 0.01},
        run_dir=run_dir,
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
        objective_fn=objective_fn,
        direction="maximize",
    )

    assert resultat["bestObjective"] == pytest.approx(0.9)  # le pic de l'objectif (epoch 2)
    assert resultat["history"]["bleu"] == pytest.approx([0.1, 0.9, 0.4, 0.2])

    # divergence : la meilleure val loss (calculée normalement) ne coïncide PAS avec
    # l'epoch de meilleur objectif -- sinon le cas ne prouverait rien.
    meilleure_epoch_val_loss = (
        min(range(len(resultat["history"]["valLoss"])), key=lambda i: resultat["history"]["valLoss"][i]) + 1
    )
    assert meilleure_epoch_val_loss != 2, (
        "cas dégénéré : la meilleure val loss coïncide avec l'epoch de meilleur objectif -- "
        "changer objectif_par_epoch pour forcer la divergence"
    )

    # best.pt == poids capturés à l'epoch 2 (meilleur objectif), au tenseur près
    best_state = torch.load(run_dir / "best.pt", weights_only=True)
    for cle, valeur in etats_captures[2].items():
        assert torch.equal(best_state[cle], valeur)


def test_reprise_avec_direction_differente_de_celle_stockee_est_refusee(tmp_path) -> None:
    run_dir = tmp_path / "run-direction-differente"
    params = {"learningRate": 0.01}

    train_final_model(
        run_id="direction-differente",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
        objective_fn=lambda m: 1.0,
        direction="maximize",
    )

    with pytest.raises(ValueError, match="direction"):
        train_final_model(
            run_id="direction-differente",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=params,
            run_dir=run_dir,
            max_epochs=1,
            device=torch.device("cpu"),
            patience=10,
        )  # direction="minimize" (défaut) -- incompatible avec "maximize" stockée


def test_reprise_run_legacy_sans_direction_stockee_est_traitee_comme_minimize(tmp_path) -> None:
    """Un `run_dir` entraîné AVANT ce correctif n'a jamais eu de `direction` dans
    `state.json` -- il a nécessairement tourné avec l'ancien défaut implicite
    `"minimize"` (aucun autre chemin de code n'existait) : reprendre avec
    `direction="maximize"` doit être refusé -- exactement le cas du run v3/gru en
    cours, qui devra repartir de zéro."""
    run_dir = tmp_path / "run-legacy"
    params = {"learningRate": 0.01}

    train_final_model(
        run_id="legacy",
        model_factory=_mini_model,
        loaders=_mini_loaders(),
        params=params,
        run_dir=run_dir,
        max_epochs=1,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )
    # simule un state.json ANTÉRIEUR à ce correctif (jamais eu de champ "direction")
    state = load_state(run_dir)
    del state["direction"]
    final_module.save_state(run_dir, state)

    with pytest.raises(ValueError, match="direction"):
        train_final_model(
            run_id="legacy",
            model_factory=_mini_model,
            loaders=_mini_loaders(),
            params=params,
            run_dir=run_dir,
            max_epochs=1,
            device=torch.device("cpu"),
            patience=10,
            objective_fn=lambda m: 1.0,
            direction="maximize",
        )


def test_garde_fou_val_loss_croissante_sans_objective_fn_s_affiche(tmp_path, capsys) -> None:
    resultat = train_final_model(
        run_id="garde-fou",
        model_factory=lambda: _CroissanceForceeModel(VOCAB_SIZE),
        loaders=_mini_loaders(),
        params={"learningRate": 0.01},
        run_dir=tmp_path / "run-garde-fou",
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
    )
    assert resultat["history"]["valLoss"] == sorted(resultat["history"]["valLoss"])  # strictement croissante
    sortie = capsys.readouterr().out
    assert "ATTENTION" in sortie
    assert "objective_fn" in sortie


def test_garde_fou_ne_s_affiche_pas_avec_objective_fn(tmp_path, capsys) -> None:
    train_final_model(
        run_id="garde-fou-avec-objectif",
        model_factory=lambda: _CroissanceForceeModel(VOCAB_SIZE),
        loaders=_mini_loaders(),
        params={"learningRate": 0.01},
        run_dir=tmp_path / "run-garde-fou-avec-objectif",
        max_epochs=4,
        device=torch.device("cpu"),
        patience=10,
        seed=0,
        objective_fn=lambda m: 1.0,
        direction="maximize",
    )
    sortie = capsys.readouterr().out
    assert "ATTENTION" not in sortie
