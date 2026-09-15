"""Tests du lot correctif « budget cumulatif / échecs non fatals »
(`src/training/optuna_utils.py`).

Études Optuna **en mémoire** (`storage=None`), rapides, objectifs jouets --
aucune ne touche au vrai pipeline (données, modèle, entraînement).
"""

from __future__ import annotations

import optuna
import pytest

from src.training.optuna_utils import (
    FAILURE_LOGGED_ATTR,
    format_budget_message,
    log_trial_failure,
    remaining_trials,
    resolve_tokenization_config,
    run_study_with_budget,
    wrap_objective_with_failure_logging,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _etude_en_memoire(nom: str = "etude-test") -> optuna.Study:
    return optuna.create_study(study_name=nom, storage=None, direction="maximize")


def _objectif_toujours_complete(trial: optuna.Trial) -> float:
    trial.suggest_float("x", 0.0, 1.0)  # au moins un hyperparamètre, comme un vrai essai
    return 1.0


def _objectif_par_sequence(sequence: list[str]):
    """`sequence[trial.number]` vaut `"complete"`, `"pruned"` ou `"fail"` --
    pilote déterministe de l'état de chaque essai, indexé par `trial.number`
    (compteur GLOBAL de l'étude, cumulatif entre plusieurs appels)."""

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        etat = sequence[trial.number]
        if etat == "pruned":
            raise optuna.TrialPruned("élagué (test)")
        if etat == "fail":
            raise RuntimeError("échec simulé (test)")
        return 1.0

    return objective


# ---------------------------------------------------------------------------
# remaining_trials -- budget cumulatif.
# ---------------------------------------------------------------------------
def test_remaining_trials_etude_vide() -> None:
    etude = _etude_en_memoire()
    budget = remaining_trials(etude, n_trials=10)
    assert budget == {"countsByState": {}, "alreadyDone": 0, "remaining": 10, "target": 10}


def test_remaining_trials_compte_complete_et_pruned_dans_le_budget() -> None:
    etude = _etude_en_memoire()
    sequence = ["complete", "pruned", "complete"]
    etude.optimize(_objectif_par_sequence(sequence), n_trials=len(sequence), catch=(Exception,))

    budget = remaining_trials(etude, n_trials=10)
    assert budget["countsByState"] == {"COMPLETE": 2, "PRUNED": 1}
    assert budget["alreadyDone"] == 3
    assert budget["remaining"] == 7


# ---------------------------------------------------------------------------
# remaining_trials -- lot K : un FAIL ne consomme le budget que s'il est
# MARQUÉ (`failureLogged`, posé par `wrap_objective_with_failure_logging`) --
# un FAIL sans marque (interruption) ou un RUNNING périmé ne le consomment pas
#.
# ---------------------------------------------------------------------------
def test_remaining_trials_fail_marque_compte_fail_non_marque_et_running_ne_comptent_pas() -> None:
    etude = _etude_en_memoire()

    complete = etude.ask()
    complete.suggest_float("x", 0.0, 1.0)
    etude.tell(complete, 1.0, state=optuna.trial.TrialState.COMPLETE)

    pruned = etude.ask()
    pruned.suggest_float("x", 0.0, 1.0)
    etude.tell(pruned, state=optuna.trial.TrialState.PRUNED)

    fail_marque = etude.ask()
    fail_marque.suggest_float("x", 0.0, 1.0)
    fail_marque.set_user_attr(FAILURE_LOGGED_ATTR, True)  # comme wrap_objective_with_failure_logging
    etude.tell(fail_marque, state=optuna.trial.TrialState.FAIL)

    fail_non_marque = etude.ask()
    fail_non_marque.suggest_float("x", 0.0, 1.0)
    etude.tell(fail_non_marque, state=optuna.trial.TrialState.FAIL)  # interruption simulée : pas de marque

    running_perime = etude.ask()  # jamais "tell" -- reste RUNNING (interruption simulée)
    running_perime.suggest_float("x", 0.0, 1.0)

    budget = remaining_trials(etude, n_trials=10)
    assert budget["countsByState"] == {"COMPLETE": 1, "PRUNED": 1, "FAIL": 2, "RUNNING": 1}
    # COMPLETE + PRUNED + le FAIL marqué = 3 -- le FAIL non marqué et le RUNNING
    # périmé ne consomment PAS le budget.
    assert budget["alreadyDone"] == 3
    assert budget["remaining"] == 7


def test_run_study_with_budget_fail_marque_consomme_le_budget_a_la_reprise(tmp_path) -> None:
    """Intégration bout en bout via `run_study_with_budget` (donc
    `wrap_objective_with_failure_logging` réel) : un essai qui plante avec une
    VRAIE exception applicative est marqué et consomme le budget -- une reprise
    à la même cible ne relance rien de plus."""
    etude = _etude_en_memoire()
    sequence = ["complete", "fail", "complete"]
    log_path = tmp_path / "failures.log"

    run_study_with_budget(etude, _objectif_par_sequence(sequence), n_trials=3, log_path=log_path)
    assert len(etude.trials) == 3

    budget = remaining_trials(etude, n_trials=3)
    assert budget["alreadyDone"] == 3  # 2 COMPLETE + 1 FAIL marqué
    assert budget["remaining"] == 0


# ---------------------------------------------------------------------------
# wrap_objective_with_failure_logging -- pose (ou non) la marque failureLogged.
# ---------------------------------------------------------------------------
def test_wrap_objective_marque_l_essai_en_echec_applicatif(tmp_path) -> None:
    etude = _etude_en_memoire()
    trial = etude.ask()
    trial.suggest_float("x", 0.0, 1.0)

    def objectif_qui_plante(_t: optuna.Trial) -> float:
        raise RuntimeError("échec simulé (test)")

    wrapped = wrap_objective_with_failure_logging(objectif_qui_plante, tmp_path / "failures.log")

    with pytest.raises(RuntimeError):
        wrapped(trial)

    assert trial.user_attrs.get(FAILURE_LOGGED_ATTR) is True


def test_wrap_objective_ne_marque_pas_un_trial_pruned(tmp_path) -> None:
    """Non-régression (lot I) : `optuna.TrialPruned` n'est jamais journalisé
    comme un échec -- et, lot K, jamais marqué non plus."""
    etude = _etude_en_memoire()
    trial = etude.ask()
    trial.suggest_float("x", 0.0, 1.0)

    def objectif_elague(_t: optuna.Trial) -> float:
        raise optuna.TrialPruned("élagué (test)")

    wrapped = wrap_objective_with_failure_logging(objectif_elague, tmp_path / "failures.log")

    with pytest.raises(optuna.TrialPruned):
        wrapped(trial)

    assert FAILURE_LOGGED_ATTR not in trial.user_attrs
    assert not (tmp_path / "failures.log").exists()


def test_keyboardinterrupt_ne_marque_pas_et_ne_consomme_pas_le_budget(tmp_path) -> None:
    """KeyboardInterrupt hérite de BaseException, pas d'Exception : il ne passe
    jamais par `wrap_objective_with_failure_logging`, donc jamais marqué --
    vérifié ici plutôt que supposé."""
    etude = _etude_en_memoire()
    trial = etude.ask()
    trial.suggest_float("x", 0.0, 1.0)

    def objectif_interrompu(_t: optuna.Trial) -> float:
        raise KeyboardInterrupt()

    wrapped = wrap_objective_with_failure_logging(objectif_interrompu, tmp_path / "failures.log")

    with pytest.raises(KeyboardInterrupt):
        wrapped(trial)

    assert FAILURE_LOGGED_ATTR not in trial.user_attrs
    assert not (tmp_path / "failures.log").exists()

    # Optuna marquerait quand même cet essai FAIL (ou le laisserait RUNNING) --
    # simulé ici en le "tell"-ant FAIL sans marque, comme une reprise le verrait.
    etude.tell(trial, state=optuna.trial.TrialState.FAIL)
    budget = remaining_trials(etude, n_trials=5)
    assert budget["alreadyDone"] == 0  # l'interruption ne consomme pas le budget
    assert budget["remaining"] == 5


# ---------------------------------------------------------------------------
# run_study_with_budget -- budget déjà atteint / partiellement consommé.
# ---------------------------------------------------------------------------
def test_budget_deja_atteint_aucun_essai_supplementaire_lance(tmp_path) -> None:
    etude = _etude_en_memoire()
    etude.optimize(_objectif_toujours_complete, n_trials=5)
    assert len(etude.trials) == 5

    def objectif_qui_ne_doit_jamais_etre_appele(trial: optuna.Trial) -> float:
        raise AssertionError("l'objectif ne doit pas être appelé, le budget est déjà atteint")

    run_study_with_budget(
        etude,
        objectif_qui_ne_doit_jamais_etre_appele,
        n_trials=5,
        log_path=tmp_path / "failures.log",
    )

    assert len(etude.trials) == 5  # rien lancé de plus
    assert not (tmp_path / "failures.log").exists()


def test_budget_partiellement_consomme_lance_exactement_le_complement(tmp_path) -> None:
    etude = _etude_en_memoire()
    etude.optimize(_objectif_toujours_complete, n_trials=3)
    assert len(etude.trials) == 3

    run_study_with_budget(
        etude,
        _objectif_toujours_complete,
        n_trials=5,
        log_path=tmp_path / "failures.log",
    )

    assert len(etude.trials) == 5  # exactement le complément (2), pas 5 de plus


# ---------------------------------------------------------------------------
# format_budget_message -- message affiché avant lancement.
# ---------------------------------------------------------------------------
def test_format_budget_message_budget_deja_atteint() -> None:
    budget = {
        "countsByState": {"COMPLETE": 13, "PRUNED": 31, "FAIL": 2},
        "alreadyDone": 46,
        "remaining": 0,
        "target": 40,
    }
    message = format_budget_message("rosetta-rnn-v2", budget)
    assert "rosetta-rnn-v2" in message
    assert "budget déjà atteint" in message.lower()
    assert "0 essai à lancer" in message
    assert "COMPLETE=13" in message and "PRUNED=31" in message and "FAIL=2" in message


def test_format_budget_message_essais_a_lancer() -> None:
    budget = {"countsByState": {"COMPLETE": 3}, "alreadyDone": 3, "remaining": 2, "target": 5}
    message = format_budget_message("rosetta-gru-v2", budget)
    assert "lancement de 2 essai(s) additionnel(s)" in message.lower()


# ---------------------------------------------------------------------------
# Échecs non fatals, tracés -- l'étude continue, le pruning reste intact.
# ---------------------------------------------------------------------------
def test_objectif_qui_leve_une_exception_letude_continue_et_la_trace_est_ecrite(tmp_path) -> None:
    etude = _etude_en_memoire()
    sequence = ["complete", "fail", "complete"]  # essai 1 plante, 0 et 2 réussissent
    log_path = tmp_path / "failures_rnn.log"

    run_study_with_budget(etude, _objectif_par_sequence(sequence), n_trials=3, log_path=log_path)

    # l'étude a continué au-delà de l'échec : les 3 essais existent
    assert len(etude.trials) == 3
    etats = [t.state for t in etude.trials]
    assert etats == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]

    assert log_path.exists()
    contenu = log_path.read_text(encoding="utf-8")
    assert "essai 1" in contenu
    assert "RuntimeError" in contenu
    assert "échec simulé (test)" in contenu


def test_trial_pruned_nest_jamais_capture_comme_un_echec(tmp_path) -> None:
    etude = _etude_en_memoire()
    sequence = ["pruned", "complete"]
    log_path = tmp_path / "failures_gru.log"

    run_study_with_budget(etude, _objectif_par_sequence(sequence), n_trials=2, log_path=log_path)

    assert len(etude.trials) == 2
    assert etude.trials[0].state == optuna.trial.TrialState.PRUNED
    assert etude.trials[1].state == optuna.trial.TrialState.COMPLETE
    # rien n'est écrit dans le log d'échecs pour un pruning normal
    assert not log_path.exists()


def test_log_trial_failure_ecrit_en_append_sans_ecraser(tmp_path) -> None:
    etude = _etude_en_memoire()
    log_path = tmp_path / "failures.log"

    trial1 = etude.ask()
    trial1.suggest_float("x", 0.0, 1.0)
    log_trial_failure(log_path, trial1, RuntimeError("premier échec"))
    etude.tell(trial1, state=optuna.trial.TrialState.FAIL)

    trial2 = etude.ask()
    trial2.suggest_float("x", 0.0, 1.0)
    log_trial_failure(log_path, trial2, ValueError("second échec"))
    etude.tell(trial2, state=optuna.trial.TrialState.FAIL)

    contenu = log_path.read_text(encoding="utf-8")
    assert "premier échec" in contenu
    assert "second échec" in contenu
    assert contenu.index("premier échec") < contenu.index("second échec")


# ---------------------------------------------------------------------------
# resolve_tokenization_config -- bug lot J : study.best_params (v3) contient
# tokenizationMethod/vocabSize, jamais tokenizationConfig (dérivé par
# suggestHyperparams, pas un paramètre Optuna suggéré) -- les repos .get(...,
# tokenizationConfigDefault) de l'étape 9 retombaient SILENCIEUSEMENT sur
# words95. Voir docstring de `resolve_tokenization_config`.
# ---------------------------------------------------------------------------
def test_resolve_depuis_tokenization_method_et_vocab_size_v3() -> None:
    """Format v3 (Optuna a suggéré tokenizationMethod/vocabSize, jamais
    tokenizationConfig) -- doit dériver la bonne config via config_name()."""
    params = {
        "tokenizationMethod": "unigram",
        "vocabSize": 4000,
        "learningRate": 0.002,
        "hiddenDim": 512,
        "embDim": 256,
        "dropout": 0.1,
        "teacherForcingDecay": 0.95,
    }
    resolved = resolve_tokenization_config(params, default="unigram4k")
    assert resolved["tokenizationConfig"] == "unigram4k"
    # les autres hyperparamètres traversent inchangés
    assert resolved["learningRate"] == 0.002
    assert resolved["hiddenDim"] == 512


def test_resolve_conserve_lancien_format_v2_tel_quel() -> None:
    """Format v2 (tokenizationConfig déjà présent, cf. best_params_rnn.json réel) --
    les fichiers produits avant le lot J doivent rester lisibles sans migration."""
    params = {
        "tokenizationConfig": "bpe4k",
        "learningRate": 0.0013,
        "hiddenDim": 256,
        "embDim": 128,
        "dropout": 0.35,
        "teacherForcingDecay": 0.96,
    }
    resolved = resolve_tokenization_config(params, default="unigram4k")
    assert resolved["tokenizationConfig"] == "bpe4k"
    assert resolved == params  # inchangé, pas de champ tokenizationMethod/vocabSize ajouté


def test_resolve_ni_lancien_ni_le_nouveau_format_repli_visible_pas_muet(capsys) -> None:
    """Ni tokenizationConfig, ni tokenizationMethod/vocabSize : repli sur `default`,
    mais avec un avertissement TRÈS visible sur stdout -- jamais un repli muet."""
    params = {"learningRate": 0.001, "hiddenDim": 256, "embDim": 128, "dropout": 0.2}
    resolved = resolve_tokenization_config(
        params, default="unigram4k", cell_type="rnn", source="best_params_rnn.json"
    )
    assert resolved["tokenizationConfig"] == "unigram4k"
    sortie = capsys.readouterr().out
    assert "ATTENTION" in sortie
    assert "unigram4k" in sortie
    assert "rnn" in sortie
    assert "best_params_rnn.json" in sortie


def test_resolve_ne_mute_pas_le_dict_dorigine() -> None:
    params = {"tokenizationMethod": "bpe", "vocabSize": 6000}
    resolve_tokenization_config(params, default="unigram4k")
    assert "tokenizationConfig" not in params  # le dict passé en entrée n'est pas modifié
