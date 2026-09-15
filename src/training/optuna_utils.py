"""Utilitaires Optuna partagés par `notebooks/Rosetta_Modelisation.ipynb`
(`runStudy`) : budget cumulatif entre reprises et échecs non fatals.

Extraits dans `src/` pour être testables avec des études Optuna **en
mémoire** (rapides, sans toucher au vrai pipeline). Défauts corrigés :

1. **Budget non cumulatif.** `study.optimize(objective, n_trials=nTrials)`
   lance `nTrials` essais **NEUFS** à chaque appel, pas « atteindre `nTrials`
   au total » -- chaque reprise ajoutait donc `nTrials` essais de plus,
   indéfiniment. Voir `remaining_trials` : `nTrials` devient une cible TOTALE.
2. **Échec fatal pour toute l'étude.** Une exception levée dans l'objectif
   remontait et interrompait `study.optimize` -- sur une étude de plusieurs
   heures, une seule epoch qui plante perdait tout le reste. Voir
   `log_trial_failure` (trace complète, en append) et
   `wrap_objective_with_failure_logging` / `run_study_with_budget`
   (`catch=(Exception,)`, `optuna.TrialPruned` jamais intercepté -- c'est le
   fonctionnement normal du pruning, pas un échec).
3. **`FAIL` comptait pour le budget quelle qu'en soit la cause.** Or une
   interruption (`KeyboardInterrupt`, coupure de courant, ...) EN COURS
   d'essai marque aussi celui-ci `FAIL` -- sans distinction, la reprise ne
   relance donc pas cet essai (l'objectif n'a jamais réellement produit de
   résultat) ET le décompte comme s'il en avait produit un.
   `wrap_objective_with_failure_logging` MARQUE désormais l'essai qu'il
   journalise (`trial.set_user_attr("failureLogged", True)`), et
   `remaining_trials` ne compte un `FAIL` que s'il porte cette marque -- un
   `FAIL` sans marque (interruption) ou un `RUNNING` périmé ne consomme plus
   le budget. `KeyboardInterrupt` hérite de `BaseException`, pas
   d'`Exception` : il ne passe jamais par le journaliseur, donc jamais marqué
   -- vérifié par test, pas supposé.
4. **`study.best_params` ne contient jamais `tokenizationConfig`**, mais
   `tokenizationMethod`/`vocabSize` (dérivé par `suggestHyperparams`, pas un
   paramètre Optuna suggéré). Un accès `hp.get("tokenizationConfig", default)`
   à l'étape 9 retombait donc SILENCIEUSEMENT sur la config par défaut -- un
   entraînement final de plusieurs heures pouvait tourner sur la mauvaise
   config sans lever la moindre exception. `resolve_tokenization_config`
   élimine ce repli muet.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import optuna

from src.tokenization.vocab_builder import config_name

# États d'essai qui ont TOUJOURS consommé du calcul -- comptent dans le budget
# cumulatif sans condition. FAIL n'est PAS ici (défaut 3 ci-dessus) : un FAIL
# ne compte que s'il porte la marque `failureLogged` posée par
# `wrap_objective_with_failure_logging`, voir `remaining_trials`.
_CONSUMED_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
)

# Clé d'user_attr posée par `wrap_objective_with_failure_logging` sur tout essai
# dont l'échec a été journalisé -- distingue un VRAI échec applicatif (exception
# capturée, tracée, essai marqué) d'une interruption (KeyboardInterrupt ou coupure
# en cours d'essai, jamais passée par le journaliseur, jamais marquée).
FAILURE_LOGGED_ATTR = "failureLogged"


def _consumes_budget(trial: optuna.trial.FrozenTrial) -> bool:
    """`True` si `trial` a réellement consommé du calcul, au sens du budget
    cumulatif : COMPLETE/PRUNED toujours, FAIL seulement s'il porte la
    marque `FAILURE_LOGGED_ATTR` (un FAIL sans marque est une interruption, pas
    un résultat produit -- RUNNING n'est jamais dans `_CONSUMED_STATES` et n'a
    pas de marque non plus, donc ne compte jamais)."""
    if trial.state in _CONSUMED_STATES:
        return True
    return trial.state == optuna.trial.TrialState.FAIL and bool(
        trial.user_attrs.get(FAILURE_LOGGED_ATTR, False)
    )


def remaining_trials(study: optuna.Study, n_trials: int) -> dict:
    """Calcule le nombre d'essais à lancer pour atteindre la CIBLE TOTALE
    `n_trials`, pas un nombre d'essais additionnels.

    Renvoie un dict :
    - `countsByState` : `{nom_etat: compte}` pour tous les états présents dans
      l'étude (COMPLETE, PRUNED, FAIL, RUNNING, WAITING...) -- sert à la
      ventilation affichée avant lancement (`format_budget_message`).
    - `alreadyDone` : nombre d'essais COMPLETE + PRUNED (`_CONSUMED_STATES`,
      toujours comptés) + FAIL **marqués** `failureLogged` (un FAIL issu d'une
      interruption -- jamais marqué -- ne consomme pas le budget, pour que la
      reprise le relance réellement au lieu de le compter comme un résultat
      produit).
    - `remaining` : `max(0, n_trials - alreadyDone)`, le nombre d'essais
      RÉELLEMENT à lancer.
    - `target` : `n_trials`, renvoyé tel quel pour affichage.
    """
    counts_by_state: dict[str, int] = {}
    for trial in study.trials:
        counts_by_state[trial.state.name] = counts_by_state.get(trial.state.name, 0) + 1

    already_done = sum(1 for trial in study.trials if _consumes_budget(trial))
    remaining = max(0, n_trials - already_done)

    return {
        "countsByState": counts_by_state,
        "alreadyDone": already_done,
        "remaining": remaining,
        "target": n_trials,
    }


def format_budget_message(study_name: str, budget: dict) -> str:
    """Message affiché avant lancement : ventilation par état + nombre
    RÉELLEMENT à lancer -- doit rester sans ambiguïté sur le budget déjà
    consommé.

    `budget` : le dict renvoyé par `remaining_trials`.
    """
    ventilation = (
        ", ".join(f"{etat}={compte}" for etat, compte in sorted(budget["countsByState"].items()))
        or "aucun essai enregistré"
    )
    entete = (
        f"Étude '{study_name}' -- {budget['alreadyDone']} essai(s) déjà comptabilisé(s) "
        f"dans le budget sur {budget['target']} visé(s) au total ({ventilation})."
    )
    if budget["remaining"] == 0:
        conclusion = (
            "Budget déjà atteint, 0 essai à lancer -- augmente n_trials pour en "
            "lancer davantage."
        )
    else:
        conclusion = f"Lancement de {budget['remaining']} essai(s) additionnel(s)."
    return f"{entete} {conclusion}"


def log_trial_failure(log_path: str | Path, trial: optuna.Trial, exc: BaseException) -> None:
    """Journalise en APPEND la trace complète d'un essai en échec dans
    `log_path` (fichier créé si absent, dossier parent créé si besoin) :
    horodatage (UTC, ISO 8601), numéro d'essai, hyperparamètres, traceback
    complet.

    ⚠️ Ne JAMAIS appeler cette fonction pour `optuna.TrialPruned` -- c'est le
    fonctionnement normal du pruning, pas un échec
    (`wrap_objective_with_failure_logging` applique cette règle).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now(timezone.utc).isoformat()
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"=== {horodatage} -- essai {trial.number} -- params={trial.params} ===\n"
            f"{trace}\n"
        )


def wrap_objective_with_failure_logging(
    objective_fn: Callable[[optuna.Trial], float], log_path: str | Path
) -> Callable[[optuna.Trial], float]:
    """Enveloppe `objective_fn` : toute exception AUTRE que `optuna.TrialPruned`
    est journalisée (`log_trial_failure`), MARQUE l'essai
    (`trial.set_user_attr(FAILURE_LOGGED_ATTR, True)`), un avertissement
    visible est affiché, puis l'exception est RELEVÉE -- Optuna marque alors
    l'essai `FAIL` et, combiné à `catch=(Exception,)` dans `study.optimize`,
    l'étude CONTINUE au lieu de s'interrompre.

    La marque `failureLogged` est ce qui permet à `remaining_trials` de
    distinguer un VRAI échec applicatif (passé par ici, donc marqué) d'une
    interruption en cours d'essai (`KeyboardInterrupt` ou coupure -- hérite de
    `BaseException`, jamais capturée ici, donc jamais marquée) : seul le
    premier consomme le budget cumulatif.

    `optuna.TrialPruned` traverse SANS jamais être journalisée, marquée, ni
    comptée comme un échec : c'est le fonctionnement normal du pruning.
    """

    def wrapped(trial: optuna.Trial) -> float:
        try:
            return objective_fn(trial)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            log_trial_failure(log_path, trial, exc)
            trial.set_user_attr(FAILURE_LOGGED_ATTR, True)
            print(
                f"[AVERTISSEMENT] essai {trial.number} en échec ({type(exc).__name__}: {exc}) "
                f"-- trace journalisée dans {log_path}, l'étude continue."
            )
            raise

    return wrapped


def run_study_with_budget(
    study: optuna.Study,
    objective_fn: Callable[[optuna.Trial], float],
    n_trials: int,
    log_path: str | Path,
    callbacks: Sequence[Callable] | None = None,
) -> optuna.Study:
    """Orchestre le budget cumulatif et les échecs non fatals autour de
    `study.optimize` :

    1. calcule le budget restant (`remaining_trials`) et affiche la
       ventilation par état (`format_budget_message`) ;
    2. si le budget est déjà atteint (`remaining == 0`), N'APPELLE PAS
       `study.optimize` et renvoie l'étude telle quelle (les artefacts
       CSV/JSON restent à réécrire par l'appelant -- ils doivent être à jour
       même sans nouvel essai) ;
    3. sinon, enveloppe `objective_fn` (`wrap_objective_with_failure_logging`)
       et appelle `study.optimize(..., n_trials=remaining, catch=(Exception,))` :
       un essai qui plante est marqué `FAIL`, journalisé, et l'étude CONTINUE.
       `optuna.TrialPruned` n'est JAMAIS intercepté par `catch` -- Optuna le
       traite en interne comme le mécanisme normal du pruning, pas comme une
       exception applicative ;
    4. affiche un décompte des échecs (`FAIL`) sur l'étude entière en fin
       d'appel.
    """
    budget = remaining_trials(study, n_trials)
    print(format_budget_message(study.study_name, budget))

    if budget["remaining"] == 0:
        return study

    wrapped_objective = wrap_objective_with_failure_logging(objective_fn, log_path)
    study.optimize(
        wrapped_objective,
        n_trials=budget["remaining"],
        callbacks=list(callbacks) if callbacks is not None else None,
        catch=(Exception,),
    )

    n_failures = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    print(
        f"Étude '{study.study_name}' -- {n_failures} échec(s) au total sur "
        f"{len(study.trials)} essai(s)."
    )

    return study


def resolve_tokenization_config(
    params: dict,
    default: str,
    cell_type: str = "",
    source: str | Path = "",
) -> dict:
    """Garantit `params["tokenizationConfig"]`, quel que soit le format de
    `params` (cf. docstring du module) :

    - **ancien format** : `tokenizationConfig` déjà présent -- renvoyé tel quel,
      pour rester lisible sans migration.
    - **format courant** : `tokenizationMethod` + `vocabSize` présents (les
      paramètres RÉELLEMENT suggérés par Optuna, cf. `suggestHyperparams` du
      notebook) -- dérivé par `config_name()`, LA MÊME fonction que celle qui
      construit l'espace de recherche (`src.tokenization.vocab_builder`),
      jamais réimplémentée.
    - **ni l'un ni l'autre** (fichier corrompu ou modifié à la main -- ne devrait
      jamais arriver en pratique) : AUCUN repli muet. Avertissement TRÈS visible sur
      stdout (un entraînement de plusieurs heures ne doit jamais partir sur la
      mauvaise config sans que ça saute aux yeux), puis repli explicite sur `default`.

    `cell_type`/`source` ne servent qu'à l'avertissement (nom du run, chemin du
    fichier lu) -- optionnels, pour rester utilisable hors contexte notebook.
    """
    if "tokenizationConfig" in params:
        return dict(params)

    resolved = dict(params)
    if "tokenizationMethod" in params and "vocabSize" in params:
        resolved["tokenizationConfig"] = config_name(params["tokenizationMethod"], params["vocabSize"])
        return resolved

    resolved["tokenizationConfig"] = default
    bandeau = "!" * 78
    print(bandeau)
    print(
        f"[ATTENTION] {cell_type or 'run'} ({source or 'source inconnue'}) : "
        "ni 'tokenizationConfig' ni 'tokenizationMethod'/'vocabSize' dans bestParams."
    )
    print(
        f"[ATTENTION] REPLI SUR '{default}' -- VÉRIFIER CE FICHIER avant de lancer "
        "un entraînement de plusieurs heures sur la mauvaise config."
    )
    print(bandeau)
    return resolved
