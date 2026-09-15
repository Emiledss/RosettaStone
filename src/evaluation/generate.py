"""Génération libre (étape 10, brique réutilisée par l'objectif Optuna) :
décodage glouton, autorégressif -- le décodeur repart de `<sos>` et ne se
nourrit QUE de ses propres prédictions précédentes, jamais du `tgt` du
dataloader. Aucun teacher forcing : c'est le point de contrôle central de
l'étape 10 (`docs/plan-seq2seq.md`) -- les métriques finales (SacreBLEU,
METEOR, BERTScore) doivent porter sur du texte réellement généré, pas sur des
prédictions gonflées par le vrai token précédent.

`generate_translations_resumable` (bas du fichier) enveloppe cette fonction
pour une génération REPRENABLE sur le test complet (39 517 paires) : écrit les
hypothèses au fil des batchs dans un fichier JSONL et, en cas d'interruption,
ne régénère au redémarrage que ce qui manque (équivalent, côté génération, de
`src/training/final.py` côté entraînement).

Padding dynamique par batch : `TranslationDataset` (`src/tokenization/
numerize.py`) stocke des séquences de longueur variable, non paddées -- le
`DataLoader` construit ci-dessous par `generate_translations_
resumable` doit donc réutiliser le même `collate_fn` (padding par batch,
source/cible indépendamment) que `make_dataloaders`, sous peine d'échouer au
premier batch de longueurs hétérogènes (le collate par défaut de PyTorch exige
des tenseurs de même taille). Sans effet sur les tests de ce module, qui
utilisent des `TensorDataset` déjà de longueur fixe : `collate_fn` s'y comporte
alors comme le collate par défaut.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.seq2seq import PAD_ID
from src.tokenization.numerize import collate_fn as _collate_batch_dynamique
from src.tokenization.vocab_builder import Vocabulary


def generate_translations(
    model: nn.Module,
    dataloader: DataLoader,
    en_vocab: Vocabulary,
    device: torch.device,
    max_len: int | None = None,
    batch_callback: Callable[[int, list[str]], None] | None = None,
) -> list[str]:
    """Génère des traductions en génération libre (glouton, autorégressif).

    Pour chaque batch, le décodeur repart de `<sos>` (constante, `en_vocab.sos_id`
    -- jamais lue depuis `tgt`) et se nourrit à chaque pas de son propre `argmax`
    précédent. `model.encoder(src)` puis `model.decoder.forward_step(input_token,
    hidden, encoder_outputs)` sont appelés directement (pas
    `model.forward(src, tgt, ...)`) : cette fonction n'a besoin -- et ne reçoit
    -- jamais la cible réelle.

    `max_len=None` -> déduit de `tgt.shape[1]` du batch courant (le padding fixe
    posé par `src.tokenization.numerize.make_dataloaders` ; `tgt` sert alors
    uniquement à lire cette forme, jamais son contenu). Le nombre de pas générés
    est `max_len - 1` (la position 0, `<sos>`, est déjà consommée en entrée --
    même convention que `Seq2Seq.forward`).

    Chaque hypothèse s'arrête au premier `<eos>` prédit (troncature avant
    décodage, pas un simple filtrage a posteriori) : les tokens qui suivraient
    ne sont jamais générés pour cette phrase. `en_vocab.decode` retire en plus
    tout `<pad>`/`<sos>`/`<eos>` résiduel -- la sortie ne contient donc jamais
    ces tokens en texte.

    `batch_callback(index_debut, hypotheses_du_batch)`, si fourni, est appelé
    après chaque batch avec l'indice de sa première phrase dans la liste
    renvoyée -- point d'accroche pour la sauvegarde incrémentale (étape 10).

    `torch.no_grad()` + `model.eval()` : aucun gradient, dropout désactivé.
    """
    model.eval()
    hypotheses: list[str] = []
    index_debut = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            batch_max_len = max_len if max_len is not None else int(tgt.shape[1])
            src = src.to(device)
            batch_size = src.shape[0]

            encoder_outputs, hidden = model.encoder(src)
            # (batch, src_len), True sur les vraies positions -- transmis à l'attention
            # pour ignorer le padding pendant la génération libre aussi. `PAD_ID`
            # (constante du module, pas `model.encoder.embedding.padding_idx`) pour rester
            # compatible avec les encodeurs factices des tests (`test_evaluation.py`).
            src_mask = (src != PAD_ID).to(device)
            input_token = torch.full(
                (batch_size, 1), en_vocab.sos_id, dtype=torch.long, device=device
            )
            generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max(batch_max_len - 1, 0)):
                prediction, hidden = model.decoder.forward_step(
                    input_token, hidden, encoder_outputs, src_mask
                )
                top1 = prediction.argmax(dim=1)  # (batch,)
                for i in range(batch_size):
                    if finished[i]:
                        continue
                    token_id = int(top1[i].item())
                    if token_id == en_vocab.eos_id:
                        finished[i] = True
                    else:
                        generated_ids[i].append(token_id)
                if bool(finished.all()):
                    break
                input_token = top1.unsqueeze(1)

            batch_hypotheses = [en_vocab.decode(ids) for ids in generated_ids]
            hypotheses.extend(batch_hypotheses)

            if batch_callback is not None:
                batch_callback(index_debut, batch_hypotheses)
            index_debut += batch_size

    return hypotheses


# ---------------------------------------------------------------------------
# Génération REPRENABLE (étape 10) : écrit `hypotheses_path` (JSONL) au fil des
# batchs via `batch_callback` -- à la reprise, ne régénère que ce qui manque.
# ---------------------------------------------------------------------------
def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)
        os.replace(tmp, path)  # atomique (POSIX rename / Windows MoveFileEx)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_existing_hypotheses(path: Path) -> list[str]:
    """Relit `path` (JSONL, une ligne `{"index": i, "hypothesis": h}` par
    phrase déjà générée, dans l'ordre) -- renvoie `[]` si `path` n'existe pas.

    S'arrête au premier écart dans la séquence (ligne illisible ou `index`
    hors-suite) : c'est la signature d'une écriture interrompue en cours de
    ligne. Le préfixe valide qui précède cet écart est alors RÉÉCRIT de façon
    atomique (`_atomic_write_lines`) pour purger la queue corrompue avant que
    l'appelant ne reprenne l'ajout -- le fichier ne peut donc jamais accumuler
    de contenu tronqué au fil des reprises successives.
    """
    if not path.exists():
        return []

    lignes_brutes = path.read_text(encoding="utf-8").splitlines()
    hypotheses: list[str] = []
    for i, ligne in enumerate(lignes_brutes):
        try:
            enregistrement = json.loads(ligne)
        except json.JSONDecodeError:
            break
        if not isinstance(enregistrement, dict) or enregistrement.get("index") != i:
            break
        hypotheses.append(enregistrement["hypothesis"])

    if len(hypotheses) != len(lignes_brutes):
        _atomic_write_lines(
            path,
            [json.dumps({"index": i, "hypothesis": h}, ensure_ascii=False) for i, h in enumerate(hypotheses)],
        )
        print(
            f"[etape 10] {path.name} réparé : {len(lignes_brutes)} ligne(s) lues, "
            f"{len(hypotheses)} valide(s) conservée(s) (reste purgé, régénéré ci-après)."
        )

    return hypotheses


def generate_translations_resumable(
    model: nn.Module,
    dataset: Dataset,
    en_vocab: Vocabulary,
    device: torch.device,
    hypotheses_path: str | Path,
    batch_size: int = 64,
    max_len: int | None = None,
) -> list[str]:
    """Génère les traductions de `dataset` (génération libre, voir
    `generate_translations`) et les écrit AU FIL DE L'EAU dans
    `hypotheses_path` (JSONL, une ligne par phrase). REPREND automatiquement :
    si `hypotheses_path` existe déjà, compte les hypothèses déjà écrites (via
    `_read_existing_hypotheses`, qui purge aussi une éventuelle queue
    corrompue par une interruption précédente) et ne régénère QUE le reste,
    dans l'ordre du dataset (`Subset(dataset, range(n_deja, len(dataset)))`).

    Renvoie la liste COMPLÈTE des hypothèses (déjà écrites + nouvellement
    générées), dans l'ordre original du dataset -- longueur == `len(dataset)`.
    Ne régénère jamais une phrase déjà générée lors d'un appel précédent.
    """
    hypotheses_path = Path(hypotheses_path)
    dataset_size = len(dataset)  # type: ignore[arg-type]

    hypotheses_existantes = _read_existing_hypotheses(hypotheses_path)
    n_deja = len(hypotheses_existantes)

    if n_deja >= dataset_size:
        print(f"[etape 10] {hypotheses_path.name}: génération déjà complète ({n_deja}/{dataset_size}) -- rien à régénérer.")
        return hypotheses_existantes[:dataset_size]

    if n_deja > 0:
        print(
            f"[etape 10] {hypotheses_path.name}: reprise -- {n_deja}/{dataset_size} hypothèses déjà écrites, "
            f"{dataset_size - n_deja} restantes à générer."
        )
    sous_dataset = Subset(dataset, range(n_deja, dataset_size)) if n_deja > 0 else dataset
    loader = DataLoader(
        sous_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch_dynamique
    )

    hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hypotheses_path, "a", encoding="utf-8") as f:

        def _batch_callback(index_debut: int, batch_hypotheses: list[str]) -> None:
            for offset, hypothese in enumerate(batch_hypotheses):
                enregistrement = {"index": n_deja + index_debut + offset, "hypothesis": hypothese}
                f.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
            f.flush()  # visible immédiatement sur disque, pas seulement en fin de génération

        nouvelles_hypotheses = generate_translations(
            model, loader, en_vocab, device, max_len=max_len, batch_callback=_batch_callback
        )

    return hypotheses_existantes + nouvelles_hypotheses
