"""Tests d'invariants structurels sur les notebooks du projet.

Ces tests ne ré-exécutent pas les notebooks (coûteux, et dépend de `src/` pour
`Rosetta_Modelisation.ipynb`) : ils vérifient des invariants bon marché mais utiles
pour éviter de livrer un notebook corrompu ou un fichier contenant des outputs
périmés par erreur.
"""

import json
from pathlib import Path

import pytest

notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
aed_notebook = notebooks_dir / "Rosetta_AED.ipynb"
modelisation_notebook = notebooks_dir / "Rosetta_Modelisation.ipynb"
tokenisation_notebook = notebooks_dir / "Rosetta_Tokenisation.ipynb"
all_notebooks = [aed_notebook, modelisation_notebook, tokenisation_notebook]
# Notebooks livrés sans output stocké (voir `test_notebook_ne_stocke_aucun_output`
# ci-dessous) -- à l'inverse de `Rosetta_AED.ipynb`, qui stocke ses outputs par
# conception.
notebooks_sans_outputs = [modelisation_notebook, tokenisation_notebook]


def load_notebook(path: Path) -> dict:
    """Charge un notebook Jupyter depuis le disque et le parse en JSON."""
    with path.open(encoding="utf-8") as fichier:
        return json.load(fichier)


def code_cells_of(notebook: dict) -> list[dict]:
    """Retourne les cellules de code d'un notebook déjà chargé."""
    return [cellule for cellule in notebook["cells"] if cellule["cell_type"] == "code"]


@pytest.mark.parametrize("notebook_path", all_notebooks, ids=lambda p: p.name)
def test_notebook_est_du_json_valide_en_nbformat_4(notebook_path: Path) -> None:
    """Chaque notebook doit être du JSON valide, au format nbformat 4."""
    notebook = load_notebook(notebook_path)
    assert notebook["nbformat"] == 4


@pytest.mark.parametrize("notebook_path", all_notebooks, ids=lambda p: p.name)
def test_cellules_de_code_compilent(notebook_path: Path) -> None:
    """Chaque cellule de code doit être syntaxiquement valide (compile() réussit)."""
    notebook = load_notebook(notebook_path)
    cellules = code_cells_of(notebook)
    assert cellules, f"{notebook_path.name} ne contient aucune cellule de code"
    for indice, cellule in enumerate(cellules):
        source = "".join(cellule["source"])
        nom_module = f"<{notebook_path.name}:cellule {indice}>"
        compile(source, nom_module, "exec")


@pytest.mark.parametrize("notebook_path", notebooks_sans_outputs, ids=lambda p: p.name)
def test_notebook_ne_stocke_aucun_output(notebook_path: Path) -> None:
    """
    Convention de livraison : `Rosetta_Modelisation.ipynb` et `Rosetta_Tokenisation.ipynb`
    doivent être livrés sans output stocké, pour rester diffables et ne jamais
    contenir un résultat périmé.

    `execution_count` n'est volontairement PAS vérifié : c'est un entier posé par
    Jupyter à chaque exécution, sans coût de taille ni risque de résultat périmé.
    L'exiger à `None` ferait échouer les tests après toute exécution manuelle, sans
    rien protéger. Cette convention ne s'applique pas à `Rosetta_AED.ipynb`, qui
    stocke ses outputs par conception.
    """
    notebook = load_notebook(notebook_path)
    for cellule in code_cells_of(notebook):
        assert cellule["outputs"] == []
