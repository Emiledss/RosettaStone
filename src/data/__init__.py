"""Chargement, nettoyage, déduplication et split du corpus (étapes 0→3).

Symbole du contrat consommé par `notebooks/Rosetta_Optuna.ipynb` : `load_splits`.
"""

from src.data.splits import load_splits

__all__ = ["load_splits"]
