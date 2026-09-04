"""Racine du projet ajoutée à `sys.path` pour que `import src...` fonctionne sous pytest.

Sans ce fichier, pytest (mode d'import "prepend") n'insère que le répertoire
`tests/` dans `sys.path` (absence de `__init__.py` dans `tests/`), pas la racine
du dépôt : `import src.data` échouerait avec `ModuleNotFoundError`.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
