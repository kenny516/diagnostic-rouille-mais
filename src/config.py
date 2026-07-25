"""Configuration centrale et chemins du projet."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
HEALTHY_DIR = DATASET_DIR / "saines"
DISEASED_DIR = DATASET_DIR / "malades"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"

# Evite les erreurs de cache Matplotlib dans les environnements Windows restreints.
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

RANDOM_STATE = 42
TEST_SIZE = 0.20


def ensure_project_directories() -> None:
    """Cree les dossiers de sortie qui peuvent manquer."""

    for directory in (
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        UPLOADS_DIR,
        MATPLOTLIB_CONFIG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
