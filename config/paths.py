"""
config/paths.py
the paths for the project
including
- the root directory,
- config directory,
- datasets directory,
- checkpoints directory,
- outputs directory,
- logs directory,
- visualizations directory,
- best model path,
- last model path.


"""

from pathlib import Path

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# config directory
CONFIG_DIR = ROOT_DIR / "config"

# datasets directory
DATASETS_DIR = ROOT_DIR / "datasets"

# models checkpoints directory
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# outputs directory
OUTPUTS_DIR = ROOT_DIR / "outputs"

# logs directory
LOGS_DIR = ROOT_DIR / "logs"

# visualizations directory
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"

# best model directory
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"

# last model directory
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

if __name__ == "__main__":
    print(f"Project root directory: {ROOT_DIR}")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Visualizations directory: {VISUALIZATIONS_DIR}")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Last model path: {LAST_MODEL_PATH}")
