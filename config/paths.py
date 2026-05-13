"""
项目路径管理

分为两层:
    1. 根目录常量 — 顶层目录
    2. 运行时路径函数 — 按 {模型}/{数据集}/ 层级自动拼接，调用时即确保目录存在

用法:
    from config.paths import getCheckpointDir, getBestModelPath

    ckptDir = getCheckpointDir("vgg16", "cifar10")
    # → checkpoints/vgg16/cifar10/

    best = getBestModelPath("vgg16", "cifar10")
    # → checkpoints/vgg16/cifar10/best_model.pth
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 1. 根目录常量
# ═══════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATASETS_DIR = ROOT_DIR / "datasets"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# 向后兼容（新代码请用 getBestModelPath / getLastModelPath）
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"


# ═══════════════════════════════════════════════════════════════
# 2. 运行时路径 — 按 {模型}/{数据集} 组织
# ═══════════════════════════════════════════════════════════════


def ensureDir(path: Path) -> Path:
    """确保目录存在，返回自身便于链式调用"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def getCheckpointDir(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/checkpoints/"""
    return OUTPUTS_DIR / model / dataset / "checkpoints"


def getBestModelPath(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/checkpoints/best_model.pth"""
    return getCheckpointDir(model, dataset) / "best_model.pth"


def getLastModelPath(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/checkpoints/last_model.pth"""
    return getCheckpointDir(model, dataset) / "last_model.pth"


def getOutputDir(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/"""
    return OUTPUTS_DIR / model / dataset


def getBenchmarkResultPath(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/result.json"""
    return getOutputDir(model, dataset) / "result.json"


def getBenchmarkSummaryPath() -> Path:
    """outputs/benchmark_summary.json"""
    return OUTPUTS_DIR / "benchmark_summary.json"


def getLogDir(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/logs/"""
    return getOutputDir(model, dataset) / "logs"


def getTensorboardDir(model: str, dataset: str) -> Path:
    """outputs/{model}/{dataset}/tensorboard/"""
    return getOutputDir(model, dataset) / "tensorboard"


def getVisualizationDir(model: str, dataset: str) -> Path:
    """visualizations/{model}/{dataset}/"""
    return OUTPUTS_DIR / model / dataset / "visualizations"


# ═══════════════════════════════════════════════════════════════
# 3. 调试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"ROOT:              {ROOT_DIR}")
    print(f"CONFIG:            {CONFIG_DIR}")
    print(f"DATASETS:          {DATASETS_DIR}")
    print(f"CHECKPOINTS:       {CHECKPOINTS_DIR}")
    print(f"OUTPUTS:           {OUTPUTS_DIR}")
    print(f"LOGS:              {LOGS_DIR}")
    print()
    print(f"BEST_MODEL_PATH:   {BEST_MODEL_PATH}")
    print(f"LAST_MODEL_PATH:   {LAST_MODEL_PATH}")
    print()
    print("--- 运行时路径示例 (vgg16, cifar10) ---")
    print(f"checkpoint dir:    {getCheckpointDir('vgg16', 'cifar10')}")
    print(f"best model:        {getBestModelPath('vgg16', 'cifar10')}")
    print(f"last model:        {getLastModelPath('vgg16', 'cifar10')}")
    print(f"output dir:        {getOutputDir('vgg16', 'cifar10')}")
    print(f"log dir:           {getLogDir('vgg16', 'cifar10')}")
    print(f"tensorboard dir:   {getTensorboardDir('vgg16', 'cifar10')}")
    print(f"visualization dir: {getVisualizationDir('vgg16', 'cifar10')}")
