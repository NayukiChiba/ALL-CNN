"""
交互式 CLI 菜单系统

渐进式菜单引导用户选择模型、数据集、配置参数并执行操作.
一行指令模式:python main.py train --model vgg16 --dataset cifar10
交互模式:python main.py(无参数时进入)

用法:
    from cnnlib.cli.interactive import InteractiveCLI

    cli = InteractiveCLI()
    cli.run()
"""

import os
from typing import List

import torch

from cnnlib.registry.datasets import get_dataset_info, list_datasets
from cnnlib.registry.models import get_model_info, list_models
from config.defaults import DefaultParams


# ANSI 颜色
class _C:
    """终端颜色代码"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def _printBox(title: str, width: int = 64) -> None:
    """打印带边框的标题框"""
    print(f"\n{_C.BOLD}{_C.CYAN}╔{'═' * (width - 2)}╗")
    print(f"║ {title:<{width - 4}} ║")
    print(f"╚{'═' * (width - 2)}╝{_C.END}\n")


def _section(text: str) -> None:
    print(f"\n{_C.BOLD}{_C.BLUE}  ▸ {text}{_C.END}")


def _info(key: str, value: str) -> None:
    print(f"    {_C.DIM}{key}:{_C.END} {_C.GREEN}{value}{_C.END}")


def _warn(text: str) -> None:
    print(f"  {_C.YELLOW}⚠ {text}{_C.END}")


def _error(text: str) -> None:
    print(f"  {_C.RED}✗ {text}{_C.END}")


def _success(text: str) -> None:
    print(f"  {_C.GREEN}✓ {text}{_C.END}")


def _prompt(text: str, default: str = "") -> str:
    """带默认值的输入提示"""
    if default:
        prompt = f"  {text} [{_C.DIM}{default}{_C.END}]: "
    else:
        prompt = f"  {text}: "
    result = input(prompt).strip()
    return result if result else default


def _promptChoice(text: str, options: List[str], default: str = "") -> str:
    """从选项中选一个"""
    print(f"\n  {text}:")
    for i, opt in enumerate(options, 1):
        print(f"    {_C.BOLD}{i}{_C.END}. {opt}")
    choice = _prompt("  选择", default)
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except (ValueError, IndexError):
        pass
    return choice if choice in options else options[0]


def _promptInt(text: str, default: int, minV: int = 1, maxV: int = 9999) -> int:
    """整数输入"""
    val = _prompt(text, str(default))
    try:
        result = int(val)
        return max(minV, min(maxV, result))
    except ValueError:
        return default


def _promptFloat(
    text: str, default: float, minV: float = 0.0, maxV: float = 100.0
) -> float:
    """浮点数输入"""
    val = _prompt(text, str(default))
    try:
        result = float(val)
        return max(minV, min(maxV, result))
    except ValueError:
        return default


def _showRegisteredModels() -> None:
    """显示已注册的模型列表"""
    models = list_models()
    _section(f"已注册模型({len(models)} 个)")
    print()
    for name in sorted(models):
        info = get_model_info(name)
        print(
            f"  {_C.BOLD}{name:<12}{_C.END} "
            f"input={info['input_size']}x{info['input_size']}  "
            f"ch={info['channels']}  "
            f"{_C.DIM}{info['description']}{_C.END}"
        )


def _showRegisteredDatasets() -> None:
    """显示已注册的数据集列表"""
    datasets = list_datasets()
    _section(f"已注册数据集({len(datasets)} 个)")
    print()
    for name in sorted(datasets):
        info = get_dataset_info(name)
        sizeStr = (
            f"{info['image_size']}x{info['image_size']}"
            if info["image_size"]
            else "variable"
        )
        print(
            f"  {_C.BOLD}{name:<14}{_C.END} "
            f"ch={info['channels']}  classes={info['num_classes']:<4}  "
            f"size={sizeStr:<10}  "
            f"{_C.DIM}{info['description']}{_C.END}"
        )


class InteractiveCLI:
    """交互式命令行界面"""

    def __init__(self):
        self.model = "lenet"
        self.dataset = "mnist"
        self.device = DefaultParams.DEVICE
        self.batchSize = 64
        self.epochs = 20
        self.lr = 0.001
        self.optimizer = "adam"
        self.valSplit = 0.1
        self.gradClip = 0.0

    def run(self) -> None:
        """主循环"""
        self._printBanner()

        while True:
            self._showMainMenu()
            choice = input(f"\n{_C.BOLD}  >>> {_C.END}").strip()

            if choice == "0":
                print(f"\n{_C.DIM}  再见！{_C.END}\n")
                break
            elif choice == "1":
                self._selectModel()
            elif choice == "2":
                self._selectDataset()
            elif choice == "3":
                self._trainWorkflow()
            elif choice == "4":
                self._evalWorkflow()
            elif choice == "5":
                self._inferWorkflow()
            elif choice == "6":
                self._benchmarkWorkflow()
            elif choice == "7":
                _showRegisteredModels()
            elif choice == "8":
                _showRegisteredDatasets()
            elif choice == "9":
                self._settingsMenu()
            elif choice == "":
                continue
            else:
                _warn(f"无效选项: {choice}")

    def _printBanner(self) -> None:
        """打印欢迎横幅"""
        torch.cuda.init()
        gpuInfo = (
            f"CUDA: {torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else "无 GPU"
        )
        print(f"""{_C.BOLD}{_C.CYAN}
  ╔══════════════════════════════════════════════════════╗
  ║        CNN Image Classification CLI                  ║
  ║        8 种模型 · 10 个数据集 · 一键训练评估             ║
  ╚══════════════════════════════════════════════════════╝{_C.END}
  {_C.DIM}设备: {self.device} | {gpuInfo}{_C.END}
""")

    def _showMainMenu(self) -> None:
        """显示主菜单"""
        print(f"""
{_C.BOLD}{"─" * 50}{_C.END}
{_C.BOLD}  模型{_C.END}: {_C.GREEN}{self.model:<14}{_C.END} {_C.BOLD}数据集{_C.END}: {_C.GREEN}{self.dataset:<14}{_C.END} {_C.BOLD}设备{_C.END}: {_C.GREEN}{self.device}{_C.END}
{_C.BOLD}{"─" * 50}{_C.END}
  {_C.BOLD}1{_C.END}. 选择模型       {_C.DIM}5{_C.END}. 推理
  {_C.BOLD}2{_C.END}. 选择数据集     {_C.DIM}6{_C.END}. 基准测试
  {_C.BOLD}3{_C.END}. 训练模型       {_C.DIM}7{_C.END}. 查看模型列表
  {_C.BOLD}4{_C.END}. 评估模型       {_C.DIM}8{_C.END}. 查看数据集列表
                    {_C.DIM}9{_C.END}. 参数设置
  ──────────────────────────────────────────
  {_C.DIM}0{_C.END}. 退出
""")

    # ── 子菜单 ──────────────────────────────────────────────

    def _selectModel(self) -> None:
        """选择模型"""
        models = sorted(list_models())
        _printBox("选择模型")
        for i, name in enumerate(models, 1):
            info = get_model_info(name)
            print(
                f"  {_C.BOLD}{i}{_C.END}. {name:<12} "
                f"input={info['input_size']}x{info['input_size']}  "
                f"ch={info['channels']}  "
                f"{_C.DIM}{info['description']}{_C.END}"
            )
        choice = _prompt("\n  选择模型编号", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                self.model = models[idx]
                _success(f"已选择: {self.model}")
                return
        except ValueError:
            pass
        _warn("保持当前选择")

    def _selectDataset(self) -> None:
        """选择数据集"""
        datasets = sorted(list_datasets())
        _printBox("选择数据集")
        for i, name in enumerate(datasets, 1):
            info = get_dataset_info(name)
            sizeStr = (
                f"{info['image_size']}x{info['image_size']}"
                if info["image_size"]
                else "variable"
            )
            print(
                f"  {_C.BOLD}{i}{_C.END}. {name:<14} "
                f"ch={info['channels']}  classes={info['num_classes']:<4}  "
                f"size={sizeStr:<10}  "
                f"{_C.DIM}{info['description']}{_C.END}"
            )
        choice = _prompt("\n  选择数据集编号", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                self.dataset = datasets[idx]
                _success(f"已选择: {self.dataset}")
                return
        except ValueError:
            pass
        _warn("保持当前选择")

    def _settingsMenu(self) -> None:
        """参数设置"""
        _printBox("参数设置")
        self.epochs = _promptInt("    训练轮数", self.epochs, 1, 500)
        self.batchSize = _promptInt("    批次大小", self.batchSize, 1, 1024)
        self.lr = _promptFloat("    学习率", self.lr, 1e-6, 1.0)
        self.valSplit = _promptFloat("    验证集比例", self.valSplit, 0.0, 0.5)
        self.gradClip = _promptFloat(
            "    梯度裁剪(0=不裁剪)", self.gradClip, 0.0, 100.0
        )
        self.optimizer = _promptChoice(
            "    优化器",
            ["adam", "adamw", "sgd", "rmsprop"],
            default=self.optimizer,
        )
        _success("参数已更新")

    def _trainWorkflow(self) -> None:
        """训练流程"""
        _printBox(f"训练: {self.model} + {self.dataset}")

        info = get_dataset_info(self.dataset)
        modelInfo = get_model_info(self.model)

        _info(
            "模型",
            f"{self.model} (input={modelInfo['input_size']}, ch={modelInfo['channels']})",
        )
        _info(
            "数据集",
            f"{self.dataset} (classes={info['num_classes']}, size={info['image_size']})",
        )
        _info("Epochs", str(self.epochs))
        _info("Batch Size", str(self.batchSize))
        _info("Learning Rate", str(self.lr))
        _info("Optimizer", self.optimizer)
        _info("设备", self.device)
        print()

        confirm = _prompt("  开始训练? [Y/n]", "y").lower()
        if confirm not in ("y", "yes", ""):
            _warn("已取消")
            return

        _success("启动训练...")
        print(f"\n{_C.DIM}{'─' * 60}{_C.END}\n")

        self._executeTrain()

    def _executeTrain(self) -> None:
        """实际执行训练"""
        from cnnlib.data.loader import build_dataloaders
        from cnnlib.models.factory import create_model_for_dataset
        from cnnlib.training import (
            EarlyStopping,
            Trainer,
            TrainingLogger,
            createLoss,
            createOptimizer,
            createScheduler,
        )
        from config.paths import (
            ensureDir,
            getBestModelPath,
            getCheckpointDir,
            getLogDir,
        )

        device = torch.device(self.device)

        # 模型
        model = create_model_for_dataset(self.model, self.dataset, device=self.device)
        _info("参数量", f"{model.param_count():,}")

        # 数据
        trainLoader, valLoader, testLoader = build_dataloaders(
            model_name=self.model,
            dataset_name=self.dataset,
            batch_size=self.batchSize,
            val_split=self.valSplit,
            num_workers=0,
            seed=42,
        )
        _info("训练集", f"{len(trainLoader.dataset):,} samples")
        _info("验证集", f"{len(valLoader.dataset):,} samples")
        _info("测试集", f"{len(testLoader.dataset):,} samples")

        # 损失 / 优化器 / 调度器
        lossFn = createLoss("cross_entropy")
        optimizer = createOptimizer(
            model, name=self.optimizer, lr=self.lr, weight_decay=1e-4
        )
        scheduler = createScheduler(optimizer, "plateau", factor=0.5, patience=3)

        # 日志 & 早停
        logDir = getLogDir(self.model, self.dataset)
        ensureDir(logDir)
        logger = TrainingLogger(logDir, self.model, self.dataset)
        earlyStopping = EarlyStopping(patience=10)

        checkpointDir = getCheckpointDir(self.model, self.dataset)
        ensureDir(checkpointDir)

        trainer = Trainer(
            model=model,
            trainLoader=trainLoader,
            valLoader=valLoader,
            testLoader=testLoader,
            optimizer=optimizer,
            scheduler=scheduler,
            lossFn=lossFn,
            device=device,
            epochs=self.epochs,
            checkpointDir=checkpointDir,
            logger=logger,
            earlyStopping=earlyStopping,
            gradClip=self.gradClip,
        )

        result = trainer.train()

        print(f"\n{_C.BOLD}{'─' * 50}{_C.END}")
        bestPath = getBestModelPath(self.model, self.dataset)
        _success(f"训练完成！最佳模型: {bestPath}")
        _info(
            "最佳 val_acc",
            f"{result['best_metric']:.2f}% (epoch {result['best_epoch']})",
        )
        if result["test_metrics"]:
            _info("测试 acc", f"{result['test_metrics']['accuracy']:.2f}%")
            _info("测试 loss", f"{result['test_metrics']['loss']:.4f}")

    def _evalWorkflow(self) -> None:
        """评估流程"""
        _printBox(f"评估: {self.model} + {self.dataset}")

        from config.paths import getBestModelPath

        defaultPath = str(getBestModelPath(self.model, self.dataset))
        ckptPath = _prompt("  checkpoint 路径", defaultPath)

        if not os.path.exists(ckptPath):
            _error(f"checkpoint 不存在: {ckptPath}")
            return

        confirm = _prompt("  开始评估? [Y/n]", "y").lower()
        if confirm not in ("y", "yes", ""):
            _warn("已取消")
            return

        self._executeEval(ckptPath)

    def _executeEval(self, checkpointPath: str) -> None:
        """实际执行评估"""
        import torch

        from cnnlib.data.loader import build_dataloaders
        from cnnlib.models.factory import create_model_for_dataset
        from cnnlib.training import createLoss, loadCheckpoint, validate

        device = torch.device(self.device)

        model = create_model_for_dataset(self.model, self.dataset, device=self.device)
        ckpt = loadCheckpoint(checkpointPath, model, device=self.device)
        _info("checkpoint epoch", str(ckpt["epoch"]))

        _, _, testLoader = build_dataloaders(
            model_name=self.model,
            dataset_name=self.dataset,
            batch_size=self.batchSize,
            val_split=self.valSplit,
            num_workers=0,
            seed=42,
        )

        lossFn = createLoss("cross_entropy")
        metrics = validate(model, testLoader, lossFn, device, desc="Test")

        print(f"\n{_C.BOLD}  评估结果:{_C.END}")
        _info("Loss", f"{metrics['loss']:.4f}")
        _info("Accuracy", f"{metrics['accuracy']:.2f}%")

    def _inferWorkflow(self) -> None:
        """推理流程"""
        _printBox(f"推理: {self.model} + {self.dataset}")

        from config.paths import getBestModelPath

        defaultCkpt = str(getBestModelPath(self.model, self.dataset))
        ckptPath = _prompt("  checkpoint 路径", defaultCkpt)
        if not os.path.exists(ckptPath):
            _error(f"checkpoint 不存在: {ckptPath}")
            return

        imagePath = _prompt("  图片路径")
        if not imagePath:
            _warn("请输入图片路径")
            return
        if not os.path.exists(imagePath):
            _error(f"图片不存在: {imagePath}")
            return

        topK = _promptInt("  Top-K", 3, 1, 20)

        self._executeInfer(ckptPath, imagePath, topK)

    def _executeInfer(self, checkpointPath: str, imagePath: str, topK: int) -> None:
        """实际执行推理"""
        import torch

        from cnnlib.data.transform import build_transform
        from cnnlib.inference.predictor import Predictor
        from cnnlib.models.factory import create_model_for_dataset
        from cnnlib.training import loadCheckpoint

        device = torch.device(self.device)

        model = create_model_for_dataset(self.model, self.dataset, device=self.device)
        loadCheckpoint(checkpointPath, model, device=self.device)
        transform = build_transform(self.model, self.dataset, augment=False)

        predictor = Predictor(model, transform, device=self.device)
        results = predictor.predictFromFile(imagePath, topK=topK)

        print(f"\n{_C.BOLD}  Top-{topK} 预测:{_C.END}")
        for rank, r in enumerate(results, 1):
            label = r.get("class", f"class_{r['class_idx']}")
            print(
                f"    {_C.BOLD}{rank}{_C.END}. "
                f"{_C.GREEN}{label}{_C.END} "
                f"({r['confidence']:.4f})"
            )

    def _benchmarkWorkflow(self) -> None:
        """基准测试流程"""
        _printBox("基准测试")

        _warn("基准测试将下载数据集并训练多个模型,耗时较长")
        print()

        _info("当前模型", self.model)
        _info("当前数据集", self.dataset)

        testEpochs = _promptInt("    训练轮数", 3, 1, 10)
        confirm = _prompt("  开始基准测试? [y/N]", "n").lower()
        if confirm not in ("y", "yes"):
            _warn("已取消")
            return

        self._executeBenchmark(testEpochs)

    def _executeBenchmark(self, epochs: int) -> None:
        """实际执行基准测试"""
        from cnnlib.experiments.benchmark import runBenchmark

        _success("启动基准测试...")
        print()

        result = runBenchmark(
            modelName=self.model,
            datasetName=self.dataset,
            device=self.device,
            epochs=epochs,
            batchSize=self.batchSize,
            numWorkers=0,
        )

        print(f"\n{_C.BOLD}  基准测试结果:{_C.END}")
        _info("模型", result["model"])
        _info("数据集", result["dataset"])
        _info("参数量", f"{result['params']:,}")
        _info("模型大小", f"{result['model_size_mb']:.1f} MB")
        _info("推理时间", f"{result['inference_time_ms']:.1f} ms/batch")
        _info("最佳 val_acc", f"{result.get('best_val_acc', 0):.2f}%")
        if result.get("test_acc"):
            _info("测试 acc", f"{result['test_acc']:.2f}%")
