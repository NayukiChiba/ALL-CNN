"""
src/train/logger.py

Training logger: console (stderr), file, and TensorBoard.

- Console: epoch / loss / accuracy / lr via stderr (tqdm-safe, won't clobber progress bars)
- File: timestamped text log with DEBUG-level detail
- TensorBoard: scalar curves + model graph for interactive exploration

Usage:
    from src.train.logger import TrainLogger

    logger = TrainLogger()
    logger.logConfig({"batch_size": 64, "lr": 0.001})
    logger.logGraph(model, sample_input)

    for epoch in range(1, total_epochs + 1):
        ...
        logger.logEpoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

    logger.logSummary(best_epoch, best_val_acc)
    logger.close()
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.paths import LOGS_DIR, OUTPUTS_DIR


class TrainLogger:
    """
    Three-channel training logger.

    Channels:
    1. stderr (INFO level) — visible in terminal, integrates with tqdm via sys.stderr
    2. File (DEBUG level) — full detail saved to LOOGS_DIR/train_<timestamp>.log
    3. TensorBoard — scalar curves and model graph under OUTPUTS_DIR/tensorboard/<timestamp>/

    Why sys.stderr for console:
        tqdm writes progress bars to sys.stderr. By also writing our log
        messages to sys.stderr via logging.StreamHandler, they interleave
        cleanly without corrupting the progress bar display (which would
        happen if we used print() or stdout).

    Why two logging levels:
        - INFO (console): epoch summaries, config — not overwhelming
        - DEBUG (file): everything, including timestamps — useful for
          post-mortem analysis without cluttering the terminal
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        tensorboard_dir: str | Path | None = None,
        resume: bool = False,
        log_filename: str | None = None,
    ):
        """
        Args:
            log_dir:
                Directory for text log files. Defaults to config.paths.LOGS_DIR.
                Created automatically if it doesn't exist.
                Each run produces one .log file inside this directory.

            tensorboard_dir:
                Directory for TensorBoard event files.
                Defaults to OUTPUTS_DIR/tensorboard/<timestamp>.
                Event files contain scalar curves, histograms, and model graph.
                View with: tensorboard --logdir <tensorboard_dir>

            resume:
                Set True when continuing from a checkpoint. Only changes the
                header message logged at startup — has no effect on file paths
                or the actual checkpoint loading (that's checkpoint.py's job).

            log_filename:
                Custom filename for the text log. If None, auto-generates:
                train_YYYYMMDD_HHMMSS.log
                The timestamp ensures each run gets a unique file — no
                overwriting even if you run multiple experiments back-to-back.
        """
        # ---- Resolve directories ----
        # Use project-configured paths when no override given.
        # log_dir defaults to LOGS_DIR (ROOT_DIR/logs/) from config/paths.py.
        if log_dir is None:
            log_dir = LOGS_DIR
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # tensorboard_dir defaults to OUTPUTS_DIR/tensorboard/<timestamp>.
        # OUTPUTS_DIR is ROOT_DIR/outputs/ from config/paths.py.
        # Timestamp subdirectory keeps experiments separated.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tensorboard_dir is None:
            tensorboard_dir = OUTPUTS_DIR / "tensorboard" / timestamp
        self.tensorboard_dir = Path(tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # ---- Log file path ----
        if log_filename is None:
            log_filename = f"train_{timestamp}.log"
        self.log_path = log_dir / log_filename

        # ============================================================
        # Python logging setup: two handlers, two levels
        # ============================================================

        # Each logger is identified by its name. Using the full log path
        # as the name guarantees uniqueness across runs (preventing handler
        # leakage if two TrainLogger instances exist simultaneously).
        self._logger = logging.getLogger(str(self.log_path))
        self._logger.setLevel(logging.DEBUG)  # Capture everything
        self._logger.handlers.clear()  # Defensive: avoid duplicate handlers

        # --- Console handler (stderr, INFO) ---
        # stderr is chosen because tqdm also uses stderr — they share the
        # stream without fighting over cursor position.
        # Formatter: message only (no timestamp — terminal already has context).
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(console_handler)

        # --- File handler (DEBUG) ---
        # Full detail: timestamp + level + message. Useful for comparing
        # runs or debugging silent failures that didn't show on console.
        file_handler = logging.FileHandler(str(self.log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(file_handler)

        # ============================================================
        # TensorBoard SummaryWriter
        # ============================================================
        # SummaryWriter is the main entry point for writing data to
        # TensorBoard event files. Each call to add_scalar / add_graph
        # appends an event to the file in log_dir.
        # The writer must be closed (self.close()) to flush remaining
        # buffered data — otherwise you may lose the last few events.
        self._writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        # ---- Startup header ----
        self._logger.info("=" * 60)
        if resume:
            self._logger.info("Resuming training from checkpoint")
        else:
            self._logger.info("New training run started")
        self._logger.info(f"Log file:    {self.log_path}")
        self._logger.info(f"TensorBoard: {self.tensorboard_dir}")

    # ================================================================
    # Console + file methods
    # ================================================================

    def logConfig(self, config: dict) -> None:
        """
        Log hyperparameters and training configuration.

        Writes to both console (INFO) and file (DEBUG). Called once at
        the start of training so the log file is self-documenting —
        you can always look back and see what settings produced a result.

        Args:
            config: Flat dict of parameter names to values, e.g.
                    {"batch_size": 64, "lr": 0.001, "epochs": 30}
        """
        self._logger.info("-" * 40)
        self._logger.info("Training config:")
        for key, value in config.items():
            self._logger.info(f"  {key}: {value}")
        self._logger.info("-" * 40)

    def logGraph(self, model: nn.Module, sample_input: torch.Tensor) -> None:
        """
        Write the model computation graph to TensorBoard.

        Uses torch.jit.trace under the hood to capture the forward pass
        as a static graph. This graph appears under the GRAPHS tab in
        TensorBoard and is useful for verifying the architecture visually.

        Args:
            model:
                The nn.Module to trace. Must be in eval mode for a clean
                trace (no dropout randomness, no batchnorm updates).
            sample_input:
                A dummy input matching the model's expected shape.
                For MNISTCNN: torch.randn(1, 1, 28, 28).
                The batch dimension can be any size; 1 is typical to
                keep the graph diagram uncluttered.
        """
        self._writer.add_graph(model, sample_input)
        self._logger.info("Model graph written to TensorBoard")

    def logEpoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float | None = None,
    ) -> None:
        """
        Log train/val metrics after each epoch.

        Called once per epoch. Writes to:
        - Console + file: single-line summary with all metrics
        - TensorBoard: one scalar point per metric, allowing loss/acc
          curves to be viewed over time

        Args:
            epoch:
                Current epoch number (1-indexed). Used as the x-axis
                step for TensorBoard scalar plots.

            train_loss:
                Average training loss over the full epoch. Lower is better.
                For CrossEntropyLoss, typical MNIST range: 1.5 → 0.01.

            train_acc:
                Training accuracy as a fraction in [0.0, 1.0].
                Should trend upward and approach 1.0.

            val_loss:
                Average validation loss. If val_loss starts increasing
                while train_loss still decreases, that's overfitting.

            val_acc:
                Validation accuracy as a fraction in [0.0, 1.0].
                The primary metric for model selection (best checkpoint).

            lr:
                Current learning rate (float). Logged because schedulers
                like ReduceLROnPlateau change lr mid-training. Pass None
                if no scheduler is used.
        """
        # Console / file line
        parts = [
            f"Epoch {epoch:>3d} |",
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} |",
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
        ]
        if lr is not None:
            parts.append(f"  lr={lr:.2e}")
        self._logger.info("".join(parts))

        # TensorBoard scalars — each call adds a data point at step=epoch
        # The tag names use "group/name" convention: TensorBoard groups
        # them under foldable sections in the UI (e.g., "Loss" folder
        # contains "train" and "val" plots overlayed).
        self._writer.add_scalar("Loss/train", train_loss, epoch)
        self._writer.add_scalar("Loss/val", val_loss, epoch)
        self._writer.add_scalar("Accuracy/train", train_acc, epoch)
        self._writer.add_scalar("Accuracy/val", val_acc, epoch)
        if lr is not None:
            self._writer.add_scalar("LR", lr, epoch)

    def logBest(self, metric_name: str, value: float, epoch: int) -> None:
        """
        Announce a new best metric value.

        Called by the training loop when a metric (typically val_acc)
        surpasses the previous best. Highlights the improvement in the
        log so you can scan for best-epoch lines without reading every line.

        Args:
            metric_name: Human-readable name, e.g. "val accuracy".
            value: The new best value.
            epoch: The epoch at which this best was achieved.
        """
        self._logger.info(f"  >> New best {metric_name}: {value:.4f} (epoch {epoch})")

    def logMessage(self, msg: str) -> None:
        """
        Log a generic INFO-level message.

        For one-off events: "Starting validation pass", "Saving checkpoint", etc.
        """
        self._logger.info(msg)

    def logWarning(self, msg: str) -> None:
        """
        Log a WARNING-level message.

        For non-fatal issues: "Checkpoint file not found, training from scratch",
        "GPU memory low, consider reducing batch size", etc.
        Warnings appear in both console (stderr, yellow in many terminals)
        and the log file.
        """
        self._logger.warning(msg)

    def logSummary(self, best_epoch: int, best_val_acc: float) -> None:
        """
        Log training completion summary.

        Called once at the end of training. Prints a clearly-delimited
        block with the key takeaway numbers so they're easy to find when
        scrolling back through a long log.

        Args:
            best_epoch: The epoch that produced the best validation accuracy.
            best_val_acc: The best validation accuracy achieved.
        """
        self._logger.info("=" * 60)
        self._logger.info("Training complete")
        self._logger.info(f"  Best epoch:        {best_epoch}")
        self._logger.info(f"  Best val accuracy: {best_val_acc:.4f}")
        self._logger.info(f"  Log file:          {self.log_path}")
        self._logger.info(f"  TensorBoard:       {self.tensorboard_dir}")
        self._logger.info("=" * 60)

    # ================================================================
    # TensorBoard-only methods
    # ================================================================

    def logScalar(self, tag: str, value: float, step: int) -> None:
        """
        Write an arbitrary scalar to TensorBoard.

        For custom metrics not covered by logEpoch: per-class accuracy,
        gradient norms, weight histograms, etc.

        Args:
            tag: Scalar tag name (e.g. "Accuracy/class_0", "Gradients/max").
                 Use "group/name" convention for organized TensorBoard UI.
            value: Scalar value at this step.
            step: Global step (usually epoch number).
        """
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """
        Close the TensorBoard writer and flush all buffered data.

        Must be called before the program exits. Without this:
        - Buffered TensorBoard events may be lost
        - The event file may be incomplete/corrupt
        - File handles remain open (resource leak)

        Safe to call multiple times (no-op after first close).
        """
        self._writer.close()
        self._logger.info("TensorBoard writer closed")
