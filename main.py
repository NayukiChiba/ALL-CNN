"""
main.py — MNIST-CNN entry point

This is the only executable entry for the entire project. It does NOT define
any CLI arguments itself; all argument definitions live in config/settings.py.

Responsibility:
  1. Call getSettings() to parse CLI arguments into a Namespace.
  2. Set global random seeds for reproducibility.
  3. Dispatch to train / eval / infer subcommand based on args.command.

Usage:
    python main.py [global args] <subcommand> [subcommand args]
    python main.py --device cuda --seed 42 train --epochs 50
    python main.py eval --checkpoint checkpoints/best_model.pth
    python main.py infer --image test.png --checkpoint checkpoints/best_model.pth
"""

import argparse
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from config.settings import getSettings

# ============================================================================
# Subcommand handlers
# ============================================================================
# Each handler receives the full argparse.Namespace and calls into the
# corresponding src/ module. They are intentionally thin — all heavy logic
# lives in src/.
# ============================================================================


def runTrain(args: argparse.Namespace) -> None:
    """Train subcommand — delegates to scripts/train.py."""
    from scripts.train import trainRun

    trainRun(args)


def runEval(args: argparse.Namespace) -> None:
    """Eval subcommand — delegates to scripts/eval.py."""
    from scripts.eval import evalRun

    evalRun(args)


def runInference(args: argparse.Namespace) -> None:
    """Inference subcommand — delegates to scripts/infer.py."""
    from scripts.infer import inferRun

    inferRun(args)


# ============================================================================
# Main entry point
# ============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for the MNIST-CNN project.

    Flow:
      1. Parse CLI arguments via config.settings.getSettings().
      2. If no subcommand is given, print help and exit.
      3. Set global random seeds (Python, NumPy, PyTorch) for reproducibility.
      4. Dispatch to the appropriate handler via a dictionary lookup.

    Args:
        argv: Optional list of CLI argument strings. When None (production),
              argparse reads from sys.argv automatically. When provided
              (testing), it overrides sys.argv — useful for simulating
              command-line invocations in tests.

              Example values:
                None                              → read sys.argv
                ["train", "--epochs", "5"]        → simulate training
                ["eval", "--checkpoint", "x.pth"] → simulate eval

    Why argv instead of directly reading sys.argv?
      - Testability: you can call main(["train", "--epochs", "1"]) in a test
        without subprocess or monkey-patching sys.argv.
      - The None default triggers argparse's built-in sys.argv[1:] behavior
        in production, so normal usage is unaffected.
    """
    # Step 1: Parse all CLI arguments.
    # getSettings(None) → argparse reads sys.argv
    # getSettings(list) → argparse uses the provided list (for testing)
    args: argparse.Namespace = getSettings(argv)

    # Step 2: No subcommand given (e.g., user ran "python main.py" with no args).
    # argparse sets dest="command" to None when no subparser is matched.
    if args.command is None:
        # Lazily import buildParser only when needed (small optimization).
        from config.settings import buildParser

        buildParser().print_help()
        return

    # Step 3: Seed everything for reproducibility.
    # Even if different subcommands don't all use every source of randomness,
    # setting all three upfront is cheap and prevents accidental non-determinism.
    random.seed(args.seed)  # Python stdlib random
    np.random.seed(args.seed)  # NumPy random
    torch.manual_seed(args.seed)  # PyTorch CPU random
    # NOTE: torch.cuda.manual_seed_all(args.seed) should be added if using CUDA.

    # Step 4: Dispatch to subcommand handler.
    # Using a dict instead of if/elif chain:
    #   - Adding a new subcommand only requires a new entry in the dict.
    #   - The key (args.command) comes from argparse subparser dest="command".
    dispatch: Dict[str, Callable[[argparse.Namespace], None]] = {
        "train": runTrain,
        "eval": runEval,
        "infer": runInference,
    }
    dispatch[args.command](args)
    # Equivalent expanded form:
    #   if args.command == "train":
    #       runTrain(args)
    #   elif args.command == "eval":
    #       runEval(args)
    #   elif args.command == "infer":
    #       runInference(args)


# ============================================================================
# Script guard
# ============================================================================

if __name__ == "__main__":
    # When run as:  python main.py train --epochs 50
    # sys.argv = ["main.py", "train", "--epochs", "50"]
    # main() calls getSettings(None), which internally reads sys.argv[1:].
    main()
