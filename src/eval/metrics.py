"""
src/eval/metrics.py

Evaluation metrics for MNIST CNN classification.

Functions:
    gatherPredictions       — run model over a DataLoader, collect outputs
    computeConfusionMatrix  — build confusion matrix from labels + predictions
    classificationReport    — per-class precision / recall / F1 / support
    evaluateModel           — one-shot: loss + accuracy + confusion matrix + report

Usage:
    from src.eval.metrics import evaluateModel, gatherPredictions

    result = evaluateModel(model, test_loader, criterion, device)
    print(result["accuracy"])
    print(result["confusion_matrix"])
    print(result["report"])
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================================================================
# Core functions
# ================================================================


@torch.no_grad()
def gatherPredictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run a full forward pass over the DataLoader and collect all outputs.

    This is the foundational helper for all metrics functions. It runs
    the model once, gathering every label, predicted class, and softmax
    probability. The caller can then compute whatever statistics they
    need without re-running the model.

    The function is decorated with @torch.no_grad() so it never builds
    a computation graph — it's read-only evaluation.

    Args:
        model:
            The nn.Module to evaluate. Will be temporarily set to eval()
            mode inside this function (and restored afterwards).

        dataloader:
            DataLoader yielding (images, labels) batches. Should have
            shuffle=False for deterministic results.

        device:
            Torch device string: "cuda" or "cpu".

    Returns:
        (all_labels, all_predictions, all_probabilities):
            all_labels:         (N,) int64 Tensor — ground truth class 0-9
            all_predictions:    (N,) int64 Tensor — predicted class 0-9
            all_probabilities:  (N, 10) float32 Tensor — softmax probabilities
                                for every class (rows sum to 1). Useful for
                                confidence analysis and top-k metrics.
    """
    # Save original mode so we can restore it. This makes the function
    # non-destructive — the caller doesn't need to remember to re-set
    # train/eval mode after calling us.
    was_training = model.training
    model.eval()

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # (B, 10) raw logits

        # softmax converts logits → probabilities: p_i = exp(z_i) / Σ exp(z_j)
        # dim=1: normalize across the class dimension for each sample
        probabilities = logits.softmax(dim=1)  # (B, 10)

        # argmax picks the class with the highest probability
        _, predictions = logits.max(dim=1)  # (B,)

        # Move to CPU so we can concatenate without GPU memory pressure.
        # The final tensors can be large (10k+ samples) but that's fine on CPU.
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())

    # Restore the model's original mode
    model.train(was_training)

    # Concatenate all batches along dim=0 (the sample dimension)
    return (
        torch.cat(all_labels),  # (N,)
        torch.cat(all_predictions),  # (N,)
        torch.cat(all_probabilities),  # (N, 10)
    )


def computeConfusionMatrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Build a confusion matrix from ground-truth labels and predictions.

    Confusion matrix C has shape (num_classes, num_classes):
        C[i, j] = number of samples with true class i predicted as class j

    Diagonal entries C[i, i] are correct predictions for class i.
    Off-diagonal entries C[i, j] (i ≠ j) are confusions: true class i
    misclassified as class j.

    This implementation uses torch.histogramdd for efficiency with large N.
    No for-loops over classes or samples.

    Args:
        labels:      (N,) int64 Tensor — ground truth class indices 0..num_classes-1
        predictions: (N,) int64 Tensor — predicted class indices 0..num_classes-1
        num_classes: Total number of classes (10 for MNIST digits 0-9)

    Returns:
        (num_classes, num_classes) int64 Tensor — confusion matrix
    """
    # Stack labels and predictions into (N, 2) so each row is a (true, pred)
    # pair, then histogramdd counts how many of each pair exist.
    # bins=num_classes: one bin per class value (0 through 9)
    # range=[0, num_classes-1]: ensures bins align with class indices
    stacked = torch.stack([labels, predictions], dim=1)  # (N, 2)
    confusion_matrix = torch.histogramdd(
        stacked.float(),
        bins=num_classes,
        range=[0, num_classes - 1, 0, num_classes - 1],
    ).histogram.long()

    return confusion_matrix  # (num_classes, num_classes)


def classificationReport(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str] | None = None,
    num_classes: int = 10,
) -> dict:
    """
    Compute per-class precision, recall, F1-score, and support count.

    For each class c (where c ∈ {0, ..., num_classes-1}):

        TP_c = C[c, c]                              — correctly predicted as c
        FP_c = Σ_{i≠c} C[i, c]                      — wrongly predicted as c
        FN_c = Σ_{j≠c} C[c, j]                      — true c predicted as something else

        Precision_c = TP_c / (TP_c + FP_c)           — of all "c" predictions, how many correct?
        Recall_c    = TP_c / (TP_c + FN_c)           — of all true c samples, how many found?
        F1_c        = 2 × P_c × R_c / (P_c + R_c)   — harmonic mean of precision and recall
        Support_c   = Σ_j C[c, j]                    — number of true samples of class c

    Handles the division-by-zero edge case: if a class never appears in
    predictions or ground truth, its precision/recall/F1 are set to 0.0.

    Args:
        labels:      (N,) int64 Tensor — ground truth class indices
        predictions: (N,) int64 Tensor — predicted class indices
        class_names: Optional human-readable names, e.g. ["0","1",...,"9"].
                     If None, defaults to str(i) for each class.
        num_classes: Total number of classes.

    Returns:
        report: dict with keys:
            "per_class": list of dicts, each containing:
                {"class": str, "precision": float, "recall": float,
                 "f1": float, "support": int}
            "macro_avg": dict — macro-averaged precision/recall/f1
            "accuracy": float — overall accuracy (same as micro-avg for classification)
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    confusion_matrix = computeConfusionMatrix(labels, predictions, num_classes)

    # Per-class sums
    # confusion_matrix.sum(dim=1): row sums → total true samples per class (support)
    # confusion_matrix.sum(dim=0): column sums → total predictions per class
    support_per_class = confusion_matrix.sum(dim=1)  # (num_classes,) — true samples
    predictions_per_class = confusion_matrix.sum(
        dim=0
    )  # (num_classes,) — predicted samples
    true_positives_per_class = confusion_matrix.diagonal()  # (num_classes,) — correct

    per_class: list[dict] = []
    for class_index in range(num_classes):
        true_positives = true_positives_per_class[class_index].item()
        false_positives = (predictions_per_class[class_index] - true_positives).item()
        false_negatives = (support_per_class[class_index] - true_positives).item()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append(
            {
                "class": class_names[class_index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support_per_class[class_index].item(),
            }
        )

    # Macro average: average each metric equally across all classes,
    # regardless of how many samples each class has. This treats every
    # digit (0-9) as equally important, even if the test set is imbalanced.
    macro_precision = sum(entry["precision"] for entry in per_class) / num_classes
    macro_recall = sum(entry["recall"] for entry in per_class) / num_classes
    macro_f1 = sum(entry["f1"] for entry in per_class) / num_classes

    # Overall accuracy: total correct / total samples
    accuracy = true_positives_per_class.sum().item() / labels.size(0)

    return {
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "accuracy": accuracy,
    }


# ================================================================
# One-shot evaluation
# ================================================================


@torch.no_grad()
def evaluateModel(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None = None,
    device: str = "cuda",
    class_names: list[str] | None = None,
    num_classes: int = 10,
) -> dict:
    """
    Run a full evaluation pass and return all metrics in one dict.

    This is the main entry point for evaluation. It runs one forward
    pass over the entire DataLoader, computes loss (if criterion given),
    and returns accuracy, confusion matrix, and a full per-class report.

    Typical usage:
        result = evaluateModel(model, test_loader, criterion, device)
        print(f"Test accuracy: {result['accuracy']:.4f}")
        print(result["confusion_matrix"])

    Args:
        model:
            The nn.Module to evaluate. Will be set to eval() temporarily.

        dataloader:
            DataLoader for the evaluation set. Should have shuffle=False.

        criterion:
            Optional loss function. If provided, the average loss over
            the full dataset is computed and included in the result.
            Pass None to skip loss computation (faster).

        device:
            Torch device: "cuda" or "cpu".

        class_names:
            Human-readable class names for the report, e.g. ["0","1",...,"9"].
            Defaults to str(i) for each digit.

        num_classes:
            Number of classes (10 for MNIST).

    Returns:
        result: dict with keys:
            "loss"              — float | None, average loss
            "accuracy"          — float, overall accuracy in [0.0, 1.0]
            "confusion_matrix"  — (num_classes, num_classes) int64 Tensor
            "report"            — dict from classificationReport()
            "num_samples"       — int, total number of samples evaluated
    """
    was_training = model.training
    model.eval()

    running_loss = 0.0
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

        _, predictions = logits.max(dim=1)

        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())

    model.train(was_training)

    # Concatenate into single tensors
    labels_tensor = torch.cat(all_labels)  # (N,)
    predictions_tensor = torch.cat(all_predictions)  # (N,)

    total_samples = labels_tensor.size(0)

    # Loss
    average_loss = (running_loss / total_samples) if criterion is not None else None

    # Accuracy
    correct_count = predictions_tensor.eq(labels_tensor).sum().item()
    accuracy = correct_count / total_samples

    # Confusion matrix
    confusion_matrix = computeConfusionMatrix(
        labels_tensor, predictions_tensor, num_classes
    )

    # Per-class report
    report = classificationReport(
        labels_tensor, predictions_tensor, class_names, num_classes
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "report": report,
        "num_samples": total_samples,
    }


# ================================================================
# Formatting helpers
# ================================================================


def formatReport(report: dict) -> str:
    """
    Pretty-print a classification report dict as a formatted string.

    Produces output similar to sklearn's classification_report:

        Class     Precision  Recall    F1        Support
        0         0.9900     0.9850    0.9875     980
        1         0.9950     0.9920    0.9935    1135
        ...
        Macro Avg 0.9870     0.9850    0.9860   10000

    Args:
        report: The dict returned by classificationReport() or
                evaluateModel()["report"].

    Returns:
        Multi-line formatted string.
    """
    lines: list[str] = []
    lines.append(
        f"{'Class':<10s} {'Precision':>10s} {'Recall':>10s} "
        f"{'F1':>10s} {'Support':>10s}"
    )
    lines.append("-" * 50)

    for entry in report["per_class"]:
        lines.append(
            f"{entry['class']:<10s} "
            f"{entry['precision']:>10.4f} "
            f"{entry['recall']:>10.4f} "
            f"{entry['f1']:>10.4f} "
            f"{entry['support']:>10d}"
        )

    lines.append("-" * 50)
    macro = report["macro_avg"]
    lines.append(
        f"{'Macro Avg':<10s} "
        f"{macro['precision']:>10.4f} "
        f"{macro['recall']:>10.4f} "
        f"{macro['f1']:>10.4f} "
        f"{'':>10s}"
    )

    if "accuracy" in report:
        lines.append(f"\nOverall Accuracy: {report['accuracy']:.4f}")

    return "\n".join(lines)
