"""
评估模块

提供模型评估、指标计算和结果可视化.

用法:
    from cnnlib.evaluation import Evaluator
    from cnnlib.evaluation.metrics import computeAllMetrics
    from cnnlib.evaluation.visualize import plotConfusionMatrix
"""

from cnnlib.evaluation.evaluator import Evaluator
from cnnlib.evaluation.metrics import (
    computeAccuracy,
    computeAllMetrics,
    computeConfusionMatrix,
    computePerClassAccuracy,
    computePrecisionRecallF1,
    computeTopKAccuracy,
)
from cnnlib.evaluation.visualize import (
    generateAllCharts,
    plotConfusionMatrix,
    plotPerClassAccuracy,
    plotPredictions,
    plotTrainingHistory,
)

__all__ = [
    # Evaluator
    "Evaluator",
    # Metrics
    "computeAccuracy",
    "computeTopKAccuracy",
    "computeAllMetrics",
    "computeConfusionMatrix",
    "computePerClassAccuracy",
    "computePrecisionRecallF1",
    # Visualization
    "generateAllCharts",
    "plotConfusionMatrix",
    "plotPredictions",
    "plotTrainingHistory",
    "plotPerClassAccuracy",
]
