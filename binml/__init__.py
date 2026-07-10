"""
BinML -- deep-learning microlensing light-curve classifier.

A 3-class classifier (Flat / PSPL / Binary) for microlensing light curves. PSPL is the
"is this a microlensing event" proxy; Binary is the distinct anomalous (planetary/binary)
class. Trained on Roman Space Telescope cadence.

Quick start
-----------
    import binml
    clf = binml.Classifier()                       # fine-tuned model (CPU)
    t, m, e = binml.surveys.fetch_ogle_ews(2017, 482)   # a real OGLE event
    print(clf.predict(t, m, e))
"""
from .classifier import CLASS_NAMES, Classifier, Evolution, Prediction
from .preprocess import Preprocessed, preprocess, to_magnification
from .evaluate import (ClassificationReport, DetectabilityReport,
                       classification_report, detectability_curve, evaluate_dataset)
from . import surveys

__version__ = "0.2.0"
__all__ = [
    "Classifier", "Prediction", "Evolution", "CLASS_NAMES",
    "preprocess", "Preprocessed", "to_magnification", "surveys",
    "classification_report", "detectability_curve", "evaluate_dataset",
    "ClassificationReport", "DetectabilityReport",
    "plot_prediction", "plot_evolution",
]


def __getattr__(name):
    # lazy plotting import so `import binml` doesn't require matplotlib
    if name in ("plot_prediction", "plot_evolution"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module 'binml' has no attribute {name!r}")
