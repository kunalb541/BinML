"""BinML legacy — the original 3-class (Flat / PSPL / Binary) CNN-GRU classifier.

Retained for reproducibility and provenance. The current 6-class model is the top-level
``binml.Classifier``. See docs/legacy_3class.md for how this model was built.

    from binml.legacy import Classifier as LegacyClassifier
"""
from .classifier import CLASS_NAMES, Classifier, Evolution, Prediction
from .preprocess import Preprocessed, preprocess, to_magnification
from .evaluate import (classification_report, detectability_curve, evaluate_dataset)
from . import surveys
__all__ = ["Classifier", "Prediction", "Evolution", "CLASS_NAMES", "preprocess",
           "to_magnification", "Preprocessed", "surveys",
           "classification_report", "detectability_curve", "evaluate_dataset"]
