"""BinML — a 6-class classifier for Nancy Grace Roman Space Telescope light curves.

Six classes: Flat, PSPL, NonPSPL (binary/planetary microlensing), PeriodicVar, LongPeriodVar,
Eruptive. Training labels follow an adopted synthetic detectability policy and, for partial
seasons, an intended Flat -> PSPL -> NonPSPL progression. This is not a guarantee against
premature alerts; see the evaluation documentation for the measured timing rates.

    import binml
    clf = binml.Classifier()
    r = clf.predict({"F146": (t, mag), "F087": (t2, mag2), "F213": (t3, mag3)})
    print(r.probabilities)

The earlier 3-class model is preserved at ``binml.legacy``.
"""
from .classifier import CLASS_NAMES, Classifier, Prediction
from .preprocess import Tokens, to_tokens, estimate_baseline

__version__ = "1.0.0"
__all__ = ["Classifier", "Prediction", "CLASS_NAMES", "Tokens", "to_tokens",
           "estimate_baseline", "__version__"]
