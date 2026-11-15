"""Model for RainCheck: a scikit-learn pipeline (scaler + k-NN)."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(k: int) -> Pipeline:
    """Return an unfitted pipeline: scale the features, then k-NN with k neighbours.

    The scaler is part of the pipeline, so fit() learns its mean and stdev
    from the training data only.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
