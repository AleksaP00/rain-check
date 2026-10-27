"""Models for RainCheck: StandardScaler and KNNClassifier."""

import statistics


class StandardScaler:
    """Scale each column to mean 0 and standard deviation 1."""

    def __init__(self) -> None:
        """Create an unfitted scaler with empty means and stdevs."""

        self.means = []
        self.stdevs = []

    def fit(self, X: list[list[float]]) -> None:
        """Compute the mean and standard deviation of each column of X."""

        self.means = [statistics.mean(column) for column in zip(*X)]
        self.stdevs = [statistics.pstdev(column) or 1.0 for column in zip(*X)]

    def transform(self, X:list[list[float]]) -> list[list[float]]:
        """Return X with every value scaled as (x - mean) / stdev."""

        X_scaled = []
        for row in X:
            X_scaled.append([(row[i] - self.means[i]) / self.stdevs[i] for i in range(len(row))])

        return X_scaled

    def to_dict(self) -> dict:
        """Return the fitted state as a plain dict, ready for JSON."""
        return {"means": self.means, "stdevs": self.stdevs}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        """Build a fitted scaler from a dict made by to_dict."""
        s = cls()
        s.means = d["means"]
        s.stdevs = d["stdevs"]
        return s


class KNNClassifier:
    """k-nearest-neighbours classifier for binary labels (0 or 1)."""

    def __init__(self, k: int = 15) -> None:
        """Create a classifier that votes among k neighbours."""
        self.k = k

    @property
    def k(self) -> int:
        """Number of neighbours that vote."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """Set k, which must be a positive odd integer."""
        if not isinstance(value, int) or value <= 0 or value % 2 == 0:
            raise ValueError(f"k must be a positive odd integer, got {value!r}")
        self._k = value


def main() -> None:
    """Run the model module."""
    pass


if __name__ == "__main__":
    main()
