"""
ClusterSimilarity Transformer
-----------------------------
Creates RBF similarity features from geographic coordinates based on KMeans clusters.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
from logger_module import Logger

logger = Logger().get_logger()


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Generate cluster similarity features using KMeans centers and RBF kernel."""

    def __init__(
        self,
        n_clusters: int = 10,
        gamma: float = 1.0,
        random_state: int | None = None,
        columns: list[str] | None = None,
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        if self.columns is None:
            raise ValueError("ClusterSimilarity requires 'columns' to be set.")

        if not set(self.columns).issubset(X.columns):
            missing = list(set(self.columns) - set(X.columns))
            raise KeyError(f"Missing required columns for clustering: {missing}")

        X_fit = X[self.columns]

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X_fit, sample_weight=sample_weight)

        logger.info(
            f"ClusterSimilarity fitted with {self.n_clusters} clusters, gamma={self.gamma}"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_trans = X[self.columns]
        features = rbf_kernel(X_trans, self.kmeans_.cluster_centers_, gamma=self.gamma)

        return pd.DataFrame(
            features,
            index=X.index,
            columns=self.get_feature_names_out(),
        )

    def get_feature_names_out(self) -> list[str]:
        return [f"cluster_sim{i}" for i in range(self.n_clusters)]
