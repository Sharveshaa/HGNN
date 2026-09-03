"""
Feature Normalization for the HGNN Campaign Detection System.

Implements a FeatureNormalizer with per-type scaling strategies:
    - Continuous numeric  → RobustScaler (median/IQR, outlier-resistant)
    - Counts              → log1p → StandardScaler
    - Embeddings          → L2 normalization (unit sphere)
    - Binary flags        → Pass-through
    - Bounded scores      → Min-max to [0, 1]

The normalizer follows a fit/transform pattern and can serialize
fitted scalers for inference consistency.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ScaleStrategy(Enum):
    """Scaling strategy for a feature group."""
    ROBUST = 'robust'         # median/IQR scaling (for continuous numeric)
    LOG_STANDARD = 'log_std'  # log1p → standard scaling (for counts)
    L2_NORM = 'l2_norm'       # L2 normalization (for embeddings)
    PASSTHROUGH = 'pass'      # no scaling (for binary flags)
    MINMAX = 'minmax'         # min-max to [0, 1] (for bounded scores)


@dataclass
class FeatureGroupSpec:
    """Specification for a group of features."""
    name: str
    start_idx: int
    end_idx: int  # exclusive
    strategy: ScaleStrategy

    @property
    def dim(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class FittedParams:
    """Fitted parameters for a feature group."""
    strategy: ScaleStrategy
    # RobustScaler params
    median: Optional[np.ndarray] = None
    iqr: Optional[np.ndarray] = None
    # StandardScaler params (after log1p)
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    # MinMax params
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────
# Default Feature Group Definitions
# ──────────────────────────────────────────────────────────────

# User features (32 dims)
USER_FEATURE_GROUPS = [
    FeatureGroupSpec('bio_embedding', 0, 16, ScaleStrategy.L2_NORM),
    FeatureGroupSpec('followers_count', 16, 17, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('following_count', 17, 18, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('account_age', 18, 19, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('post_count', 19, 20, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('ff_ratio', 20, 21, ScaleStrategy.ROBUST),
    FeatureGroupSpec('has_profile_pic', 21, 22, ScaleStrategy.PASSTHROUGH),
    FeatureGroupSpec('has_url', 22, 23, ScaleStrategy.PASSTHROUGH),
    FeatureGroupSpec('username_digit_ratio', 23, 24, ScaleStrategy.MINMAX),
    FeatureGroupSpec('fullname_digit_ratio', 24, 25, ScaleStrategy.MINMAX),
    FeatureGroupSpec('bio_length', 25, 26, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('posting_frequency', 26, 27, ScaleStrategy.ROBUST),
    FeatureGroupSpec('avg_sentiment', 27, 28, ScaleStrategy.MINMAX),
    FeatureGroupSpec('url_share_rate', 28, 29, ScaleStrategy.MINMAX),
    FeatureGroupSpec('urgency_mean', 29, 30, ScaleStrategy.MINMAX),
    FeatureGroupSpec('distinct_domains', 30, 31, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('age_normalized', 31, 32, ScaleStrategy.MINMAX),
]

# Post features (48 dims)
POST_FEATURE_GROUPS = [
    FeatureGroupSpec('text_embedding', 0, 16, ScaleStrategy.L2_NORM),
    FeatureGroupSpec('sentiment', 16, 17, ScaleStrategy.MINMAX),
    FeatureGroupSpec('urgency', 17, 18, ScaleStrategy.MINMAX),
    FeatureGroupSpec('clickbait', 18, 19, ScaleStrategy.MINMAX),
    FeatureGroupSpec('toxicity', 19, 20, ScaleStrategy.MINMAX),
    FeatureGroupSpec('emotion_vector', 20, 28, ScaleStrategy.MINMAX),
    FeatureGroupSpec('topic_embedding', 28, 44, ScaleStrategy.L2_NORM),
    FeatureGroupSpec('text_length', 44, 45, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('word_count', 45, 46, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('caps_ratio', 46, 47, ScaleStrategy.MINMAX),
    FeatureGroupSpec('excl_count', 47, 48, ScaleStrategy.LOG_STANDARD),
]

# URL features (14 dims)
URL_FEATURE_GROUPS = [
    FeatureGroupSpec('url_length', 0, 1, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('url_entropy', 1, 2, ScaleStrategy.ROBUST),
    FeatureGroupSpec('is_shortened', 2, 3, ScaleStrategy.PASSTHROUGH),
    FeatureGroupSpec('has_suspicious_tld', 3, 4, ScaleStrategy.PASSTHROUGH),
    FeatureGroupSpec('path_depth', 4, 5, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('subdomain_count', 5, 6, ScaleStrategy.LOG_STANDARD),
    FeatureGroupSpec('entropy_ratio', 6, 7, ScaleStrategy.ROBUST),
    FeatureGroupSpec('char_patterns', 7, 14, ScaleStrategy.MINMAX),
]

# Map node types to their feature groups
DEFAULT_FEATURE_GROUPS: Dict[str, List[FeatureGroupSpec]] = {
    'user': USER_FEATURE_GROUPS,
    'post': POST_FEATURE_GROUPS,
    'url': URL_FEATURE_GROUPS,
}


class FeatureNormalizer:
    """
    Normalizes feature matrices using per-group scaling strategies.

    Follows sklearn-style fit/transform API with serialization support.

    Usage:
        normalizer = FeatureNormalizer(feature_groups=USER_FEATURE_GROUPS)
        normalizer.fit(train_features)
        normalized_train = normalizer.transform(train_features)
        normalized_test = normalizer.transform(test_features)

        # Save/load for inference
        normalizer.save('normalizer.pkl')
        normalizer = FeatureNormalizer.load('normalizer.pkl')
    """

    def __init__(
        self,
        feature_groups: Optional[List[FeatureGroupSpec]] = None,
        node_type: Optional[str] = None,
        clip_value: float = 5.0,
    ):
        """
        Args:
            feature_groups: List of FeatureGroupSpec defining the scaling strategy
                            per feature slice. If None, uses default for node_type.
            node_type: Node type name to lookup default feature groups.
            clip_value: Clip normalized values to [-clip_value, clip_value]
                        (for robust/standard scaling only).
        """
        if feature_groups is not None:
            self.feature_groups = feature_groups
        elif node_type and node_type in DEFAULT_FEATURE_GROUPS:
            self.feature_groups = DEFAULT_FEATURE_GROUPS[node_type]
        else:
            self.feature_groups = []

        self.clip_value = clip_value
        self._fitted_params: Dict[str, FittedParams] = {}
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer parameters on training data.

        Args:
            X: Feature matrix of shape (N, D)
        """
        for group in self.feature_groups:
            data = X[:, group.start_idx:group.end_idx].astype(np.float32)
            params = FittedParams(strategy=group.strategy)

            if group.strategy == ScaleStrategy.ROBUST:
                params.median = np.median(data, axis=0)
                q75 = np.percentile(data, 75, axis=0)
                q25 = np.percentile(data, 25, axis=0)
                params.iqr = q75 - q25
                params.iqr = np.where(params.iqr == 0, 1.0, params.iqr)

            elif group.strategy == ScaleStrategy.LOG_STANDARD:
                log_data = np.log1p(np.abs(data))
                params.mean = np.mean(log_data, axis=0)
                params.std = np.std(log_data, axis=0)
                params.std = np.where(params.std == 0, 1.0, params.std)

            elif group.strategy == ScaleStrategy.MINMAX:
                params.min_val = np.min(data, axis=0)
                params.max_val = np.max(data, axis=0)
                range_val = params.max_val - params.min_val
                params.max_val = np.where(range_val == 0, params.min_val + 1.0, params.max_val)

            # L2_NORM and PASSTHROUGH don't need fitted params

            self._fitted_params[group.name] = params

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.

        Args:
            X: Feature matrix of shape (N, D)

        Returns:
            Normalized feature matrix of shape (N, D)
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform()")

        result = X.copy().astype(np.float32)

        for group in self.feature_groups:
            s, e = group.start_idx, group.end_idx
            data = result[:, s:e]
            params = self._fitted_params[group.name]

            if group.strategy == ScaleStrategy.ROBUST:
                normalized = (data - params.median) / params.iqr
                normalized = np.clip(normalized, -self.clip_value, self.clip_value)
                result[:, s:e] = normalized

            elif group.strategy == ScaleStrategy.LOG_STANDARD:
                log_data = np.log1p(np.abs(data))
                normalized = (log_data - params.mean) / params.std
                normalized = np.clip(normalized, -self.clip_value, self.clip_value)
                result[:, s:e] = normalized

            elif group.strategy == ScaleStrategy.L2_NORM:
                norms = np.linalg.norm(data, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                result[:, s:e] = data / norms

            elif group.strategy == ScaleStrategy.MINMAX:
                range_val = params.max_val - params.min_val
                range_val = np.where(range_val == 0, 1.0, range_val)
                result[:, s:e] = (data - params.min_val) / range_val

            elif group.strategy == ScaleStrategy.PASSTHROUGH:
                pass  # No transformation

        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def save(self, path: str):
        """Save fitted normalizer to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'feature_groups': self.feature_groups,
            'fitted_params': self._fitted_params,
            'clip_value': self.clip_value,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """Load fitted normalizer from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        normalizer = cls(
            feature_groups=state['feature_groups'],
            clip_value=state['clip_value'],
        )
        normalizer._fitted_params = state['fitted_params']
        normalizer._is_fitted = True
        return normalizer

    def describe(self) -> str:
        """Return a human-readable description of the normalization strategy."""
        lines = ["Feature Normalization Strategy:"]
        for group in self.feature_groups:
            lines.append(
                f"  {group.name:25s} [{group.start_idx}:{group.end_idx}] "
                f"→ {group.strategy.value}"
            )
        return "\n".join(lines)


class MultiNodeNormalizer:
    """
    Convenience wrapper that manages FeatureNormalizer instances
    for multiple node types.

    Usage:
        normalizer = MultiNodeNormalizer()
        normalizer.fit({'user': user_features, 'post': post_features})
        normalized = normalizer.transform({'user': user_features, 'post': post_features})
    """

    def __init__(self, clip_value: float = 5.0):
        self.clip_value = clip_value
        self._normalizers: Dict[str, FeatureNormalizer] = {}

    def fit(self, feature_dict: Dict[str, np.ndarray]) -> 'MultiNodeNormalizer':
        """Fit normalizers for all node types."""
        for node_type, features in feature_dict.items():
            if node_type in DEFAULT_FEATURE_GROUPS:
                normalizer = FeatureNormalizer(
                    node_type=node_type,
                    clip_value=self.clip_value,
                )
            else:
                # For unknown node types, apply robust scaling to all features
                groups = [
                    FeatureGroupSpec(
                        name=f'{node_type}_all',
                        start_idx=0,
                        end_idx=features.shape[1],
                        strategy=ScaleStrategy.ROBUST,
                    )
                ]
                normalizer = FeatureNormalizer(
                    feature_groups=groups,
                    clip_value=self.clip_value,
                )
            normalizer.fit(features)
            self._normalizers[node_type] = normalizer
        return self

    def transform(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform features for all node types."""
        result = {}
        for node_type, features in feature_dict.items():
            if node_type in self._normalizers:
                result[node_type] = self._normalizers[node_type].transform(features)
            else:
                result[node_type] = features  # pass through if not fitted
        return result

    def fit_transform(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(feature_dict).transform(feature_dict)

    def save(self, directory: str):
        """Save all normalizers to a directory."""
        os.makedirs(directory, exist_ok=True)
        for node_type, normalizer in self._normalizers.items():
            normalizer.save(os.path.join(directory, f'{node_type}_normalizer.pkl'))

    @classmethod
    def load(cls, directory: str) -> 'MultiNodeNormalizer':
        """Load all normalizers from a directory."""
        multi = cls()
        for fname in os.listdir(directory):
            if fname.endswith('_normalizer.pkl'):
                node_type = fname.replace('_normalizer.pkl', '')
                multi._normalizers[node_type] = FeatureNormalizer.load(
                    os.path.join(directory, fname)
                )
        return multi


if __name__ == '__main__':
    print("=== Feature Normalizer Demo ===\n")

    # Simulate user features (32 dims)
    np.random.seed(42)
    n = 100
    features = np.random.randn(n, 32).astype(np.float32)
    # Make some columns look like counts (positive, skewed)
    features[:, 16:20] = np.abs(features[:, 16:20]) * 1000
    # Make some columns binary
    features[:, 21:23] = (features[:, 21:23] > 0).astype(np.float32)

    normalizer = FeatureNormalizer(node_type='user')
    print(normalizer.describe())

    normalized = normalizer.fit_transform(features)

    print(f"\nOriginal stats:")
    print(f"  mean: {features.mean():.4f}, std: {features.std():.4f}")
    print(f"  min: {features.min():.4f}, max: {features.max():.4f}")
    print(f"\nNormalized stats:")
    print(f"  mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    print(f"  min: {normalized.min():.4f}, max: {normalized.max():.4f}")

    # Test embeddings are L2-normalized
    emb_norms = np.linalg.norm(normalized[:, :16], axis=1)
    print(f"\nEmbedding L2 norms (should be ~1.0): {emb_norms[:5]}")
