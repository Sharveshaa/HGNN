"""
Feature extraction modules for the HGNN Campaign Detection System.

Submodules:
    - text_features: Bio embeddings, topic embeddings, sentiment, urgency,
                     clickbait, toxicity, emotion vectors
    - user_features: Profile numeric + behavioral feature construction
    - normalization: FeatureNormalizer with per-type scaling strategies
"""

from .text_features import TextFeatureExtractor
from .user_features import UserFeatureExtractor
from .normalization import FeatureNormalizer

__all__ = [
    'TextFeatureExtractor',
    'UserFeatureExtractor',
    'FeatureNormalizer',
]
