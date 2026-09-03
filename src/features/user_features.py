"""
User Feature Extraction for the HGNN Campaign Detection System.

Combines profile-level numeric features with bio embeddings and
behavioral features to produce a fixed-dimension user feature vector.

Feature breakdown (32 dimensions):
    - Bio embedding: 16 dims (from TextFeatureExtractor SVD)
    - Profile numeric: 10 dims
        - followers_count (log1p)
        - following_count (log1p)
        - account_age_days (log1p)
        - post_count (log1p)
        - followers_following_ratio
        - has_profile_pic
        - has_url
        - username_digit_ratio
        - fullname_digit_ratio
        - bio_length
    - Behavioral: 6 dims
        - posting_frequency
        - avg_post_sentiment
        - url_share_rate
        - urgency_mean
        - distinct_domains_shared
        - account_age_normalized
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.text_features import TextFeatureExtractor


class UserFeatureExtractor:
    """
    Extracts and assembles user-level feature vectors.

    Requires a fitted TextFeatureExtractor for bio embeddings.

    Usage:
        text_ext = TextFeatureExtractor(embedding_dim=16)
        text_ext.fit(all_bio_texts)

        user_ext = UserFeatureExtractor(text_extractor=text_ext)
        features = user_ext.extract(users_df, posts_df)
        # features shape: (num_users, 32)
    """

    BIO_DIM = 16
    NUMERIC_DIM = 10
    BEHAVIORAL_DIM = 6
    TOTAL_DIM = BIO_DIM + NUMERIC_DIM + BEHAVIORAL_DIM  # 32

    def __init__(self, text_extractor: Optional[TextFeatureExtractor] = None):
        self.text_extractor = text_extractor or TextFeatureExtractor(
            embedding_dim=self.BIO_DIM
        )

    def _extract_bio_embeddings(self, bios: List[str]) -> np.ndarray:
        """Extract bio text embeddings via SVD."""
        embeddings = self.text_extractor.get_embeddings(bios)
        # Ensure correct dim
        if embeddings.shape[1] < self.BIO_DIM:
            pad = np.zeros(
                (embeddings.shape[0], self.BIO_DIM - embeddings.shape[1]),
                dtype=np.float32,
            )
            embeddings = np.hstack([embeddings, pad])
        elif embeddings.shape[1] > self.BIO_DIM:
            embeddings = embeddings[:, :self.BIO_DIM]
        return embeddings

    def _extract_numeric_features(self, users_df: pd.DataFrame) -> np.ndarray:
        """
        Extract 10 numeric profile features.

        Handles missing columns gracefully by filling with defaults.
        """
        n = len(users_df)
        features = np.zeros((n, self.NUMERIC_DIM), dtype=np.float32)

        # Column mappings with defaults
        col_map = {
            0: ('followers_count', 0),
            1: ('following_count', 0),
            2: ('account_age_days', 0),
            3: ('post_count', 0),       # may come from posts_count or calculated
        }

        for idx, (col, default) in col_map.items():
            if col in users_df.columns:
                features[:, idx] = np.log1p(
                    users_df[col].fillna(default).values.astype(np.float32)
                )
            # Try alternative column names
            elif col == 'post_count' and 'posts_count' in users_df.columns:
                features[:, 3] = np.log1p(
                    users_df['posts_count'].fillna(default).values.astype(np.float32)
                )

        # followers/following ratio (idx=4)
        followers = users_df.get('followers_count', pd.Series(np.zeros(n))).fillna(0).values.astype(np.float32)
        following = users_df.get('following_count', pd.Series(np.ones(n))).fillna(1).values.astype(np.float32)
        following = np.where(following == 0, 1.0, following)  # avoid div by zero
        features[:, 4] = np.log1p(followers / following)

        # has_profile_pic (idx=5)
        if 'profile_pic' in users_df.columns:
            features[:, 5] = users_df['profile_pic'].fillna(0).values.astype(np.float32)
        elif 'default_profile_image' in users_df.columns:
            # Inverted: default_profile_image=1 means no custom pic
            features[:, 5] = (1 - users_df['default_profile_image'].fillna(0)).values.astype(np.float32)
        else:
            features[:, 5] = 1.0  # assume has pic if column missing

        # has_url (idx=6) - external URL in profile
        if 'external URL' in users_df.columns:
            features[:, 6] = users_df['external URL'].fillna(0).values.astype(np.float32)
        else:
            features[:, 6] = 0.0

        # username_digit_ratio (idx=7)
        if 'username_digit_ratio' in users_df.columns:
            features[:, 7] = users_df['username_digit_ratio'].fillna(0).values.astype(np.float32)
        elif 'nums/length username' in users_df.columns:
            features[:, 7] = users_df['nums/length username'].fillna(0).values.astype(np.float32)
        elif 'username' in users_df.columns:
            features[:, 7] = users_df['username'].apply(
                lambda u: sum(c.isdigit() for c in str(u)) / max(len(str(u)), 1)
            ).values.astype(np.float32)

        # fullname_digit_ratio (idx=8)
        if 'fullname_digit_ratio' in users_df.columns:
            features[:, 8] = users_df['fullname_digit_ratio'].fillna(0).values.astype(np.float32)
        elif 'nums/length fullname' in users_df.columns:
            features[:, 8] = users_df['nums/length fullname'].fillna(0).values.astype(np.float32)
        else:
            features[:, 8] = 0.0

        # bio_length (idx=9)
        if 'bio_length' in users_df.columns:
            features[:, 9] = np.log1p(
                users_df['bio_length'].fillna(0).values.astype(np.float32)
            )
        elif 'description length' in users_df.columns:
            features[:, 9] = np.log1p(
                users_df['description length'].fillna(0).values.astype(np.float32)
            )
        elif 'description' in users_df.columns:
            features[:, 9] = np.log1p(
                users_df['description'].fillna('').apply(len).values.astype(np.float32)
            )

        return features

    def _extract_behavioral_features(
        self,
        users_df: pd.DataFrame,
        posts_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Extract 6 behavioral features derived from user activity.

        Requires posts_df to compute content-based behavioral metrics.
        """
        n = len(users_df)
        features = np.zeros((n, self.BEHAVIORAL_DIM), dtype=np.float32)

        if posts_df is None:
            return features

        # Build user_id index
        user_id_col = 'user_id'
        if user_id_col not in users_df.columns:
            return features

        user_ids = users_df[user_id_col].tolist()

        # Group posts by user
        if user_id_col in posts_df.columns:
            user_posts = posts_df.groupby(user_id_col)
        else:
            return features

        for i, uid in enumerate(user_ids):
            if uid not in user_posts.groups:
                continue

            user_post_data = user_posts.get_group(uid)

            # posting_frequency (idx=0): posts per day of account age
            account_age = users_df.iloc[i].get('account_age_days', 1)
            if account_age and account_age > 0:
                features[i, 0] = np.log1p(len(user_post_data) / account_age)

            # avg_post_sentiment (idx=1)
            if 'text' in user_post_data.columns:
                from features.text_features import _compute_sentiment
                sentiments = [
                    _compute_sentiment(str(t))
                    for t in user_post_data['text'].fillna('')
                ]
                features[i, 1] = np.mean(sentiments) if sentiments else 0.0

            # url_share_rate (idx=2): fraction of posts containing URLs
            url_col = 'url' if 'url' in user_post_data.columns else 'URL'
            if url_col in user_post_data.columns:
                has_url = user_post_data[url_col].notna().sum()
                features[i, 2] = has_url / max(len(user_post_data), 1)

            # urgency_mean (idx=3): average urgency across posts
            if 'text' in user_post_data.columns:
                from features.text_features import URGENCY_KEYWORDS
                urgencies = []
                for text in user_post_data['text'].fillna(''):
                    text_lower = str(text).lower()
                    matches = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
                    urgencies.append(min(1.0, matches / 3.0))
                features[i, 3] = np.mean(urgencies) if urgencies else 0.0

            # distinct_domains_shared (idx=4)
            if url_col in user_post_data.columns:
                urls = user_post_data[url_col].dropna().tolist()
                try:
                    import urllib.parse
                    domains = set()
                    for u in urls:
                        parsed = urllib.parse.urlparse(str(u))
                        if parsed.netloc:
                            domains.add(parsed.netloc)
                    features[i, 4] = np.log1p(len(domains))
                except Exception:
                    features[i, 4] = 0.0

            # account_age_normalized (idx=5): normalized 0-1 where older = higher
            if account_age:
                features[i, 5] = min(1.0, account_age / 365.0)  # cap at 1 year

        return features

    def extract(
        self,
        users_df: pd.DataFrame,
        posts_df: Optional[pd.DataFrame] = None,
        bio_column: str = 'description',
    ) -> np.ndarray:
        """
        Extract complete user feature vectors.

        Args:
            users_df: DataFrame with user profile data
            posts_df: Optional DataFrame with post data for behavioral features
            bio_column: Column name containing bio/description text

        Returns:
            np.ndarray of shape (num_users, 32)
        """
        n = len(users_df)

        # Bio embeddings (16 dims)
        if bio_column in users_df.columns:
            bios = users_df[bio_column].fillna('').tolist()
        else:
            bios = [''] * n
        bio_emb = self._extract_bio_embeddings(bios)

        # Profile numeric features (10 dims)
        numeric = self._extract_numeric_features(users_df)

        # Behavioral features (6 dims)
        behavioral = self._extract_behavioral_features(users_df, posts_df)

        # Concatenate: [bio_emb | numeric | behavioral]
        full_features = np.hstack([bio_emb, numeric, behavioral])
        assert full_features.shape == (n, self.TOTAL_DIM), \
            f"Expected shape ({n}, {self.TOTAL_DIM}), got {full_features.shape}"

        return full_features


if __name__ == '__main__':
    # Demo with synthetic data
    print("=== User Feature Extractor Demo ===\n")

    users_df = pd.DataFrame({
        'user_id': [f'U{i:05d}' for i in range(5)],
        'username': ['alice', 'b0t_123', 'charlie', 'david99', 'eve'],
        'followers_count': [1500, 10, 800, 50, 2000],
        'following_count': [200, 5000, 300, 3000, 150],
        'account_age_days': [500, 5, 300, 10, 1000],
        'description': [
            'Software engineer, coffee lover',
            '',
            'Travel blogger exploring the world',
            'Check out my deals!',
            'Mom, teacher, and avid reader',
        ],
        'is_fake': [0, 1, 0, 1, 0],
    })

    posts_df = pd.DataFrame({
        'post_id': [f'P{i:05d}' for i in range(8)],
        'user_id': ['U00000', 'U00001', 'U00001', 'U00002',
                    'U00003', 'U00003', 'U00004', 'U00004'],
        'text': [
            'Great day at the park!',
            'URGENT: Verify your account NOW! http://evil.com',
            'Click here to claim your FREE prize! http://scam.xyz',
            'Beautiful sunset over the mountains',
            'Act immediately! Your account will be suspended http://phish.tk',
            'Warning! Unauthorized access detected http://fake.top',
            'Just finished a wonderful book',
            'Loving the new recipe I found',
        ],
        'url': [None, 'http://evil.com', 'http://scam.xyz', None,
                'http://phish.tk', 'http://fake.top', None, None],
    })

    # Fit text extractor on all bios
    bios = users_df['description'].fillna('').tolist()
    text_ext = TextFeatureExtractor(embedding_dim=16)
    text_ext.fit(bios + posts_df['text'].fillna('').tolist())

    # Extract features
    user_ext = UserFeatureExtractor(text_extractor=text_ext)
    features = user_ext.extract(users_df, posts_df)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Expected: ({len(users_df)}, {UserFeatureExtractor.TOTAL_DIM})")
    print(f"\nFeature breakdown:")
    print(f"  Bio embeddings:  [:, 0:16]  → {features[:, :16].shape}")
    print(f"  Profile numeric: [:, 16:26] → {features[:, 16:26].shape}")
    print(f"  Behavioral:      [:, 26:32] → {features[:, 26:32].shape}")

    for i, row in users_df.iterrows():
        label = "FAKE" if row['is_fake'] else "REAL"
        print(f"\n  [{label}] {row['username']}")
        print(f"    numeric:    {features[i, 16:26]}")
        print(f"    behavioral: {features[i, 26:32]}")
