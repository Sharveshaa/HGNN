"""
Feature Building Orchestrator for the HGNN Campaign Detection System.

Runs all feature extractors on raw data and produces feature matrices
ready for graph construction:
    1. User features (32 dims): bio embeddings + numeric + behavioral
    2. Post features (48 dims): text embeddings + NLP scores + topic + numeric
    3. URL features (14 dims): structural + character patterns
    4. Saves fitted extractors for inference consistency
"""

import os
import sys
import numpy as np
import pandas as pd
import re
import math

# Ensure imports work from any CWD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.text_features import TextFeatureExtractor
from features.user_features import UserFeatureExtractor
from features.normalization import FeatureNormalizer, MultiNodeNormalizer


def shannon_entropy(data: str) -> float:
    """Compute Shannon entropy of a string."""
    if not data:
        return 0.0
    freq = {}
    for c in data:
        freq[c] = freq.get(c, 0) + 1
    length = len(data)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in freq.values())


def build_url_features(urls_df: pd.DataFrame) -> np.ndarray:
    """
    Build 14-dimensional URL feature vectors.

    Features:
        0:  url_length (log1p)
        1:  url_entropy
        2:  is_shortened (binary)
        3:  has_suspicious_tld (binary)
        4:  path_depth
        5:  subdomain_count
        6:  entropy_ratio (entropy / length)
        7:  has_at_symbol
        8:  has_double_slash_redirect
        9:  num_dots
        10: num_hyphens
        11: num_digits_ratio
        12: has_ip_in_url
        13: query_length
    """
    import urllib.parse

    url_col = 'URL' if 'URL' in urls_df.columns else 'url'
    n = len(urls_df)
    features = np.zeros((n, 14), dtype=np.float32)

    shorteners = {'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly',
                  'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc', 'rb.gy'}
    suspicious_tlds = {'xyz', 'top', 'club', 'online', 'site', 'vip',
                       'click', 'info', 'tk', 'ml', 'ga', 'cf', 'gq'}

    for i, row in urls_df.iterrows():
        url = str(row.get(url_col, ''))
        if not url or url == 'nan':
            continue

        idx = urls_df.index.get_loc(i)

        # 0: url_length
        features[idx, 0] = np.log1p(len(url))

        # 1: url_entropy
        entropy = shannon_entropy(url)
        features[idx, 1] = entropy

        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
        except Exception:
            domain = ''

        # 2: is_shortened
        for s in shorteners:
            if s in domain:
                features[idx, 2] = 1.0
                break

        # 3: has_suspicious_tld
        parts = domain.rsplit('.', 1)
        if len(parts) > 1 and parts[-1] in suspicious_tlds:
            features[idx, 3] = 1.0

        # 4: path_depth
        try:
            path_parts = [p for p in parsed.path.split('/') if p]
            features[idx, 4] = len(path_parts)
        except Exception:
            pass

        # 5: subdomain_count
        domain_parts = domain.split('.')
        features[idx, 5] = max(0, len(domain_parts) - 2)

        # 6: entropy_ratio
        if len(url) > 0:
            features[idx, 6] = entropy / len(url)

        # 7: has_at_symbol
        features[idx, 7] = 1.0 if '@' in url else 0.0

        # 8: has_double_slash_redirect
        if url.count('//') > 1:
            features[idx, 8] = 1.0

        # 9: num_dots
        features[idx, 9] = url.count('.')

        # 10: num_hyphens
        features[idx, 10] = url.count('-')

        # 11: num_digits_ratio
        num_digits = sum(c.isdigit() for c in url)
        features[idx, 11] = num_digits / max(len(url), 1)

        # 12: has_ip_in_url
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            features[idx, 12] = 1.0

        # 13: query_length
        try:
            features[idx, 13] = len(parsed.query)
        except Exception:
            pass

    return features


def build_post_features(
    posts_df: pd.DataFrame,
    text_extractor: TextFeatureExtractor,
) -> np.ndarray:
    """
    Build 48-dimensional post feature vectors.

    Features:
        0-15:  text embedding (SVD, 16 dims)
        16:    sentiment
        17:    urgency
        18:    clickbait
        19:    toxicity
        20-27: emotion vector (8 dims)
        28-43: topic embedding (NMF, 16 dims)
        44:    text_length (log1p)
        45:    word_count (log1p)
        46:    caps_ratio
        47:    excl_count (log1p)
    """
    texts = posts_df['text'].fillna('').tolist()
    n = len(texts)

    # Extract NLP features
    nlp_features = text_extractor.transform(texts)

    # Numeric features
    text_lengths = np.array([len(t) for t in texts], dtype=np.float32)
    word_counts = np.array([len(t.split()) for t in texts], dtype=np.float32)
    caps_ratios = np.array([
        sum(c.isupper() for c in t) / max(len(t), 1)
        for t in texts
    ], dtype=np.float32)
    excl_counts = np.array([t.count('!') for t in texts], dtype=np.float32)

    # Concatenate all features: 16 + 1 + 1 + 1 + 1 + 8 + 16 + 4 = 48
    features = np.hstack([
        nlp_features['embeddings'],     # 0-15:  text embedding
        nlp_features['sentiment'],       # 16:    sentiment
        nlp_features['urgency'],         # 17:    urgency
        nlp_features['clickbait'],       # 18:    clickbait
        nlp_features['toxicity'],        # 19:    toxicity
        nlp_features['emotion'],         # 20-27: emotion vector
        nlp_features['topics'],          # 28-43: topic embedding
        np.log1p(text_lengths).reshape(-1, 1),   # 44: text_length
        np.log1p(word_counts).reshape(-1, 1),    # 45: word_count
        caps_ratios.reshape(-1, 1),              # 46: caps_ratio
        np.log1p(excl_counts).reshape(-1, 1),    # 47: excl_count
    ])

    assert features.shape == (n, 48), f"Expected shape ({n}, 48), got {features.shape}"
    return features.astype(np.float32)


def build_all_features(
    raw_dir: str = os.path.join("data", "raw"),
    processed_dir: str = os.path.join("data", "processed"),
    models_dir: str = os.path.join("data", "models"),
):
    """
    Build all feature matrices from raw data.

    Outputs:
        data/processed/user_features.npy   (N_users, 32)
        data/processed/post_features.npy   (N_posts, 48)
        data/processed/url_features.npy    (N_urls, 14)
        data/models/text_extractor.pkl     (fitted TextFeatureExtractor)
        data/models/normalizers/           (fitted normalizers per node type)
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Load raw data ──
    print("Loading raw data...")
    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    posts_df = pd.read_csv(os.path.join(raw_dir, "posts.csv"))

    url_file = os.path.join(raw_dir, "urls.csv")
    if os.path.exists(url_file):
        urls_df = pd.read_csv(url_file)
    else:
        # Extract from posts
        url_col = 'url' if 'url' in posts_df.columns else 'URL'
        urls_df = posts_df[[url_col, 'label']].dropna(subset=[url_col]).drop_duplicates(subset=[url_col])

    print(f"  Users: {len(users_df)}, Posts: {len(posts_df)}, URLs: {len(urls_df)}")

    # ── Fit text extractor on all text ──
    print("\nFitting text extractor...")
    all_texts = []
    if 'description' in users_df.columns:
        all_texts.extend(users_df['description'].fillna('').tolist())
    all_texts.extend(posts_df['text'].fillna('').tolist())

    text_extractor = TextFeatureExtractor(embedding_dim=16, n_topics=16)
    text_extractor.fit(all_texts)
    text_extractor.save(os.path.join(models_dir, "text_extractor.pkl"))
    print("  -> Text extractor fitted and saved")

    # ── Build user features (32 dims) ──
    print("\nBuilding user features...")
    user_ext = UserFeatureExtractor(text_extractor=text_extractor)
    bio_col = 'description' if 'description' in users_df.columns else 'bio'
    user_features = user_ext.extract(users_df, posts_df, bio_column=bio_col)
    print(f"  -> User features: {user_features.shape}")

    # ── Build post features (48 dims) ──
    print("\nBuilding post features...")
    post_features = build_post_features(posts_df, text_extractor)
    print(f"  -> Post features: {post_features.shape}")

    # ── Build URL features (14 dims) ──
    print("\nBuilding URL features...")
    url_features = build_url_features(urls_df)
    print(f"  -> URL features: {url_features.shape}")

    # ── Compute content similarity edges ──
    print("\nComputing content similarity edges...")
    post_texts = posts_df['text'].fillna('').tolist()
    similarity_edges = text_extractor.compute_similarity_edges(
        post_texts, threshold=0.85
    )
    print(f"  -> {len(similarity_edges)} similarity edges")

    # ── Normalize features ──
    print("\nNormalizing features...")
    normalizer = MultiNodeNormalizer()
    feature_dict = {
        'user': user_features,
        'post': post_features,
        'url': url_features,
    }
    normalized = normalizer.fit_transform(feature_dict)
    normalizer_dir = os.path.join(models_dir, "normalizers")
    normalizer.save(normalizer_dir)
    print("  -> Normalizers fitted and saved")

    # ── Save feature matrices ──
    print("\nSaving feature matrices...")
    np.save(os.path.join(processed_dir, "user_features.npy"), normalized['user'])
    np.save(os.path.join(processed_dir, "post_features.npy"), normalized['post'])
    np.save(os.path.join(processed_dir, "url_features.npy"), normalized['url'])

    # Save labels
    np.save(
        os.path.join(processed_dir, "user_labels.npy"),
        users_df['is_fake'].values.astype(np.int64),
    )
    np.save(
        os.path.join(processed_dir, "post_labels.npy"),
        posts_df['label'].values.astype(np.int64),
    )

    url_label_col = 'label' if 'label' in urls_df.columns else 'Label'
    if url_label_col in urls_df.columns:
        np.save(
            os.path.join(processed_dir, "url_labels.npy"),
            urls_df[url_label_col].values.astype(np.int64),
        )

    # Save similarity edges
    np.save(
        os.path.join(processed_dir, "similarity_edges.npy"),
        np.array(similarity_edges, dtype=np.int64) if similarity_edges else np.empty((0, 2), dtype=np.int64),
    )

    print("\n[DONE] Feature building complete!")
    print(f"   User features:  {normalized['user'].shape}")
    print(f"   Post features:  {normalized['post'].shape}")
    print(f"   URL features:   {normalized['url'].shape}")
    print(f"   Similarity edges: {len(similarity_edges)}")


if __name__ == "__main__":
    build_all_features()
