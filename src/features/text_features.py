"""
Text Feature Extraction for the HGNN Campaign Detection System.

Extracts rich NLP features from text content using lightweight methods:
    - Bio/Text Embeddings: TF-IDF → TruncatedSVD (dimensionality-reduced)
    - Topic Embeddings: TF-IDF → NMF (Non-negative Matrix Factorization)
    - Sentiment Score: VADER compound score
    - Urgency Score: Keyword-based urgency detection
    - Clickbait Score: Pattern-based clickbait detection
    - Toxicity Score: Lexicon-based toxicity scoring
    - Emotion Vector: 8-dim Plutchik emotion vector via keyword lexicons

All extractors are designed to be fast (no transformer inference) and produce
fixed-dimension outputs suitable for graph node features.
"""

import re
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
import pickle
import os


# ──────────────────────────────────────────────────────────────
# Lexicons and Constants
# ──────────────────────────────────────────────────────────────

URGENCY_KEYWORDS = [
    'urgent', 'immediately', 'act now', 'expires', 'limited time',
    'hurry', 'rush', 'deadline', 'last chance', 'don\'t miss',
    'warning', 'alert', 'critical', 'important', 'action required',
    'suspend', 'compromised', 'verify', 'confirm', 'unauthorized',
    'locked', 'disabled', 'restricted', 'security', 'violation',
    'expiring', 'final notice', 'respond', 'asap', 'right away',
]

CLICKBAIT_PATTERNS = [
    r'you won\'t believe',
    r'shocking',
    r'amazing',
    r'incredible',
    r'this is why',
    r'what happens next',
    r'number \d+ will',
    r'doctors hate',
    r'one weird trick',
    r'click here',
    r'find out',
    r'free\s+(iphone|gift|money|prize|reward)',
    r'claim (your|now|today)',
    r'congratulations',
    r'winner',
    r'selected',
    r'exclusive offer',
    r'limited offer',
    r'act fast',
    r'buy now',
]

TOXICITY_WORDS = [
    'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'loser',
    'pathetic', 'disgusting', 'trash', 'garbage', 'worthless',
    'shut up', 'scam', 'fraud', 'fake', 'liar', 'cheat',
    'threat', 'attack', 'destroy', 'revenge', 'creep', 'freak',
]

# Plutchik's 8 primary emotions with associated keywords
EMOTION_LEXICON: Dict[str, List[str]] = {
    'joy': [
        'happy', 'joy', 'love', 'wonderful', 'amazing', 'great',
        'excited', 'fantastic', 'beautiful', 'celebrate', 'laugh',
        'smile', 'fun', 'delight', 'cheerful', 'pleased', 'glad',
        'thrilled', 'awesome', 'brilliant', 'excellent', 'perfect',
    ],
    'trust': [
        'trust', 'believe', 'faith', 'reliable', 'honest', 'loyal',
        'secure', 'safe', 'confident', 'certain', 'genuine', 'authentic',
        'credible', 'dependable', 'faithful', 'truthful', 'sincere',
    ],
    'fear': [
        'fear', 'afraid', 'scared', 'terrified', 'panic', 'horror',
        'dread', 'anxiety', 'worry', 'nervous', 'frightened', 'alarmed',
        'threat', 'danger', 'risk', 'warning', 'suspicious', 'creepy',
    ],
    'surprise': [
        'surprise', 'shocked', 'amazed', 'astonished', 'unexpected',
        'unbelievable', 'incredible', 'wow', 'omg', 'suddenly',
        'startled', 'stunned', 'bewildered', 'speechless', 'bizarre',
    ],
    'sadness': [
        'sad', 'depressed', 'unhappy', 'miserable', 'grief', 'sorrow',
        'heartbroken', 'lonely', 'disappointed', 'regret', 'cry',
        'tears', 'painful', 'suffer', 'loss', 'tragic', 'gloomy',
    ],
    'disgust': [
        'disgust', 'gross', 'nasty', 'revolting', 'repulsive', 'vile',
        'horrible', 'awful', 'sick', 'nausea', 'repugnant', 'loathe',
        'abhor', 'detest', 'contempt', 'foul', 'filthy', 'putrid',
    ],
    'anger': [
        'angry', 'furious', 'rage', 'hate', 'mad', 'outraged',
        'hostile', 'aggressive', 'irritated', 'annoyed', 'frustrated',
        'resentful', 'bitter', 'enraged', 'livid', 'infuriated',
    ],
    'anticipation': [
        'expect', 'anticipate', 'hope', 'wait', 'look forward',
        'predict', 'plan', 'prepare', 'ready', 'eager', 'curious',
        'interested', 'wonder', 'upcoming', 'future', 'soon',
    ],
}

EMOTION_NAMES = list(EMOTION_LEXICON.keys())
NUM_EMOTIONS = len(EMOTION_NAMES)  # 8


# ──────────────────────────────────────────────────────────────
# Simple VADER-like Sentiment Scorer
# ──────────────────────────────────────────────────────────────

# Lightweight sentiment without external dependencies
_POSITIVE_WORDS = {
    'good', 'great', 'love', 'like', 'best', 'happy', 'awesome',
    'excellent', 'wonderful', 'fantastic', 'amazing', 'beautiful',
    'brilliant', 'perfect', 'nice', 'cool', 'enjoy', 'fun',
    'glad', 'pleased', 'thanks', 'thank', 'helpful', 'recommend',
    'outstanding', 'superb', 'magnificent', 'delightful', 'pleasant',
}

_NEGATIVE_WORDS = {
    'bad', 'hate', 'worst', 'terrible', 'horrible', 'awful',
    'ugly', 'stupid', 'boring', 'annoying', 'disappointed',
    'angry', 'sad', 'wrong', 'fail', 'poor', 'useless',
    'pathetic', 'disgusting', 'dreadful', 'miserable', 'nasty',
    'scam', 'fraud', 'fake', 'suspicious', 'dangerous', 'threat',
}

_NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nor', "n't", 'dont', "don't"}
_INTENSIFIERS = {'very', 'really', 'extremely', 'so', 'absolutely', 'totally'}


def _compute_sentiment(text: str) -> float:
    """
    Compute a sentiment score in [-1, 1] using a simple lexicon approach.
    Handles negation and intensifiers.
    """
    if not text or not isinstance(text, str):
        return 0.0

    words = text.lower().split()
    score = 0.0
    total = 0
    negate = False
    intensify = 1.0

    for word in words:
        clean = re.sub(r'[^a-z]', '', word)
        if not clean:
            continue

        if clean in _NEGATION_WORDS or word.endswith("n't"):
            negate = True
            continue

        if clean in _INTENSIFIERS:
            intensify = 1.5
            continue

        val = 0.0
        if clean in _POSITIVE_WORDS:
            val = 1.0
        elif clean in _NEGATIVE_WORDS:
            val = -1.0

        if val != 0.0:
            if negate:
                val = -val
                negate = False
            val *= intensify
            intensify = 1.0
            score += val
            total += 1

    if total == 0:
        return 0.0

    # Normalize to [-1, 1] using sigmoid-like squashing
    raw = score / total
    return max(-1.0, min(1.0, raw))


# ──────────────────────────────────────────────────────────────
# Text Feature Extractor Class
# ──────────────────────────────────────────────────────────────

class TextFeatureExtractor:
    """
    Extracts multiple NLP features from text data.

    Usage:
        extractor = TextFeatureExtractor(embedding_dim=16, n_topics=16)
        extractor.fit(texts)            # fit TF-IDF, SVD, NMF on corpus
        features = extractor.transform(texts)  # returns dict of arrays
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        n_topics: int = 16,
        max_tfidf_features: int = 5000,
        similarity_threshold: float = 0.85,
    ):
        self.embedding_dim = embedding_dim
        self.n_topics = n_topics
        self.max_tfidf_features = max_tfidf_features
        self.similarity_threshold = similarity_threshold

        # Models (fitted during .fit())
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._nmf: Optional[NMF] = None
        self._is_fitted = False

    def _clean_text(self, text: str) -> str:
        """Clean text for NLP processing."""
        if not text or not isinstance(text, str) or pd.isna(text):
            return ""
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
        return text.lower()

    def fit(self, texts: List[str]) -> 'TextFeatureExtractor':
        """
        Fit TF-IDF, SVD, and NMF models on a corpus of texts.

        Args:
            texts: List of raw text strings (will be cleaned internally)
        """
        cleaned = [self._clean_text(t) for t in texts]

        # Handle empty corpus
        non_empty = [t for t in cleaned if t.strip()]
        if len(non_empty) < 2:
            # Not enough data to fit; create dummy models
            self._is_fitted = True
            return self

        # TF-IDF vectorizer
        self._tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words='english',
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        tfidf_matrix = self._tfidf.fit_transform(cleaned)

        # SVD for dense embeddings
        actual_svd_dim = min(self.embedding_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if actual_svd_dim < 1:
            actual_svd_dim = 1
        self._svd = TruncatedSVD(n_components=actual_svd_dim, random_state=42)
        self._svd.fit(tfidf_matrix)

        # NMF for topic embeddings
        actual_nmf_dim = min(self.n_topics, tfidf_matrix.shape[1], tfidf_matrix.shape[0])
        if actual_nmf_dim < 1:
            actual_nmf_dim = 1
        self._nmf = NMF(
            n_components=actual_nmf_dim,
            random_state=42,
            max_iter=500,
            init='nndsvda',
        )
        self._nmf.fit(tfidf_matrix)

        self._is_fitted = True
        return self

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get dense SVD embeddings for texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if not self._is_fitted or self._tfidf is None:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        cleaned = [self._clean_text(t) for t in texts]
        tfidf_matrix = self._tfidf.transform(cleaned)
        embeddings = self._svd.transform(tfidf_matrix)

        # Pad to target dim if SVD produced fewer components
        if embeddings.shape[1] < self.embedding_dim:
            pad = np.zeros(
                (embeddings.shape[0], self.embedding_dim - embeddings.shape[1]),
                dtype=np.float32,
            )
            embeddings = np.hstack([embeddings, pad])

        return embeddings.astype(np.float32)

    def get_topic_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get NMF topic embeddings for texts.

        Returns:
            np.ndarray of shape (len(texts), n_topics)
        """
        if not self._is_fitted or self._tfidf is None:
            return np.zeros((len(texts), self.n_topics), dtype=np.float32)

        cleaned = [self._clean_text(t) for t in texts]
        tfidf_matrix = self._tfidf.transform(cleaned)
        topics = self._nmf.transform(tfidf_matrix)

        # Pad to target dim if NMF produced fewer components
        if topics.shape[1] < self.n_topics:
            pad = np.zeros(
                (topics.shape[0], self.n_topics - topics.shape[1]),
                dtype=np.float32,
            )
            topics = np.hstack([topics, pad])

        return topics.astype(np.float32)

    def get_sentiment(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentiment scores for texts.

        Returns:
            np.ndarray of shape (len(texts), 1) with values in [-1, 1]
        """
        scores = [_compute_sentiment(t) for t in texts]
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

    def get_urgency(self, texts: List[str]) -> np.ndarray:
        """
        Compute urgency scores based on keyword density.

        Returns:
            np.ndarray of shape (len(texts), 1) with values in [0, 1]
        """
        scores = []
        for text in texts:
            if not text or not isinstance(text, str):
                scores.append(0.0)
                continue
            text_lower = text.lower()
            matches = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
            # Normalize: sigmoid-like scaling, cap at 1.0
            score = min(1.0, matches / 3.0)  # 3+ keywords = max urgency
            scores.append(score)
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

    def get_clickbait(self, texts: List[str]) -> np.ndarray:
        """
        Compute clickbait scores based on pattern matching and stylistic features.

        Returns:
            np.ndarray of shape (len(texts), 1) with values in [0, 1]
        """
        scores = []
        for text in texts:
            if not text or not isinstance(text, str):
                scores.append(0.0)
                continue

            text_lower = text.lower()
            score = 0.0

            # Pattern matching
            pattern_hits = sum(
                1 for pat in CLICKBAIT_PATTERNS if re.search(pat, text_lower)
            )
            score += min(0.5, pattern_hits * 0.15)

            # Exclamation density
            excl_count = text.count('!')
            score += min(0.2, excl_count * 0.05)

            # CAPS ratio
            if len(text) > 0:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                if caps_ratio > 0.3:
                    score += 0.2

            # Number presence (listicle-style)
            if re.search(r'\b\d+\b', text):
                score += 0.1

            scores.append(min(1.0, score))
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

    def get_toxicity(self, texts: List[str]) -> np.ndarray:
        """
        Compute toxicity scores based on lexicon matching.

        Returns:
            np.ndarray of shape (len(texts), 1) with values in [0, 1]
        """
        scores = []
        for text in texts:
            if not text or not isinstance(text, str):
                scores.append(0.0)
                continue

            text_lower = text.lower()
            words = text_lower.split()
            word_count = max(len(words), 1)

            # Count toxic word matches
            toxic_hits = sum(1 for tw in TOXICITY_WORDS if tw in text_lower)

            # Normalize by text length (longer text can have more hits naturally)
            score = min(1.0, toxic_hits / max(3.0, word_count * 0.1))
            scores.append(score)
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

    def get_emotion_vectors(self, texts: List[str]) -> np.ndarray:
        """
        Compute 8-dimensional Plutchik emotion vectors.

        Each dimension corresponds to one of the 8 primary emotions:
        [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]

        Returns:
            np.ndarray of shape (len(texts), 8) with values in [0, 1]
        """
        vectors = []
        for text in texts:
            if not text or not isinstance(text, str):
                vectors.append([0.0] * NUM_EMOTIONS)
                continue

            text_lower = text.lower()
            words = set(text_lower.split())
            word_count = max(len(words), 1)

            emotion_vec = []
            for emotion_name in EMOTION_NAMES:
                keywords = EMOTION_LEXICON[emotion_name]
                hits = sum(1 for kw in keywords if kw in words or kw in text_lower)
                # Normalize to [0, 1]
                score = min(1.0, hits / max(2.0, word_count * 0.05))
                emotion_vec.append(score)

            vectors.append(emotion_vec)
        return np.array(vectors, dtype=np.float32)

    def compute_similarity_edges(
        self, texts: List[str], threshold: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        Compute content similarity edges between texts.

        Returns list of (idx_i, idx_j) pairs where cosine similarity > threshold.
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if not self._is_fitted or self._tfidf is None:
            return []

        cleaned = [self._clean_text(t) for t in texts]
        tfidf_matrix = self._tfidf.transform(cleaned)
        embeddings = self._svd.transform(tfidf_matrix)

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        edges = []
        n = len(texts)
        # Batch compute similarities to avoid O(n^2) memory
        batch_size = 500
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            sims = normalized[i:end_i] @ normalized.T
            for local_idx in range(end_i - i):
                global_i = i + local_idx
                for j in range(global_i + 1, n):
                    if sims[local_idx, j] > threshold:
                        edges.append((global_i, j))

        return edges

    def transform(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract all text features.

        Returns:
            Dictionary with keys:
                'embeddings': (N, embedding_dim) SVD embeddings
                'topics': (N, n_topics) NMF topic vectors
                'sentiment': (N, 1)
                'urgency': (N, 1)
                'clickbait': (N, 1)
                'toxicity': (N, 1)
                'emotion': (N, 8)
        """
        return {
            'embeddings': self.get_embeddings(texts),
            'topics': self.get_topic_embeddings(texts),
            'sentiment': self.get_sentiment(texts),
            'urgency': self.get_urgency(texts),
            'clickbait': self.get_clickbait(texts),
            'toxicity': self.get_toxicity(texts),
            'emotion': self.get_emotion_vectors(texts),
        }

    def save(self, path: str):
        """Save fitted models to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'tfidf': self._tfidf,
            'svd': self._svd,
            'nmf': self._nmf,
            'embedding_dim': self.embedding_dim,
            'n_topics': self.n_topics,
            'max_tfidf_features': self.max_tfidf_features,
            'similarity_threshold': self.similarity_threshold,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'TextFeatureExtractor':
        """Load fitted models from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        extractor = cls(
            embedding_dim=state['embedding_dim'],
            n_topics=state['n_topics'],
            max_tfidf_features=state['max_tfidf_features'],
            similarity_threshold=state['similarity_threshold'],
        )
        extractor._tfidf = state['tfidf']
        extractor._svd = state['svd']
        extractor._nmf = state['nmf']
        extractor._is_fitted = True
        return extractor


if __name__ == '__main__':
    # Quick demo
    sample_texts = [
        "URGENT: Your account has been compromised. Click here to verify now!",
        "Just had a great day at the park with friends!",
        "You won a free iPhone! Claim your prize immediately!!!",
        "Check out this amazing new restaurant I found downtown.",
        "Warning! Unauthorized access detected. Secure your account now!",
        "Loving the new update. Really happy with the changes.",
        "This is absolutely disgusting behavior, I hate it.",
        "Can't wait for the concert next week, so excited!",
    ]

    print("=== Text Feature Extractor Demo ===\n")
    extractor = TextFeatureExtractor(embedding_dim=8, n_topics=4)
    extractor.fit(sample_texts)
    features = extractor.transform(sample_texts)

    for key, arr in features.items():
        print(f"{key:12s}: shape={arr.shape}")

    print("\n--- Per-text scores ---")
    for i, text in enumerate(sample_texts):
        print(f"\n[{i}] {text[:60]}...")
        print(f"     sentiment={features['sentiment'][i,0]:.3f}  "
              f"urgency={features['urgency'][i,0]:.3f}  "
              f"clickbait={features['clickbait'][i,0]:.3f}  "
              f"toxicity={features['toxicity'][i,0]:.3f}")
        emotions = {name: features['emotion'][i, j]
                    for j, name in enumerate(EMOTION_NAMES)}
        top_emotions = sorted(emotions.items(), key=lambda x: -x[1])[:3]
        print(f"     top emotions: {top_emotions}")

    # Similarity edges
    sim_edges = extractor.compute_similarity_edges(sample_texts, threshold=0.3)
    print(f"\nSimilarity edges (threshold=0.3): {len(sim_edges)} pairs")
    for i, j in sim_edges[:5]:
        print(f"  [{i}] <-> [{j}]")
