"""
Campaign Detection for the HGNN Campaign Detection System.

Detects coordinated campaigns using multi-signal heuristics:
    1. Temporal Burst Detection: Users posting same/similar URLs in short windows
    2. Content Similarity Clustering: Posts with high cosine similarity from distinct users
    3. Infrastructure Co-location: Multiple users linking to URLs on same domain/IP
    4. Behavioral Synchrony: Users created within 48h with similar posting patterns

Each detected campaign is assigned a risk label:
    0 = benign, 1 = suspicious, 2 = malicious
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import urllib.parse


@dataclass
class Campaign:
    """Represents a detected campaign cluster."""
    campaign_id: str
    user_ids: Set[str] = field(default_factory=set)
    post_ids: Set[str] = field(default_factory=set)
    url_domains: Set[str] = field(default_factory=set)
    detection_signals: List[str] = field(default_factory=list)
    risk_label: int = 0  # 0=benign, 1=suspicious, 2=malicious
    risk_score: float = 0.0
    features: Optional[np.ndarray] = None


class CampaignDetector:
    """
    Detects coordinated campaigns from social network data using
    multi-signal heuristics.

    Usage:
        detector = CampaignDetector()
        campaigns = detector.detect(users_df, posts_df, similarity_edges)
        campaign_features = detector.get_campaign_features(campaigns)
    """

    def __init__(
        self,
        temporal_window_hours: float = 1.0,
        similarity_threshold: float = 0.85,
        infra_min_users: int = 3,
        sync_window_hours: float = 48.0,
        min_campaign_size: int = 2,
    ):
        """
        Args:
            temporal_window_hours: Time window for burst detection
            similarity_threshold: Cosine sim threshold for content clustering
            infra_min_users: Min users sharing same domain to trigger infra signal
            sync_window_hours: Window for behavioral synchrony detection
            min_campaign_size: Minimum number of users to form a campaign
        """
        self.temporal_window_hours = temporal_window_hours
        self.similarity_threshold = similarity_threshold
        self.infra_min_users = infra_min_users
        self.sync_window_hours = sync_window_hours
        self.min_campaign_size = min_campaign_size

    def _detect_temporal_bursts(
        self,
        posts_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Detect groups of posts sharing the same URL within a short time window.

        Returns list of dicts with 'user_ids', 'post_ids', 'url' keys.
        """
        bursts = []

        url_col = 'url' if 'url' in posts_df.columns else 'URL'
        if url_col not in posts_df.columns:
            return bursts

        # Group posts by URL
        url_groups = posts_df.dropna(subset=[url_col]).groupby(url_col)

        for url, group in url_groups:
            if len(group) < self.min_campaign_size:
                continue

            # Get unique users posting this URL
            unique_users = group['user_id'].unique()
            if len(unique_users) < self.min_campaign_size:
                continue

            # Check temporal proximity if timestamps available
            if 'timestamp' in group.columns:
                try:
                    times = pd.to_datetime(group['timestamp']).sort_values()
                    # Sliding window: check if enough posts fall within the window
                    window_td = pd.Timedelta(hours=self.temporal_window_hours)

                    for i in range(len(times)):
                        window_end = times.iloc[i] + window_td
                        in_window = times[(times >= times.iloc[i]) & (times <= window_end)]
                        if len(in_window) >= self.min_campaign_size:
                            window_posts = group.loc[in_window.index]
                            window_users = window_posts['user_id'].unique()
                            if len(window_users) >= self.min_campaign_size:
                                bursts.append({
                                    'user_ids': set(window_users),
                                    'post_ids': set(window_posts['post_id']),
                                    'url': url,
                                    'signal': 'temporal_burst',
                                })
                                break  # one burst per URL is enough
                except Exception:
                    pass
            else:
                # Without timestamps, just flag multi-user URL sharing
                bursts.append({
                    'user_ids': set(unique_users),
                    'post_ids': set(group['post_id']),
                    'url': url,
                    'signal': 'url_sharing',
                })

        return bursts

    def _detect_content_similarity_clusters(
        self,
        posts_df: pd.DataFrame,
        similarity_edges: List[Tuple[int, int]],
    ) -> List[Dict]:
        """
        Detect clusters of similar posts from distinct users.

        Args:
            similarity_edges: List of (post_idx_i, post_idx_j) pairs
        """
        clusters = []

        if not similarity_edges:
            return clusters

        # Build adjacency from similarity edges
        adj: Dict[int, Set[int]] = defaultdict(set)
        for i, j in similarity_edges:
            adj[i].add(j)
            adj[j].add(i)

        # Connected components via BFS
        visited = set()
        for start in adj:
            if start in visited:
                continue

            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj[node] - visited)

            if len(component) < self.min_campaign_size:
                continue

            # Get user IDs for this component
            user_ids = set()
            post_ids = set()
            for idx in component:
                if idx < len(posts_df):
                    user_ids.add(posts_df.iloc[idx]['user_id'])
                    post_ids.add(posts_df.iloc[idx]['post_id'])

            if len(user_ids) >= self.min_campaign_size:
                clusters.append({
                    'user_ids': user_ids,
                    'post_ids': post_ids,
                    'signal': 'content_similarity',
                })

        return clusters

    def _detect_infra_colocation(
        self,
        posts_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Detect multiple users linking to URLs on the same domain.
        """
        clusters = []

        url_col = 'url' if 'url' in posts_df.columns else 'URL'
        if url_col not in posts_df.columns:
            return clusters

        # Extract domains from URLs
        domain_users: Dict[str, Dict] = defaultdict(
            lambda: {'user_ids': set(), 'post_ids': set(), 'urls': set()}
        )

        for _, row in posts_df.iterrows():
            url = row.get(url_col)
            if pd.isna(url):
                continue

            try:
                parsed = urllib.parse.urlparse(str(url))
                domain = parsed.netloc
                if domain:
                    domain_users[domain]['user_ids'].add(row['user_id'])
                    domain_users[domain]['post_ids'].add(row['post_id'])
                    domain_users[domain]['urls'].add(str(url))
            except Exception:
                continue

        for domain, info in domain_users.items():
            if len(info['user_ids']) >= self.infra_min_users:
                clusters.append({
                    'user_ids': info['user_ids'],
                    'post_ids': info['post_ids'],
                    'domain': domain,
                    'signal': 'infra_colocation',
                })

        return clusters

    def _detect_behavioral_synchrony(
        self,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Detect users created within a short time window with similar activity.
        """
        clusters = []

        if 'account_age_days' not in users_df.columns:
            return clusters

        # Group users by creation window
        ages = users_df[['user_id', 'account_age_days']].copy()
        ages = ages.sort_values('account_age_days')

        # Sliding window over account ages
        window_days = self.sync_window_hours / 24.0
        i = 0
        while i < len(ages):
            current_age = ages.iloc[i]['account_age_days']
            group_users = set()
            j = i

            while j < len(ages) and abs(ages.iloc[j]['account_age_days'] - current_age) <= window_days:
                group_users.add(ages.iloc[j]['user_id'])
                j += 1

            if len(group_users) >= self.min_campaign_size:
                # Check if these users have similar posting patterns
                group_posts = posts_df[posts_df['user_id'].isin(group_users)]
                if len(group_posts) >= self.min_campaign_size:
                    clusters.append({
                        'user_ids': group_users,
                        'post_ids': set(group_posts['post_id']),
                        'signal': 'behavioral_synchrony',
                    })

            i = max(i + 1, j)

        return clusters

    def _merge_overlapping_campaigns(
        self,
        raw_clusters: List[Dict],
    ) -> List[Dict]:
        """
        Merge clusters that share significant user overlap.
        Two clusters are merged if they share > 50% of their users.
        """
        if not raw_clusters:
            return []

        merged = []
        used = set()

        for i, cluster_i in enumerate(raw_clusters):
            if i in used:
                continue

            current = {
                'user_ids': set(cluster_i['user_ids']),
                'post_ids': set(cluster_i['post_ids']),
                'signals': [cluster_i['signal']],
            }

            for j, cluster_j in enumerate(raw_clusters):
                if j <= i or j in used:
                    continue

                overlap = current['user_ids'] & cluster_j['user_ids']
                min_size = min(len(current['user_ids']), len(cluster_j['user_ids']))

                if min_size > 0 and len(overlap) / min_size > 0.5:
                    current['user_ids'] |= cluster_j['user_ids']
                    current['post_ids'] |= cluster_j['post_ids']
                    current['signals'].append(cluster_j['signal'])
                    used.add(j)

            used.add(i)
            merged.append(current)

        return merged

    def _compute_risk_label(
        self,
        campaign: Dict,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
    ) -> Tuple[int, float]:
        """
        Compute risk label for a campaign based on member maliciousness.

        Returns (risk_label, risk_score)
            risk_label: 0=benign, 1=suspicious, 2=malicious
            risk_score: float in [0, 1]
        """
        # Fraction of malicious users
        label_col = 'is_fake' if 'is_fake' in users_df.columns else 'label'
        if label_col not in users_df.columns:
            return 0, 0.0

        member_users = users_df[users_df['user_id'].isin(campaign['user_ids'])]
        if len(member_users) == 0:
            return 0, 0.0

        malicious_fraction = member_users[label_col].mean()

        # Fraction of malicious posts
        post_label_col = 'label'
        member_posts = posts_df[posts_df['post_id'].isin(campaign['post_ids'])]
        if len(member_posts) > 0 and post_label_col in member_posts.columns:
            post_malicious = member_posts[post_label_col].mean()
        else:
            post_malicious = 0.0

        # Combined score
        risk_score = 0.6 * malicious_fraction + 0.4 * post_malicious

        # Multi-signal bonus: campaigns detected by multiple signals are riskier
        num_signals = len(set(campaign.get('signals', [])))
        risk_score = min(1.0, risk_score + 0.1 * (num_signals - 1))

        if risk_score > 0.6:
            risk_label = 2  # malicious
        elif risk_score > 0.3:
            risk_label = 1  # suspicious
        else:
            risk_label = 0  # benign

        return risk_label, risk_score

    def detect(
        self,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        similarity_edges: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Campaign]:
        """
        Run all detection heuristics and return discovered campaigns.

        Args:
            users_df: User profiles DataFrame
            posts_df: Posts DataFrame
            similarity_edges: Optional pre-computed content similarity edges

        Returns:
            List of Campaign objects
        """
        all_clusters = []

        # Signal 1: Temporal bursts
        bursts = self._detect_temporal_bursts(posts_df)
        all_clusters.extend(bursts)

        # Signal 2: Content similarity
        if similarity_edges:
            sim_clusters = self._detect_content_similarity_clusters(
                posts_df, similarity_edges
            )
            all_clusters.extend(sim_clusters)

        # Signal 3: Infrastructure co-location
        infra_clusters = self._detect_infra_colocation(posts_df)
        all_clusters.extend(infra_clusters)

        # Signal 4: Behavioral synchrony
        sync_clusters = self._detect_behavioral_synchrony(users_df, posts_df)
        all_clusters.extend(sync_clusters)

        # Merge overlapping clusters
        merged = self._merge_overlapping_campaigns(all_clusters)

        # Create Campaign objects
        campaigns = []
        for i, cluster in enumerate(merged):
            risk_label, risk_score = self._compute_risk_label(
                cluster, users_df, posts_df
            )

            campaign = Campaign(
                campaign_id=f"C{i:05d}",
                user_ids=cluster['user_ids'],
                post_ids=cluster['post_ids'],
                detection_signals=cluster.get('signals', []),
                risk_label=risk_label,
                risk_score=risk_score,
            )
            campaigns.append(campaign)

        return campaigns

    def get_campaign_node_features(
        self,
        campaigns: List[Campaign],
        user_features: Optional[np.ndarray] = None,
        post_features: Optional[np.ndarray] = None,
        user_ids: Optional[List[str]] = None,
        post_ids: Optional[List[str]] = None,
        feature_dim: int = 16,
    ) -> np.ndarray:
        """
        Compute campaign node feature vectors by aggregating member features.

        Uses mean pooling over member user/post features. When member features
        are unavailable, uses statistical descriptors of the campaign.

        Returns:
            np.ndarray of shape (num_campaigns, feature_dim)
        """
        campaign_features = np.zeros(
            (len(campaigns), feature_dim), dtype=np.float32
        )

        user_id_to_idx = {}
        if user_ids:
            user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        post_id_to_idx = {}
        if post_ids:
            post_id_to_idx = {pid: i for i, pid in enumerate(post_ids)}

        for c_idx, campaign in enumerate(campaigns):
            aggregated_parts = []

            # Aggregate user features
            if user_features is not None and user_id_to_idx:
                member_indices = [
                    user_id_to_idx[uid]
                    for uid in campaign.user_ids
                    if uid in user_id_to_idx
                ]
                if member_indices:
                    member_feats = user_features[member_indices]
                    # Mean pool and truncate/pad to feature_dim
                    pooled = member_feats.mean(axis=0)
                    aggregated_parts.append(pooled)

            # Aggregate post features
            if post_features is not None and post_id_to_idx:
                member_indices = [
                    post_id_to_idx[pid]
                    for pid in campaign.post_ids
                    if pid in post_id_to_idx
                ]
                if member_indices:
                    member_feats = post_features[member_indices]
                    pooled = member_feats.mean(axis=0)
                    aggregated_parts.append(pooled)

            if aggregated_parts:
                # Concatenate and resize to target dim
                combined = np.concatenate(aggregated_parts)
                if len(combined) >= feature_dim:
                    campaign_features[c_idx] = combined[:feature_dim]
                else:
                    campaign_features[c_idx, :len(combined)] = combined
            else:
                # Fallback: use campaign statistics as features
                campaign_features[c_idx, 0] = len(campaign.user_ids)
                campaign_features[c_idx, 1] = len(campaign.post_ids)
                campaign_features[c_idx, 2] = len(campaign.detection_signals)
                campaign_features[c_idx, 3] = campaign.risk_score

        return campaign_features

    def get_membership_edges(
        self,
        campaigns: List[Campaign],
        user_id_mapping: Dict[str, int],
        post_id_mapping: Dict[str, int],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Generate (user, member_of, campaign) and (post, part_of, campaign)
        edge index lists.

        Returns:
            (user_campaign_edges, post_campaign_edges) as lists of (src, dst) tuples
        """
        user_edges = []  # (user_idx, campaign_idx)
        post_edges = []  # (post_idx, campaign_idx)

        for c_idx, campaign in enumerate(campaigns):
            for uid in campaign.user_ids:
                if uid in user_id_mapping:
                    user_edges.append((user_id_mapping[uid], c_idx))

            for pid in campaign.post_ids:
                if pid in post_id_mapping:
                    post_edges.append((post_id_mapping[pid], c_idx))

        return user_edges, post_edges


if __name__ == '__main__':
    print("=== Campaign Detector Demo ===\n")

    # Create synthetic test data
    users_df = pd.DataFrame({
        'user_id': [f'U{i:05d}' for i in range(10)],
        'account_age_days': [500, 5, 3, 300, 2, 4, 200, 6, 100, 7],
        'is_fake': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    })

    posts_df = pd.DataFrame({
        'post_id': [f'P{i:05d}' for i in range(15)],
        'user_id': ['U00000', 'U00001', 'U00002', 'U00003', 'U00004',
                    'U00005', 'U00001', 'U00002', 'U00004', 'U00005',
                    'U00006', 'U00007', 'U00008', 'U00009', 'U00007'],
        'text': [
            'Great day!', 'Click here FREE!', 'URGENT verify now!',
            'Nice weather', 'Claim your prize!', 'Act immediately!',
            'Another scam link', 'Verify account!', 'Win big today!',
            'Security alert!', 'Good morning', 'Suspicious link here',
            'Coffee time', 'Free money!!!', 'Another phish',
        ],
        'url': [
            None, 'http://evil.com/a', 'http://evil.com/b',
            None, 'http://evil.com/c', 'http://scam.xyz/d',
            'http://evil.com/e', 'http://evil.com/f', 'http://scam.xyz/g',
            'http://scam.xyz/h', None, 'http://phish.tk/i',
            None, 'http://phish.tk/j', 'http://phish.tk/k',
        ],
        'label': [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    })

    detector = CampaignDetector(min_campaign_size=2, infra_min_users=2)
    campaigns = detector.detect(users_df, posts_df)

    print(f"Detected {len(campaigns)} campaigns:\n")
    for c in campaigns:
        print(f"  {c.campaign_id}: {len(c.user_ids)} users, "
              f"{len(c.post_ids)} posts, risk={c.risk_label} "
              f"({c.risk_score:.2f})")
        print(f"    Users: {sorted(c.user_ids)}")
        print(f"    Signals: {c.detection_signals}")
        print()
