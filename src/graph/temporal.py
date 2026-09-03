"""
Temporal Graph Representation for the HGNN Campaign Detection System.

Implements discrete temporal snapshots of the heterogeneous graph:
    - Configurable snapshot granularity (default: 1-day windows)
    - Each snapshot is a full HeteroData object
    - Temporal edges connect the same entity across adjacent snapshots
    - TemporalHeteroDataset yields List[HeteroData] for sequence models
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TemporalConfig:
    """Configuration for temporal graph construction."""
    snapshot_hours: float = 24.0      # Hours per snapshot window
    min_nodes_per_snapshot: int = 5   # Minimum nodes to create a snapshot
    max_snapshots: int = 30           # Maximum number of snapshots
    add_temporal_edges: bool = True   # Connect same entities across snapshots


class TemporalHeteroDataset:
    """
    Builds and manages temporal snapshots of a heterogeneous graph.

    Each snapshot contains the nodes and edges active within a time window.
    Nodes persist across snapshots (with temporal edges connecting them),
    while edges are only present in the snapshot where they occurred.

    Usage:
        dataset = TemporalHeteroDataset(config=TemporalConfig(snapshot_hours=24))
        snapshots = dataset.build_snapshots(
            full_data=hetero_data,
            posts_df=posts_df,
            users_df=users_df,
        )
        # snapshots is a list of HeteroData objects
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.snapshots: List[HeteroData] = []
        self.time_boundaries: List[Tuple] = []

    def _assign_timestamps(
        self,
        posts_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure timestamps exist on posts and users.
        If missing, generate synthetic ones based on available data.
        """
        posts = posts_df.copy()
        users = users_df.copy()

        # Posts: use existing timestamp or generate from index
        if 'timestamp' not in posts.columns:
            # Generate timestamps spread over account_age range
            max_age = users['account_age_days'].max() if 'account_age_days' in users.columns else 30
            n_posts = len(posts)
            # Spread posts uniformly over the time range
            days = np.sort(np.random.uniform(0, max_age, n_posts))
            base_time = pd.Timestamp.now() - pd.Timedelta(days=max_age)
            posts['timestamp'] = [
                base_time + pd.Timedelta(days=d) for d in days
            ]
        else:
            posts['timestamp'] = pd.to_datetime(posts['timestamp'])

        # Users: creation time from account_age_days
        if 'created_at' not in users.columns:
            if 'account_age_days' in users.columns:
                now = pd.Timestamp.now()
                users['created_at'] = users['account_age_days'].apply(
                    lambda d: now - pd.Timedelta(days=d)
                )
            else:
                users['created_at'] = pd.Timestamp.now()
        else:
            users['created_at'] = pd.to_datetime(users['created_at'])

        return posts, users

    def _compute_time_windows(
        self,
        posts: pd.DataFrame,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Compute non-overlapping time windows for snapshots.
        """
        min_time = posts['timestamp'].min()
        max_time = posts['timestamp'].max()

        window_delta = pd.Timedelta(hours=self.config.snapshot_hours)
        total_duration = max_time - min_time

        if total_duration <= window_delta:
            return [(min_time, max_time)]

        windows = []
        current = min_time
        while current < max_time and len(windows) < self.config.max_snapshots:
            end = min(current + window_delta, max_time)
            windows.append((current, end))
            current = end

        return windows

    def build_snapshots(
        self,
        full_data: HeteroData,
        posts_df: pd.DataFrame,
        users_df: pd.DataFrame,
        post_mapping: Dict[str, int],
        user_mapping: Dict[str, int],
    ) -> List[HeteroData]:
        """
        Build temporal snapshots from the full heterogeneous graph.

        Each snapshot contains:
            - All user nodes (persistent across snapshots)
            - Posts that were created within the snapshot's time window
            - URLs referenced by those posts
            - All infrastructure nodes (persistent)
            - Edges involving the active nodes

        Args:
            full_data: The complete HeteroData graph
            posts_df: Posts DataFrame (must have or will get 'timestamp')
            users_df: Users DataFrame
            post_mapping: post_id -> node_index mapping
            user_mapping: user_id -> node_index mapping

        Returns:
            List of HeteroData snapshot objects
        """
        posts, users = self._assign_timestamps(posts_df, users_df)
        windows = self._compute_time_windows(posts)
        self.time_boundaries = windows

        snapshots = []

        for w_idx, (w_start, w_end) in enumerate(windows):
            snapshot = HeteroData()

            # Find posts in this window
            mask = (posts['timestamp'] >= w_start) & (posts['timestamp'] <= w_end)
            window_posts = posts[mask]

            if len(window_posts) < self.config.min_nodes_per_snapshot:
                continue

            # Active post indices
            active_post_ids = set(window_posts['post_id'])
            active_post_indices = set()
            for pid in active_post_ids:
                if pid in post_mapping:
                    active_post_indices.add(post_mapping[pid])

            # Active user indices (users who posted in this window)
            active_user_ids = set(window_posts['user_id'])
            active_user_indices = set()
            for uid in active_user_ids:
                if uid in user_mapping:
                    active_user_indices.add(user_mapping[uid])

            # Copy node features for active nodes
            # Users: include ALL users (they persist) but mark active ones
            snapshot['user'].x = full_data['user'].x.clone()
            if hasattr(full_data['user'], 'y'):
                snapshot['user'].y = full_data['user'].y.clone()

            # Posts: only active posts
            if active_post_indices:
                post_idx_list = sorted(active_post_indices)
                snapshot['post'].x = full_data['post'].x[post_idx_list].clone()
                if hasattr(full_data['post'], 'y'):
                    snapshot['post'].y = full_data['post'].y[post_idx_list].clone()

                # Create a mapping from global post index to local snapshot index
                global_to_local_post = {g: l for l, g in enumerate(post_idx_list)}
            else:
                continue  # skip empty snapshots

            # Copy other node types (persistent)
            for ntype in full_data.node_types:
                if ntype not in ['user', 'post'] and hasattr(full_data[ntype], 'x'):
                    snapshot[ntype].x = full_data[ntype].x.clone()
                    if hasattr(full_data[ntype], 'y'):
                        snapshot[ntype].y = full_data[ntype].y.clone()

            # Filter edges to only involve active nodes
            for etype in full_data.edge_types:
                src_type, rel, dst_type = etype
                ei = full_data[etype].edge_index

                if ei.numel() == 0:
                    snapshot[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
                    continue

                # Remap edge indices for post nodes (which are subsetted)
                src_ei = ei[0].tolist()
                dst_ei = ei[1].tolist()

                new_src = []
                new_dst = []

                for s, d in zip(src_ei, dst_ei):
                    # Check if source is active
                    if src_type == 'post':
                        if s not in global_to_local_post:
                            continue
                        s_new = global_to_local_post[s]
                    else:
                        s_new = s

                    # Check if dest is active
                    if dst_type == 'post':
                        if d not in global_to_local_post:
                            continue
                        d_new = global_to_local_post[d]
                    else:
                        d_new = d

                    new_src.append(s_new)
                    new_dst.append(d_new)

                if new_src:
                    snapshot[etype].edge_index = torch.tensor(
                        [new_src, new_dst], dtype=torch.long
                    )
                else:
                    snapshot[etype].edge_index = torch.empty((2, 0), dtype=torch.long)

            # Store metadata
            snapshot.snapshot_idx = w_idx
            snapshot.time_start = str(w_start)
            snapshot.time_end = str(w_end)
            snapshot.num_active_posts = len(active_post_indices)

            snapshots.append(snapshot)

        self.snapshots = snapshots

        # Add temporal edges between consecutive snapshots
        if self.config.add_temporal_edges and len(snapshots) > 1:
            self._add_temporal_edges(snapshots)

        return snapshots

    def _add_temporal_edges(self, snapshots: List[HeteroData]):
        """
        Add temporal self-edges connecting the same user node
        across consecutive snapshots.

        These edges enable the temporal GNN to track entity evolution.
        The edges are stored as metadata on each snapshot (pointing to
        the next snapshot's indices).
        """
        for i in range(len(snapshots) - 1):
            current = snapshots[i]
            next_snap = snapshots[i + 1]

            # For user nodes (persistent, same indexing across snapshots)
            num_users = current['user'].x.shape[0]
            temporal_src = list(range(num_users))
            temporal_dst = list(range(num_users))

            current.temporal_next_user = torch.tensor(
                [temporal_src, temporal_dst], dtype=torch.long
            )

    def get_snapshot_stats(self) -> List[Dict]:
        """Return statistics for each snapshot."""
        stats = []
        for snap in self.snapshots:
            stat = {
                'idx': snap.snapshot_idx,
                'time_start': snap.time_start,
                'time_end': snap.time_end,
                'num_active_posts': snap.num_active_posts,
            }
            for ntype in snap.node_types:
                if hasattr(snap[ntype], 'x'):
                    stat[f'{ntype}_nodes'] = snap[ntype].x.shape[0]
            for etype in snap.edge_types:
                stat[f'{etype}_edges'] = snap[etype].edge_index.shape[1]
            stats.append(stat)
        return stats

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> HeteroData:
        return self.snapshots[idx]

    def __iter__(self):
        return iter(self.snapshots)


if __name__ == '__main__':
    print("=== Temporal Graph Dataset Demo ===\n")

    # Create a minimal test graph
    data = HeteroData()
    data['user'].x = torch.randn(10, 32)
    data['user'].y = torch.randint(0, 2, (10,))
    data['post'].x = torch.randn(20, 48)
    data['post'].y = torch.randint(0, 2, (20,))
    data['url'].x = torch.randn(15, 14)
    data['url'].y = torch.randint(0, 2, (15,))

    data['user', 'posts', 'post'].edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        list(range(20)),
    ], dtype=torch.long)

    data['post', 'contains', 'url'].edge_index = torch.tensor([
        list(range(15)),
        list(range(15)),
    ], dtype=torch.long)

    # Mappings
    user_mapping = {f'U{i:05d}': i for i in range(10)}
    post_mapping = {f'P{i:05d}': i for i in range(20)}

    # DataFrames
    users_df = pd.DataFrame({
        'user_id': [f'U{i:05d}' for i in range(10)],
        'account_age_days': np.random.randint(1, 60, 10),
    })
    posts_df = pd.DataFrame({
        'post_id': [f'P{i:05d}' for i in range(20)],
        'user_id': [f'U{i % 10:05d}' for i in range(20)],
    })

    config = TemporalConfig(snapshot_hours=168, max_snapshots=5)  # weekly
    dataset = TemporalHeteroDataset(config=config)
    snapshots = dataset.build_snapshots(
        data, posts_df, users_df, post_mapping, user_mapping
    )

    print(f"Created {len(snapshots)} temporal snapshots:\n")
    for stat in dataset.get_snapshot_stats():
        print(f"  Snapshot {stat['idx']}: "
              f"posts={stat.get('num_active_posts', 0)}, "
              f"time=[{stat.get('time_start', '?')[:10]}]")
