"""
Heterogeneous Graph Construction for the HGNN Campaign Detection System.

Builds a PyG HeteroData graph from pre-computed feature matrices with:
    - 8 node types: user, post, url, domain, ip, asn, registrar, campaign
    - 12 edge types including campaign membership and content similarity
    - Rich feature vectors from the feature extraction pipeline
    - Campaign node detection and integration
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_precomputed_features(processed_dir="data/processed"):
    """Load pre-computed feature matrices from .npy files."""
    features = {}

    for name in ['user_features', 'post_features', 'url_features',
                 'user_labels', 'post_labels', 'url_labels',
                 'similarity_edges']:
        path = os.path.join(processed_dir, f"{name}.npy")
        if os.path.exists(path):
            features[name] = np.load(path, allow_pickle=True)
        else:
            features[name] = None

    return features


def build_hetero_graph(
    raw_dir="data/raw",
    processed_dir="data/processed",
    use_rich_features=True,
):
    """
    Build the complete heterogeneous graph.

    If use_rich_features=True, loads pre-computed feature matrices from
    the feature pipeline. Otherwise falls back to basic feature extraction.

    Returns:
        (data, mappings_dict)
        data: HeteroData
        mappings_dict: dict of {node_type: {id: index}}
    """
    print("=" * 60)
    print("Building Heterogeneous Graph")
    print("=" * 60)

    # ── Load raw data ──
    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    posts_df = pd.read_csv(os.path.join(raw_dir, "posts.csv"))

    url_col = 'url' if 'url' in posts_df.columns else 'URL'

    # Load URLs
    urls_file = os.path.join(processed_dir, "urls.csv")
    if not os.path.exists(urls_file):
        urls_file = os.path.join(raw_dir, "urls.csv")
    if os.path.exists(urls_file):
        urls_df = pd.read_csv(urls_file)
    else:
        urls_df = posts_df[[url_col, 'label']].dropna(subset=[url_col]).drop_duplicates(subset=[url_col])

    url_id_col = 'URL' if 'URL' in urls_df.columns else 'url'

    # ── Mappings ──
    user_mapping = {uid: i for i, uid in enumerate(users_df['user_id'])}
    post_mapping = {pid: i for i, pid in enumerate(posts_df['post_id'])}
    url_mapping = {url: i for i, url in enumerate(urls_df[url_id_col].dropna())}

    # ── Load feature matrices ──
    precomputed = load_precomputed_features(processed_dir) if use_rich_features else {}

    data = HeteroData()

    # ══════════════════════════════════════════════════════════
    # NODE FEATURES
    # ══════════════════════════════════════════════════════════

    # Users
    if precomputed.get('user_features') is not None:
        data['user'].x = torch.tensor(precomputed['user_features'], dtype=torch.float)
        print(f"  [user] Rich features: {data['user'].x.shape}")
    else:
        user_features = users_df[['followers_count', 'following_count', 'account_age_days']].values
        data['user'].x = torch.tensor(user_features, dtype=torch.float)
        print(f"  [user] Basic features: {data['user'].x.shape}")

    if precomputed.get('user_labels') is not None:
        data['user'].y = torch.tensor(precomputed['user_labels'], dtype=torch.long)
    else:
        data['user'].y = torch.tensor(users_df['is_fake'].values, dtype=torch.long)

    # Posts
    if precomputed.get('post_features') is not None:
        data['post'].x = torch.tensor(precomputed['post_features'], dtype=torch.float)
        print(f"  [post] Rich features: {data['post'].x.shape}")
    else:
        posts_df['text_length'] = posts_df['text'].fillna("").apply(len)
        post_features = posts_df[['text_length']].values
        data['post'].x = torch.tensor(post_features, dtype=torch.float)
        print(f"  [post] Basic features: {data['post'].x.shape}")

    if precomputed.get('post_labels') is not None:
        data['post'].y = torch.tensor(precomputed['post_labels'], dtype=torch.long)
    else:
        data['post'].y = torch.tensor(posts_df['label'].values, dtype=torch.long)

    # URLs
    if precomputed.get('url_features') is not None:
        data['url'].x = torch.tensor(precomputed['url_features'], dtype=torch.float)
        print(f"  [url] Rich features: {data['url'].x.shape}")
    else:
        # Basic URL features
        urls_df['url_length'] = urls_df[url_id_col].apply(lambda u: len(str(u)))
        data['url'].x = torch.tensor(urls_df[['url_length']].values, dtype=torch.float)
        print(f"  [url] Basic features: {data['url'].x.shape}")

    url_label_col = 'label' if 'label' in urls_df.columns else 'Label'
    if precomputed.get('url_labels') is not None:
        data['url'].y = torch.tensor(precomputed['url_labels'], dtype=torch.long)
    elif url_label_col in urls_df.columns:
        data['url'].y = torch.tensor(urls_df[url_label_col].values, dtype=torch.long)

    # Infrastructure nodes (domain, ip, asn, registrar)
    # These use enriched URL data if available
    domain_mapping = {}
    ip_mapping = {}
    asn_mapping = {}
    registrar_mapping = {}

    enriched_urls_file = os.path.join(processed_dir, "urls.csv")
    if os.path.exists(enriched_urls_file):
        enriched_urls = pd.read_csv(enriched_urls_file)
        if 'domain' in enriched_urls.columns:
            # Domain nodes
            unique_domains = enriched_urls['domain'].dropna().unique()
            domain_mapping = {d: i for i, d in enumerate(unique_domains)}
            domain_features = []
            for d in unique_domains:
                d_rows = enriched_urls[enriched_urls['domain'] == d].iloc[0]
                domain_features.append([
                    float(d_rows.get('domain_age_days', 0)),
                    float(d_rows.get('days_to_expiry', 0)),
                    float(d_rows.get('has_mx_records', 0)),
                    float(d_rows.get('has_ssl', 0)),
                ])
            if domain_features:
                data['domain'].x = torch.tensor(domain_features, dtype=torch.float)
                print(f"  [domain] {data['domain'].x.shape}")

            # IP nodes
            if 'ip' in enriched_urls.columns:
                unique_ips = enriched_urls['ip'].dropna().unique()
                ip_mapping = {ip: i for i, ip in enumerate(unique_ips)}
                data['ip'].x = torch.ones((len(unique_ips), 1), dtype=torch.float)
                print(f"  [ip] {data['ip'].x.shape}")

            # ASN nodes
            if 'asn' in enriched_urls.columns:
                unique_asns = enriched_urls['asn'].dropna().unique()
                asn_mapping = {a: i for i, a in enumerate(unique_asns)}
                data['asn'].x = torch.ones((len(unique_asns), 1), dtype=torch.float)
                print(f"  [asn] {data['asn'].x.shape}")

            # Registrar nodes
            if 'registrar' in enriched_urls.columns:
                unique_registrars = enriched_urls['registrar'].dropna().unique()
                registrar_mapping = {r: i for i, r in enumerate(unique_registrars)}
                data['registrar'].x = torch.ones((len(unique_registrars), 1), dtype=torch.float)
                print(f"  [registrar] {data['registrar'].x.shape}")

    # If no infrastructure nodes, create minimal ones
    if not domain_mapping:
        data['domain'].x = torch.ones((1, 4), dtype=torch.float)
        domain_mapping = {'default': 0}
    if not ip_mapping:
        data['ip'].x = torch.ones((1, 1), dtype=torch.float)
        ip_mapping = {'default': 0}
    if not asn_mapping:
        data['asn'].x = torch.ones((1, 1), dtype=torch.float)
        asn_mapping = {'default': 0}
    if not registrar_mapping:
        data['registrar'].x = torch.ones((1, 1), dtype=torch.float)
        registrar_mapping = {'default': 0}

    # ══════════════════════════════════════════════════════════
    # EDGES
    # ══════════════════════════════════════════════════════════
    print("\nBuilding edges...")

    # (user, posts, post) — authorship
    src_users = [user_mapping[uid] for uid in posts_df['user_id'] if uid in user_mapping]
    dst_posts = [post_mapping[pid] for pid in posts_df['post_id'] if pid in post_mapping]
    min_len = min(len(src_users), len(dst_posts))
    data['user', 'posts', 'post'].edge_index = torch.tensor(
        [src_users[:min_len], dst_posts[:min_len]], dtype=torch.long
    )
    print(f"  (user, posts, post): {min_len} edges")

    # (post, contains, url)
    src_posts, dst_urls = [], []
    for _, row in posts_df.iterrows():
        url = row.get(url_col) or row.get('URL')
        if pd.notna(url) and url in url_mapping:
            pid = row['post_id']
            if pid in post_mapping:
                src_posts.append(post_mapping[pid])
                dst_urls.append(url_mapping[url])
    data['post', 'contains', 'url'].edge_index = torch.tensor(
        [src_posts, dst_urls], dtype=torch.long
    ) if src_posts else torch.empty((2, 0), dtype=torch.long)
    print(f"  (post, contains, url): {len(src_posts)} edges")

    # (user, follows, user)
    follows_file = os.path.join(raw_dir, "follows.csv")
    if os.path.exists(follows_file):
        follows_df = pd.read_csv(follows_file)
        src_f = [user_mapping[uid] for uid in follows_df['source_user_id'] if uid in user_mapping]
        dst_f = [user_mapping[uid] for uid in follows_df['target_user_id'] if uid in user_mapping]
        min_len = min(len(src_f), len(dst_f))
        data['user', 'follows', 'user'].edge_index = torch.tensor(
            [src_f[:min_len], dst_f[:min_len]], dtype=torch.long
        )
        print(f"  (user, follows, user): {min_len} edges")
    else:
        data['user', 'follows', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # (user, shares, post)
    shares_file = os.path.join(raw_dir, "shares.csv")
    if os.path.exists(shares_file):
        shares_df = pd.read_csv(shares_file)
        src_s = [user_mapping[uid] for uid in shares_df['user_id'] if uid in user_mapping]
        dst_s = [post_mapping[pid] for pid in shares_df['post_id'] if pid in post_mapping]
        min_len = min(len(src_s), len(dst_s))
        data['user', 'shares', 'post'].edge_index = torch.tensor(
            [src_s[:min_len], dst_s[:min_len]], dtype=torch.long
        )
        print(f"  (user, shares, post): {min_len} edges")
    else:
        data['user', 'shares', 'post'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # (user, mentions, user) — NEW
    mentions_file = os.path.join(raw_dir, "mentions.csv")
    if os.path.exists(mentions_file):
        mentions_df = pd.read_csv(mentions_file)
        src_m = [user_mapping[uid] for uid in mentions_df['source_user_id'] if uid in user_mapping]
        dst_m = [user_mapping[uid] for uid in mentions_df['target_user_id'] if uid in user_mapping]
        min_len = min(len(src_m), len(dst_m))
        data['user', 'mentions', 'user'].edge_index = torch.tensor(
            [src_m[:min_len], dst_m[:min_len]], dtype=torch.long
        )
        print(f"  (user, mentions, user): {min_len} edges")
    else:
        data['user', 'mentions', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # (post, similar_to, post) — NEW: content similarity edges
    sim_edges = precomputed.get('similarity_edges')
    if sim_edges is not None and len(sim_edges) > 0:
        if sim_edges.ndim == 2 and sim_edges.shape[1] == 2:
            src_sim = sim_edges[:, 0].tolist()
            dst_sim = sim_edges[:, 1].tolist()
        else:
            src_sim, dst_sim = [], []
        # Make bidirectional
        all_src = src_sim + dst_sim
        all_dst = dst_sim + src_sim
        data['post', 'similar_to', 'post'].edge_index = torch.tensor(
            [all_src, all_dst], dtype=torch.long
        )
        print(f"  (post, similar_to, post): {len(all_src)} edges")
    else:
        data['post', 'similar_to', 'post'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Infrastructure edges
    if 'domain' in enriched_urls.columns if 'enriched_urls' in dir() else False:
        _build_infra_edges(data, enriched_urls, url_mapping, domain_mapping,
                          ip_mapping, asn_mapping, registrar_mapping)
    else:
        # Minimal infrastructure edges
        data['url', 'hosted_on', 'domain'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['domain', 'resolves_to', 'ip'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['domain', 'registered_via', 'registrar'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['ip', 'belongs_to', 'asn'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # ══════════════════════════════════════════════════════════
    # CAMPAIGN NODES — NEW
    # ══════════════════════════════════════════════════════════
    campaign_mapping = {}
    campaigns_file = os.path.join(raw_dir, "campaigns.csv")
    if os.path.exists(campaigns_file):
        print("\nBuilding campaign nodes...")
        campaigns_df = pd.read_csv(campaigns_file)

        num_campaigns = len(campaigns_df)
        campaign_feature_dim = 16

        # Build campaign features from member aggregation
        campaign_features = np.zeros((num_campaigns, campaign_feature_dim), dtype=np.float32)

        user_x = data['user'].x.numpy()
        post_x = data['post'].x.numpy()

        user_campaign_src, user_campaign_dst = [], []
        post_campaign_src, post_campaign_dst = [], []

        for c_idx, row in campaigns_df.iterrows():
            cid = row['campaign_id']
            campaign_mapping[cid] = c_idx

            # Aggregate user features
            if pd.notna(row.get('user_ids', '')):
                member_uids = str(row['user_ids']).split(';')
                member_indices = [user_mapping[uid] for uid in member_uids if uid in user_mapping]
                if member_indices:
                    member_feats = user_x[member_indices]
                    pooled = member_feats.mean(axis=0)
                    dim = min(campaign_feature_dim, len(pooled))
                    campaign_features[c_idx, :dim] = pooled[:dim]

                for uid in member_uids:
                    if uid in user_mapping:
                        user_campaign_src.append(user_mapping[uid])
                        user_campaign_dst.append(c_idx)

            # Aggregate post features
            if pd.notna(row.get('post_ids', '')):
                member_pids = str(row['post_ids']).split(';')
                for pid in member_pids:
                    if pid in post_mapping:
                        post_campaign_src.append(post_mapping[pid])
                        post_campaign_dst.append(c_idx)

        data['campaign'].x = torch.tensor(campaign_features, dtype=torch.float)
        data['campaign'].y = torch.tensor(
            campaigns_df['risk_label'].values, dtype=torch.long
        ) if 'risk_label' in campaigns_df.columns else torch.zeros(num_campaigns, dtype=torch.long)

        # Campaign membership edges
        if user_campaign_src:
            data['user', 'member_of', 'campaign'].edge_index = torch.tensor(
                [user_campaign_src, user_campaign_dst], dtype=torch.long
            )
            print(f"  (user, member_of, campaign): {len(user_campaign_src)} edges")
        else:
            data['user', 'member_of', 'campaign'].edge_index = torch.empty((2, 0), dtype=torch.long)

        if post_campaign_src:
            data['post', 'part_of', 'campaign'].edge_index = torch.tensor(
                [post_campaign_src, post_campaign_dst], dtype=torch.long
            )
            print(f"  (post, part_of, campaign): {len(post_campaign_src)} edges")
        else:
            data['post', 'part_of', 'campaign'].edge_index = torch.empty((2, 0), dtype=torch.long)

        print(f"  [campaign] {data['campaign'].x.shape}")
    else:
        # No campaigns file — create minimal campaign node
        data['campaign'].x = torch.ones((1, 16), dtype=torch.float)
        data['campaign'].y = torch.zeros(1, dtype=torch.long)
        campaign_mapping = {'C00000': 0}
        data['user', 'member_of', 'campaign'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['post', 'part_of', 'campaign'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Heterogeneous Graph Summary")
    print("=" * 60)
    print(data)

    mappings = {
        'user': user_mapping,
        'post': post_mapping,
        'url': url_mapping,
        'domain': domain_mapping,
        'ip': ip_mapping,
        'asn': asn_mapping,
        'registrar': registrar_mapping,
        'campaign': campaign_mapping,
    }

    return data, mappings


def _build_infra_edges(data, urls_df, url_mapping, domain_mapping,
                       ip_mapping, asn_mapping, registrar_mapping):
    """Build infrastructure edges from enriched URL data."""
    url_id_col = 'URL' if 'URL' in urls_df.columns else 'url'

    src_u_d, dst_u_d = [], []
    src_d_ip, dst_d_ip = [], []
    src_d_reg, dst_d_reg = [], []
    src_ip_asn, dst_ip_asn = [], []

    seen_d_ip = set()
    seen_d_reg = set()
    seen_ip_asn = set()

    for _, row in urls_df.iterrows():
        url = row.get(url_id_col)
        if pd.isna(url) or url not in url_mapping:
            continue

        url_idx = url_mapping[url]
        dom = row.get('domain')

        if pd.notna(dom) and dom in domain_mapping:
            dom_idx = domain_mapping[dom]
            src_u_d.append(url_idx)
            dst_u_d.append(dom_idx)

            ip = row.get('ip')
            if pd.notna(ip) and ip in ip_mapping:
                key = (dom_idx, ip_mapping[ip])
                if key not in seen_d_ip:
                    src_d_ip.append(dom_idx)
                    dst_d_ip.append(ip_mapping[ip])
                    seen_d_ip.add(key)

                    asn = row.get('asn')
                    if pd.notna(asn) and asn in asn_mapping:
                        key2 = (ip_mapping[ip], asn_mapping[asn])
                        if key2 not in seen_ip_asn:
                            src_ip_asn.append(ip_mapping[ip])
                            dst_ip_asn.append(asn_mapping[asn])
                            seen_ip_asn.add(key2)

            reg = row.get('registrar')
            if pd.notna(reg) and reg in registrar_mapping:
                key = (dom_idx, registrar_mapping[reg])
                if key not in seen_d_reg:
                    src_d_reg.append(dom_idx)
                    dst_d_reg.append(registrar_mapping[reg])
                    seen_d_reg.add(key)

    data['url', 'hosted_on', 'domain'].edge_index = torch.tensor(
        [src_u_d, dst_u_d], dtype=torch.long
    ) if src_u_d else torch.empty((2, 0), dtype=torch.long)

    data['domain', 'resolves_to', 'ip'].edge_index = torch.tensor(
        [src_d_ip, dst_d_ip], dtype=torch.long
    ) if src_d_ip else torch.empty((2, 0), dtype=torch.long)

    data['domain', 'registered_via', 'registrar'].edge_index = torch.tensor(
        [src_d_reg, dst_d_reg], dtype=torch.long
    ) if src_d_reg else torch.empty((2, 0), dtype=torch.long)

    data['ip', 'belongs_to', 'asn'].edge_index = torch.tensor(
        [src_ip_asn, dst_ip_asn], dtype=torch.long
    ) if src_ip_asn else torch.empty((2, 0), dtype=torch.long)

    print(f"  (url, hosted_on, domain): {len(src_u_d)} edges")
    print(f"  (domain, resolves_to, ip): {len(src_d_ip)} edges")
    print(f"  (domain, registered_via, registrar): {len(src_d_reg)} edges")
    print(f"  (ip, belongs_to, asn): {len(src_ip_asn)} edges")


if __name__ == "__main__":
    data, mappings = build_hetero_graph()
