import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def load_node_features():
    processed_dir = os.path.join("data", "processed")
    raw_dir = os.path.join("data", "raw")
    
    # We will use raw/users.csv since preprocessing didn't alter it much in our synthetic pipeline
    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    # posts.csv from raw
    posts_df = pd.read_csv(os.path.join(raw_dir, "posts.csv"))
    # urls.csv from processed (contains SSL, URL length, Domain Age)
    urls_file = os.path.join(processed_dir, "urls.csv")
    if not os.path.exists(urls_file):
        print(f"Processed URLs not found at {urls_file}. Using raw/urls.csv instead.")
        urls_file = os.path.join(raw_dir, "urls.csv")
        
    urls_df = pd.read_csv(urls_file)
    
    # Process User features
    user_features = users_df[['followers_count', 'following_count', 'account_age_days']].values
    user_x = torch.tensor(user_features, dtype=torch.float)
    user_y = torch.tensor(users_df['is_fake'].values, dtype=torch.long)
    user_mapping = {uid: i for i, uid in enumerate(users_df['user_id'])}
    
    # Process Post features
    posts_df['text_length'] = posts_df['text'].fillna("").apply(len)
    post_features = posts_df[['text_length']].values
    post_x = torch.tensor(post_features, dtype=torch.float)
    post_y = torch.tensor(posts_df['label'].values, dtype=torch.long)
    post_mapping = {pid: i for i, pid in enumerate(posts_df['post_id'])}
    
    # Process URL features
    if 'url_length' in urls_df.columns and 'domain_age_days' in urls_df.columns:
        url_features = urls_df[['url_length', 'domain_age_days', 'has_ssl']].fillna(0).values
    else:
        # Fallback if preprocessing didn't finish
        urls_df['url_length'] = urls_df['URL'].apply(len) if 'URL' in urls_df.columns else urls_df['url'].apply(len)
        url_features = urls_df[['url_length']].values
        
    url_x = torch.tensor(url_features, dtype=torch.float)
    url_y = torch.tensor(urls_df['label'].values, dtype=torch.long)
    url_col = 'URL' if 'URL' in urls_df.columns else 'url'
    url_mapping = {url: i for i, url in enumerate(urls_df[url_col])}
    
    return user_x, user_y, user_mapping, post_x, post_y, post_mapping, url_x, url_y, url_mapping, posts_df

def build_hetero_graph():
    print("Loading node features...")
    user_x, user_y, user_mapping, post_x, post_y, post_mapping, url_x, url_y, url_mapping, posts_df = load_node_features()
    
    data = HeteroData()
    
    # Add nodes
    data['user'].x = user_x
    data['user'].y = user_y
    data['post'].x = post_x
    data['post'].y = post_y
    data['url'].x = url_x
    data['url'].y = url_y
    
    raw_dir = os.path.join("data", "raw")
    
    print("Building edges...")
    # Edge: User -> posts -> Post
    # The 'posts_df' has 'user_id' mapping to 'post_id'
    src_users = [user_mapping[uid] for uid in posts_df['user_id']]
    dst_posts = [post_mapping[pid] for pid in posts_df['post_id']]
    data['user', 'posts', 'post'].edge_index = torch.tensor([src_users, dst_posts], dtype=torch.long)
    
    # Edge: Post -> contains -> URL
    src_posts = []
    dst_urls = []
    for _, row in posts_df.iterrows():
        if pd.notna(row['url']) and row['url'] in url_mapping:
            src_posts.append(post_mapping[row['post_id']])
            dst_urls.append(url_mapping[row['url']])
    data['post', 'contains', 'url'].edge_index = torch.tensor([src_posts, dst_urls], dtype=torch.long)
    
    # Edge: User -> follows -> User
    follows_df = pd.read_csv(os.path.join(raw_dir, "follows.csv"))
    src_follows = [user_mapping[uid] for uid in follows_df['source_user_id']]
    dst_follows = [user_mapping[uid] for uid in follows_df['target_user_id']]
    data['user', 'follows', 'user'].edge_index = torch.tensor([src_follows, dst_follows], dtype=torch.long)
    
    # Edge: User -> shares -> Post
    shares_df = pd.read_csv(os.path.join(raw_dir, "shares.csv"))
    src_shares = [user_mapping[uid] for uid in shares_df['user_id']]
    dst_shares = [post_mapping[pid] for pid in shares_df['post_id']]
    data['user', 'shares', 'post'].edge_index = torch.tensor([src_shares, dst_shares], dtype=torch.long)
    
    print("\n--- Heterogeneous Graph Built Successfully ---")
    print(data)
    
    # Simple validation
    print("\nGraph Validation:")
    print(f"Number of user nodes: {data['user'].num_nodes}")
    print(f"Number of post nodes: {data['post'].num_nodes}")
    print(f"Number of url nodes: {data['url'].num_nodes}")
    
    return data

if __name__ == "__main__":
    build_hetero_graph()
