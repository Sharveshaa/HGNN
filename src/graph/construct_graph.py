import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def load_node_features():
    processed_dir = os.path.join("data", "processed")
    raw_dir = os.path.join("data", "raw")
    
    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    posts_df = pd.read_csv(os.path.join(raw_dir, "posts.csv"))
    
    urls_file = os.path.join(processed_dir, "urls.csv")
    if not os.path.exists(urls_file):
        urls_file = os.path.join(raw_dir, "urls.csv")
    urls_df = pd.read_csv(urls_file)
    url_col = 'URL' if 'URL' in urls_df.columns else 'url'
    
    # Process User
    user_features = users_df[['followers_count', 'following_count', 'account_age_days']].values
    user_x = torch.tensor(user_features, dtype=torch.float)
    user_y = torch.tensor(users_df['is_fake'].values, dtype=torch.long)
    user_mapping = {uid: i for i, uid in enumerate(users_df['user_id'])}
    
    # Process Post
    posts_df['text_length'] = posts_df['text'].fillna("").apply(len)
    post_features = posts_df[['text_length']].values
    post_x = torch.tensor(post_features, dtype=torch.float)
    post_y = torch.tensor(posts_df['label'].values, dtype=torch.long)
    post_mapping = {pid: i for i, pid in enumerate(posts_df['post_id'])}
    
    # Process URL
    url_mapping = {url: i for i, url in enumerate(urls_df[url_col])}
    
    if 'url_length' in urls_df.columns and 'domain' in urls_df.columns:
        url_features = urls_df[['url_length', 'url_entropy', 'is_shortened', 'has_suspicious_tld', 'path_depth', 'subdomain_count']].fillna(0).values
        url_x = torch.tensor(url_features, dtype=torch.float)
        
        # Build Domain Nodes
        unique_domains = urls_df['domain'].dropna().unique()
        domain_mapping = {d: i for i, d in enumerate(unique_domains)}
        
        # Build domain feature matrix (aggregate from urls_df)
        domain_features_dict = {}
        for _, row in urls_df.iterrows():
            d = row['domain']
            if pd.notna(d) and d not in domain_features_dict:
                domain_features_dict[d] = [
                    float(row.get('domain_age_days', 0)),
                    float(row.get('days_to_expiry', 0)),
                    float(row.get('has_mx_records', 0)),
                    float(row.get('has_ssl', 0))
                ]
        domain_x_list = [domain_features_dict[d] for d in unique_domains]
        domain_x = torch.tensor(domain_x_list, dtype=torch.float)
        
        # IPs
        unique_ips = urls_df['ip'].dropna().unique()
        ip_mapping = {ip: i for i, ip in enumerate(unique_ips)}
        ip_x = torch.ones((len(unique_ips), 1), dtype=torch.float) # Dummy feature
        
        # ASNs
        unique_asns = urls_df['asn'].dropna().unique()
        asn_mapping = {a: i for i, a in enumerate(unique_asns)}
        asn_x = torch.ones((len(unique_asns), 1), dtype=torch.float)
        
        # Registrars
        unique_registrars = urls_df['registrar'].dropna().unique()
        registrar_mapping = {r: i for i, r in enumerate(unique_registrars)}
        registrar_x = torch.ones((len(unique_registrars), 1), dtype=torch.float)
        
    else:
        urls_df['url_length'] = urls_df[url_col].apply(len)
        url_features = urls_df[['url_length']].values
        url_x = torch.tensor(url_features, dtype=torch.float)
        domain_x = ip_x = asn_x = registrar_x = None
        domain_mapping = ip_mapping = asn_mapping = registrar_mapping = None

    url_y = torch.tensor(urls_df['label'].values, dtype=torch.long)
    
    return (user_x, user_y, user_mapping, 
            post_x, post_y, post_mapping, 
            url_x, url_y, url_mapping, posts_df, urls_df,
            domain_x, domain_mapping, ip_x, ip_mapping, 
            asn_x, asn_mapping, registrar_x, registrar_mapping)

def build_hetero_graph():
    print("Loading node features...")
    (user_x, user_y, user_mapping, 
     post_x, post_y, post_mapping, 
     url_x, url_y, url_mapping, posts_df, urls_df,
     domain_x, domain_mapping, ip_x, ip_mapping, 
     asn_x, asn_mapping, registrar_x, registrar_mapping) = load_node_features()
    
    data = HeteroData()
    
    data['user'].x = user_x
    data['user'].y = user_y
    data['post'].x = post_x
    data['post'].y = post_y
    data['url'].x = url_x
    data['url'].y = url_y
    
    if domain_x is not None:
        data['domain'].x = domain_x
        data['ip'].x = ip_x
        data['asn'].x = asn_x
        data['registrar'].x = registrar_x
        
    raw_dir = os.path.join("data", "raw")
    
    print("Building edges...")
    # Edge: User -> posts -> Post
    src_users = [user_mapping[uid] for uid in posts_df['user_id']]
    dst_posts = [post_mapping[pid] for pid in posts_df['post_id']]
    data['user', 'posts', 'post'].edge_index = torch.tensor([src_users, dst_posts], dtype=torch.long)
    
    # Edge: Post -> contains -> URL
    src_posts, dst_urls = [], []
    url_col = 'URL' if 'URL' in posts_df.columns else 'url'
    for _, row in posts_df.iterrows():
        if pd.notna(row[url_col]) and row[url_col] in url_mapping:
            src_posts.append(post_mapping[row['post_id']])
            dst_urls.append(url_mapping[row[url_col]])
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
    
    if domain_x is not None:
        # url -> hosted_on -> domain
        src_u, dst_d = [], []
        # domain -> resolves_to -> ip
        src_d_ip, dst_ip = [], []
        # domain -> registered_via -> registrar
        src_d_reg, dst_reg = [], []
        # ip -> belongs_to -> asn
        src_ip, dst_asn = [], []
        
        seen_d_ip = set()
        seen_d_reg = set()
        seen_ip_asn = set()
        
        for _, row in urls_df.iterrows():
            url = row['URL'] if 'URL' in row else row['url']
            url_id = url_mapping[url]
            dom = row.get('domain')
            
            if pd.notna(dom):
                dom_id = domain_mapping.get(dom)
                if dom_id is not None:
                    src_u.append(url_id)
                    dst_d.append(dom_id)
                    
                    ip = row.get('ip')
                    if pd.notna(ip) and ip in ip_mapping and (dom_id, ip_mapping[ip]) not in seen_d_ip:
                        src_d_ip.append(dom_id)
                        dst_ip.append(ip_mapping[ip])
                        seen_d_ip.add((dom_id, ip_mapping[ip]))
                        
                        asn = row.get('asn')
                        if pd.notna(asn) and asn in asn_mapping and (ip_mapping[ip], asn_mapping[asn]) not in seen_ip_asn:
                            src_ip.append(ip_mapping[ip])
                            dst_asn.append(asn_mapping[asn])
                            seen_ip_asn.add((ip_mapping[ip], asn_mapping[asn]))
                            
                    reg = row.get('registrar')
                    if pd.notna(reg) and reg in registrar_mapping and (dom_id, registrar_mapping[reg]) not in seen_d_reg:
                        src_d_reg.append(dom_id)
                        dst_reg.append(registrar_mapping[reg])
                        seen_d_reg.add((dom_id, registrar_mapping[reg]))
                        
        data['url', 'hosted_on', 'domain'].edge_index = torch.tensor([src_u, dst_d], dtype=torch.long)
        
        if len(src_d_ip) > 0:
            data['domain', 'resolves_to', 'ip'].edge_index = torch.tensor([src_d_ip, dst_ip], dtype=torch.long)
        else:
            data['domain', 'resolves_to', 'ip'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if len(src_d_reg) > 0:
            data['domain', 'registered_via', 'registrar'].edge_index = torch.tensor([src_d_reg, dst_reg], dtype=torch.long)
        else:
            data['domain', 'registered_via', 'registrar'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if len(src_ip) > 0:
            data['ip', 'belongs_to', 'asn'].edge_index = torch.tensor([src_ip, dst_asn], dtype=torch.long)
        else:
            data['ip', 'belongs_to', 'asn'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    print("\n--- Heterogeneous Graph Built Successfully ---")
    print(data)
    
    return data, user_mapping, post_mapping, url_mapping, domain_mapping, ip_mapping, asn_mapping, registrar_mapping

if __name__ == "__main__":
    build_hetero_graph()
