import pandas as pd
import numpy as np
import os

def main():
    raw_dir = os.path.join("data", "raw")
    users_csv = os.path.join(raw_dir, "users.csv")
    posts_csv = os.path.join(raw_dir, "posts.csv")
    
    if not os.path.exists(users_csv) or not os.path.exists(posts_csv):
        print("Users or posts dataset missing. Run synthetic generation first.")
        return
        
    users_df = pd.read_csv(users_csv)
    posts_df = pd.read_csv(posts_csv)
    
    user_ids = users_df['user_id'].tolist()
    post_ids = posts_df['post_id'].tolist()
    
    print("Generating 'follows' edges...")
    # Follows: Scale with number of users
    num_follows = len(user_ids) * 5
    follows_data = {
        'source_user_id': np.random.choice(user_ids, num_follows, replace=True),
        'target_user_id': np.random.choice(user_ids, num_follows, replace=True)
    }
    follows_df = pd.DataFrame(follows_data)
    # Remove self loops
    follows_df = follows_df[follows_df['source_user_id'] != follows_df['target_user_id']]
    follows_df.drop_duplicates(inplace=True)
    
    follows_csv = os.path.join(raw_dir, "follows.csv")
    follows_df.to_csv(follows_csv, index=False)
    print(f"Saved {len(follows_df)} follows edges to {follows_csv}")
    
    print("Generating 'shares' edges...")
    num_shares = len(post_ids) * 2
    shares_data = {
        'user_id': np.random.choice(user_ids, num_shares, replace=True),
        'post_id': np.random.choice(post_ids, num_shares, replace=True)
    }
    shares_df = pd.DataFrame(shares_data).drop_duplicates()
    
    shares_csv = os.path.join(raw_dir, "shares.csv")
    shares_df.to_csv(shares_csv, index=False)
    print(f"Saved {len(shares_df)} shares edges to {shares_csv}")

if __name__ == "__main__":
    main()
