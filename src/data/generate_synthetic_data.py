import pandas as pd
import numpy as np
import os
import random

try:
    from faker import Faker
except ImportError:
    print("Faker not found. Run 'pip install faker'")
    exit(1)

def generate_synthetic_data():
    fake = Faker()
    raw_dir = os.path.join("data", "raw")
    
    print("Reading reference datasets to understand distributions...")
    # Reference for URL dataset (using PhiUSIIL as the main url source)
    urls_df = None
    url_file = os.path.join(raw_dir, "PhiUSIIL_Phishing_URL_Dataset.csv")
    if os.path.exists(url_file):
        try:
            urls_df = pd.read_csv(url_file, usecols=['URL', 'label'])
        except Exception as e:
            print(f"Error reading {url_file}: {e}")
        
    print("Generating synthetic users...")
    # Generate Synthetic Users (100 users)
    num_users = 100
    users_data = []
    for i in range(num_users):
        is_phishing = random.random() < 0.2  # 20% malicious
        
        # Base stats on label (phishing vs benign distributions)
        if is_phishing:
            followers = int(np.random.exponential(scale=100))
            following = int(np.random.normal(loc=1000, scale=200))
            age_days = random.randint(1, 30)
        else:
            followers = int(np.random.lognormal(mean=5, sigma=2))
            following = int(np.random.normal(loc=500, scale=300))
            age_days = random.randint(30, 3000)
            
        users_data.append({
            'user_id': f"U{i:05d}",
            'username': fake.user_name(),
            'followers_count': max(0, followers),
            'following_count': max(0, following),
            'account_age_days': age_days,
            'is_fake': int(is_phishing)
        })
        
    users_df = pd.DataFrame(users_data)
    users_csv = os.path.join(raw_dir, "users.csv")
    users_df.to_csv(users_csv, index=False)
    print(f"Generated {num_users} synthetic users to {users_csv}")
    
    print("Generating synthetic posts...")
    # Generate Synthetic Posts (200 posts)
    num_posts = 200
    posts_data = []
    
    templates_benign = [
        "Just had a great day at the park! {}",
        "Check out this amazing article I found: {}",
        "Loving the new update on this app. {}",
        "Can't believe it's already Friday. {}",
        "Anyone know a good restaurant around here? {}",
        "{}", # just url
        "Here is my latest blog post {}"
    ]
    templates_phish = [
        "URGENT: Your account has been compromised. Click here to verify: {}",
        "You won a free iPhone! Claim your prize now: {}",
        "Login to your bank account immediately to prevent suspension: {}",
        "Warning! Unauthorized access detected. Secure your account: {}",
        "{} Click the link to claim your reward!",
        "Hot singles in your area want to meet you! {}"
    ]
    
    for i in range(num_posts):
        user = random.choice(users_data)
        is_phishing = user['is_fake'] == 1
        
        # Get a URL
        url_str = ""
        if urls_df is not None and len(urls_df) > 0:
            # try to match label
            subset = urls_df[urls_df['label'] == is_phishing]
            if len(subset) > 0:
                url_str = subset.sample(1).iloc[0]['URL']
            else:
                url_str = urls_df.sample(1).iloc[0]['URL']
        else:
            url_str = fake.url()
            
        template = random.choice(templates_phish if is_phishing else templates_benign)
        text = template.format(url_str) if "{}" in template else template + " " + url_str
        
        posts_data.append({
            'post_id': f"P{i:05d}",
            'user_id': user['user_id'],
            'text': text,
            'url': url_str,
            'label': int(is_phishing)
        })
        
    posts_df = pd.DataFrame(posts_data)
    posts_csv = os.path.join(raw_dir, "posts.csv")
    posts_df.to_csv(posts_csv, index=False)
    print(f"Generated {num_posts} synthetic posts to {posts_csv}")
    
    # Generate urls.csv from the posts to match preprocessing expected input
    print("Generating urls.csv from posts...")
    urls_extracted = posts_df[['url', 'label']].drop_duplicates(subset=['url'])
    urls_extracted_csv = os.path.join(raw_dir, "urls.csv")
    urls_extracted.to_csv(urls_extracted_csv, index=False)
    print(f"Extracted {len(urls_extracted)} unique URLs to {urls_extracted_csv}")

if __name__ == "__main__":
    generate_synthetic_data()
