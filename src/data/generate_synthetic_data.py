"""
Expanded Synthetic Data Generator for the HGNN Campaign Detection System.

Generates:
    - 500 users with bios, timestamps, and profile features
    - 1000 posts with timestamps and rich text
    - URL extraction from posts
    - Coordinated campaign ground truth
    - Follow/share/mention edges
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

try:
    from faker import Faker
except ImportError:
    print("Faker not found. Run 'pip install faker'")
    exit(1)


# ──────────────────────────────────────────────────────────────
# Bio Templates
# ──────────────────────────────────────────────────────────────

BENIGN_BIOS = [
    "Software engineer passionate about {} and {}",
    "Mom/Dad of {}, coffee lover, {} enthusiast",
    "Travel blogger | {} | {} | Living my best life",
    "{} professional | {} | Views are my own",
    "Student at {} University | {} | {}",
    "Writer, reader, {} lover | Based in {}",
    "Photographer | {} | {} | DM for collabs",
    "{} developer | Open source contributor | {}",
    "Music lover | {} fan | {} | Just vibes",
    "Fitness | {} | {} | Healthy living advocate",
    "Teacher | {} | {} | Education matters",
    "Artist | {} | {} | Creating beauty daily",
]

MALICIOUS_BIOS = [
    "Follow for follow! {} {} {}",
    "DM me for exclusive deals! {} {}",
    "",  # Empty bio - common for bots
    "{}{}{}",  # Random characters
    "Click link below for FREE {} !!!",
    "Official giveaway account {}",
    "Crypto trader | {} | 10x returns guaranteed",
    "Make money online {} {} easy cash",
    "Follow me I follow back {}",
    "",
]

BIO_FILLERS = [
    'AI', 'tech', 'music', 'food', 'travel', 'sports', 'gaming',
    'coding', 'design', 'art', 'books', 'movies', 'fitness', 'yoga',
    'NYC', 'London', 'Tokyo', 'SF', 'Berlin', 'Paris', 'Toronto',
    'Python', 'JavaScript', 'React', 'ML', 'Data Science', 'Web3',
    '🔥', '✨', '💯', '🎯', '🚀', '👨‍💻', '📸', '🎵',
]


# ──────────────────────────────────────────────────────────────
# Post Templates
# ──────────────────────────────────────────────────────────────

TEMPLATES_BENIGN = [
    "Just had a great day at the park! {}",
    "Check out this amazing article I found: {}",
    "Loving the new update on this app. {}",
    "Can't believe it's already Friday! {}",
    "Anyone know a good restaurant around here? {}",
    "Here is my latest blog post {}",
    "Beautiful sunset today! Feeling grateful 🌅 {}",
    "Just finished reading an incredible book. Highly recommend! {}",
    "Working on a new project. Excited to share soon! {}",
    "Happy birthday to my best friend! 🎂 {}",
    "Great coffee at the new cafe downtown {}",
    "Weekend vibes! Who else is relaxing today? {}",
    "Just watched an amazing documentary about {} {}",
    "Learning something new every day! {} {}",
    "Throwback to last summer's trip {}",
    "New recipe turned out amazing! {}",
    "Morning run done! Feeling energized 💪 {}",
    "Interesting discussion at work about {} today {}",
]

TEMPLATES_PHISH = [
    "URGENT: Your account has been compromised. Click here to verify: {}",
    "You won a free iPhone! Claim your prize now: {}",
    "Login to your bank account immediately to prevent suspension: {}",
    "Warning! Unauthorized access detected. Secure your account: {}",
    "{} Click the link to claim your reward!",
    "Hot singles in your area want to meet you! {}",
    "CONGRATULATIONS!!! You've been selected! Claim NOW: {}",
    "Your payment of $500 is pending. Confirm here: {}",
    "SECURITY ALERT: Verify your identity immediately: {}",
    "LIMITED TIME: 90% OFF everything! Shop now: {}",
    "Your Netflix account will be suspended. Update payment: {}",
    "IRS NOTICE: You owe $3,000 in taxes. Pay here: {}",
    "FREE Bitcoin giveaway! Send 0.1 ETH get 1 BTC back: {}",
    "Your package could not be delivered. Reschedule: {}",
    "IMPORTANT: Update your password now to avoid lockout: {}",
    "Exclusive offer just for you! Don't miss out: {}",
]


# ──────────────────────────────────────────────────────────────
# Campaign Templates
# ──────────────────────────────────────────────────────────────

CAMPAIGN_TEMPLATES = [
    # Each campaign has a theme, URL domain, and post templates
    {
        'name': 'fake_bank_alert',
        'domain': 'secure-bankverify.com',
        'templates': [
            "URGENT: Your {} account needs verification: {}",
            "Security alert! {} account suspicious activity: {}",
            "Immediate action required for your {} account: {}",
        ],
        'fillers': ['bank', 'PayPal', 'Chase', 'Wells Fargo', 'Citi'],
    },
    {
        'name': 'crypto_scam',
        'domain': 'crypto-giveaway-official.xyz',
        'templates': [
            "FREE {} giveaway! Limited time only: {}",
            "Double your {} investment! Verified by {}: {}",
            "Elon just announced {} airdrop! Claim now: {}",
        ],
        'fillers': ['Bitcoin', 'Ethereum', 'Crypto', 'BTC', 'ETH'],
    },
    {
        'name': 'tech_support_scam',
        'domain': 'microsoft-support-helpdesk.tk',
        'templates': [
            "Your {} computer has a virus! Call now: {}",
            "ALERT: {} detected malware on your device: {}",
            "Critical {} update required. Download now: {}",
        ],
        'fillers': ['Windows', 'Microsoft', 'Apple', 'PC', 'Mac'],
    },
    {
        'name': 'prize_scam',
        'domain': 'winner-claims-center.online',
        'templates': [
            "Congratulations! You won {} {} ! Claim here: {}",
            "You've been selected for a {} {}! Click: {}",
            "WINNER WINNER! {} {} is yours! Verify: {}",
        ],
        'fillers': ['$1000', 'a new iPhone', 'a free vacation',
                    'a gift card', '$5000'],
    },
]


def generate_synthetic_data():
    """Generate expanded synthetic dataset with campaigns."""
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)

    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── Load reference URL dataset if available ──
    urls_df = None
    url_file = os.path.join(raw_dir, "PhiUSIIL_Phishing_URL_Dataset.csv")
    if os.path.exists(url_file):
        try:
            urls_df = pd.read_csv(url_file, usecols=['URL', 'label'])
        except Exception as e:
            print(f"Error reading {url_file}: {e}")

    # ══════════════════════════════════════════════════════════
    # USERS
    # ══════════════════════════════════════════════════════════
    print("Generating 500 synthetic users...")
    num_users = 500
    base_time = datetime.now()
    users_data = []

    # Pre-plan campaigns: assign some users to campaign groups
    num_campaigns = len(CAMPAIGN_TEMPLATES)
    campaign_user_ids = {i: [] for i in range(num_campaigns)}

    for i in range(num_users):
        is_phishing = random.random() < 0.25  # 25% malicious

        if is_phishing:
            followers = int(np.random.exponential(scale=80))
            following = int(np.random.normal(loc=1200, scale=300))
            age_days = random.randint(1, 30)
            bio_template = random.choice(MALICIOUS_BIOS)
            # Campaign assignment: 60% of malicious users join a campaign
            if random.random() < 0.6:
                campaign_idx = random.randint(0, num_campaigns - 1)
                campaign_user_ids[campaign_idx].append(f"U{i:05d}")
                # Campaign users created around the same time
                age_days = random.randint(1, 10)
        else:
            followers = int(np.random.lognormal(mean=5, sigma=1.5))
            following = int(np.random.normal(loc=400, scale=250))
            age_days = random.randint(60, 3000)
            bio_template = random.choice(BENIGN_BIOS)

        # Fill bio template
        fillers = random.sample(BIO_FILLERS, min(3, bio_template.count('{}')))
        try:
            bio = bio_template.format(*fillers)
        except (IndexError, KeyError):
            bio = bio_template

        created_at = base_time - timedelta(days=age_days)

        users_data.append({
            'user_id': f"U{i:05d}",
            'username': fake.user_name(),
            'name': fake.name(),
            'followers_count': max(0, followers),
            'following_count': max(0, following),
            'account_age_days': age_days,
            'description': bio,
            'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'profile_pic': 0 if (is_phishing and random.random() < 0.4) else 1,
            'verified': 0 if is_phishing else (1 if random.random() < 0.05 else 0),
            'is_fake': int(is_phishing),
        })

    users_df = pd.DataFrame(users_data)
    users_csv = os.path.join(raw_dir, "users.csv")
    users_df.to_csv(users_csv, index=False)
    print(f"  -> {num_users} users saved to {users_csv}")
    print(f"  -> {users_df['is_fake'].sum()} malicious ({users_df['is_fake'].mean()*100:.1f}%)")

    # ══════════════════════════════════════════════════════════
    # POSTS
    # ══════════════════════════════════════════════════════════
    print("\nGenerating 1000 synthetic posts...")
    num_posts = 1000
    posts_data = []

    # First: generate campaign posts (coordinated)
    campaign_post_ids = {i: [] for i in range(num_campaigns)}
    post_idx = 0

    for c_idx, campaign in enumerate(CAMPAIGN_TEMPLATES):
        c_users = campaign_user_ids[c_idx]
        if len(c_users) < 2:
            continue

        # Each campaign user makes 2-5 campaign posts
        campaign_base_time = base_time - timedelta(
            days=random.randint(1, 15),
            hours=random.randint(0, 12),
        )

        for uid in c_users:
            num_campaign_posts = random.randint(2, 5)
            for _ in range(num_campaign_posts):
                if post_idx >= num_posts:
                    break

                template = random.choice(campaign['templates'])
                filler = random.choice(campaign['fillers'])
                path = fake.uri_path()
                url = f"https://{campaign['domain']}/{path}"

                try:
                    text = template.format(filler, url)
                except IndexError:
                    text = template.replace('{}', url, 1).replace('{}', filler)

                # Posts in campaign happen within a tight time window
                post_time = campaign_base_time + timedelta(
                    minutes=random.randint(0, 120)
                )

                posts_data.append({
                    'post_id': f"P{post_idx:05d}",
                    'user_id': uid,
                    'text': text,
                    'url': url,
                    'timestamp': post_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'label': 1,
                    'campaign_id': f"C{c_idx:05d}",
                })
                campaign_post_ids[c_idx].append(f"P{post_idx:05d}")
                post_idx += 1

    # Then: fill remaining with regular posts
    while post_idx < num_posts:
        user = random.choice(users_data)
        is_phishing = user['is_fake'] == 1

        url_str = ""
        if urls_df is not None and len(urls_df) > 0:
            subset = urls_df[urls_df['label'] == int(is_phishing)]
            if len(subset) > 0:
                url_str = subset.sample(1).iloc[0]['URL']
            else:
                url_str = urls_df.sample(1).iloc[0]['URL']
        else:
            url_str = fake.url()

        template = random.choice(TEMPLATES_PHISH if is_phishing else TEMPLATES_BENIGN)
        try:
            text = template.format(url_str)
        except (IndexError, KeyError):
            text = template + " " + url_str

        # Random timestamp within account lifetime
        days_ago = random.randint(0, max(1, user['account_age_days']))
        post_time = base_time - timedelta(
            days=days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        has_url = random.random() < (0.95 if is_phishing else 0.5)

        posts_data.append({
            'post_id': f"P{post_idx:05d}",
            'user_id': user['user_id'],
            'text': text,
            'url': url_str if has_url else '',
            'timestamp': post_time.strftime('%Y-%m-%d %H:%M:%S'),
            'label': int(is_phishing),
            'campaign_id': '',
        })
        post_idx += 1

    posts_df = pd.DataFrame(posts_data)
    posts_csv = os.path.join(raw_dir, "posts.csv")
    posts_df.to_csv(posts_csv, index=False)
    print(f"  -> {len(posts_df)} posts saved to {posts_csv}")
    print(f"  -> {posts_df['label'].sum()} malicious ({posts_df['label'].mean()*100:.1f}%)")

    campaign_posts = posts_df[posts_df['campaign_id'] != '']
    print(f"  -> {len(campaign_posts)} campaign posts across {num_campaigns} campaigns")

    # ══════════════════════════════════════════════════════════
    # URLs
    # ══════════════════════════════════════════════════════════
    print("\nExtracting unique URLs...")
    url_posts = posts_df[posts_df['url'].fillna('').str.len() > 0]
    urls_extracted = url_posts[['url', 'label']].drop_duplicates(subset=['url'])
    urls_csv = os.path.join(raw_dir, "urls.csv")
    urls_extracted.to_csv(urls_csv, index=False)
    print(f"  -> {len(urls_extracted)} unique URLs saved to {urls_csv}")

    # ══════════════════════════════════════════════════════════
    # EDGES
    # ══════════════════════════════════════════════════════════
    user_ids = users_df['user_id'].tolist()
    post_ids = posts_df['post_id'].tolist()

    # Follows
    print("\nGenerating follow edges...")
    num_follows = num_users * 5
    follows_data = {
        'source_user_id': np.random.choice(user_ids, num_follows, replace=True),
        'target_user_id': np.random.choice(user_ids, num_follows, replace=True),
    }
    follows_df = pd.DataFrame(follows_data)
    follows_df = follows_df[follows_df['source_user_id'] != follows_df['target_user_id']]
    follows_df.drop_duplicates(inplace=True)

    # Campaign users follow each other more
    for c_idx in range(num_campaigns):
        c_users = campaign_user_ids[c_idx]
        for uid1 in c_users:
            for uid2 in c_users:
                if uid1 != uid2 and random.random() < 0.7:
                    follows_df = pd.concat([follows_df, pd.DataFrame([{
                        'source_user_id': uid1,
                        'target_user_id': uid2,
                    }])], ignore_index=True)

    follows_df.drop_duplicates(inplace=True)
    follows_csv = os.path.join(raw_dir, "follows.csv")
    follows_df.to_csv(follows_csv, index=False)
    print(f"  -> {len(follows_df)} follow edges")

    # Shares
    print("Generating share edges...")
    num_shares = num_posts * 2
    shares_data = {
        'user_id': np.random.choice(user_ids, num_shares, replace=True),
        'post_id': np.random.choice(post_ids, num_shares, replace=True),
    }
    shares_df = pd.DataFrame(shares_data).drop_duplicates()

    # Campaign users share each other's posts
    for c_idx in range(num_campaigns):
        c_users = campaign_user_ids[c_idx]
        c_posts = campaign_post_ids[c_idx]
        for uid in c_users:
            for pid in c_posts:
                if random.random() < 0.5:
                    shares_df = pd.concat([shares_df, pd.DataFrame([{
                        'user_id': uid,
                        'post_id': pid,
                    }])], ignore_index=True)

    shares_df.drop_duplicates(inplace=True)
    shares_csv = os.path.join(raw_dir, "shares.csv")
    shares_df.to_csv(shares_csv, index=False)
    print(f"  -> {len(shares_df)} share edges")

    # Mentions
    print("Generating mention edges...")
    mentions_data = []

    # Campaign users mention each other
    for c_idx in range(num_campaigns):
        c_users = campaign_user_ids[c_idx]
        for uid1 in c_users:
            for uid2 in c_users:
                if uid1 != uid2 and random.random() < 0.5:
                    mentions_data.append({
                        'source_user_id': uid1,
                        'target_user_id': uid2,
                    })

    # Random mentions
    num_random_mentions = num_users * 2
    for _ in range(num_random_mentions):
        mentions_data.append({
            'source_user_id': random.choice(user_ids),
            'target_user_id': random.choice(user_ids),
        })

    mentions_df = pd.DataFrame(mentions_data)
    mentions_df = mentions_df[mentions_df['source_user_id'] != mentions_df['target_user_id']]
    mentions_df.drop_duplicates(inplace=True)
    mentions_csv = os.path.join(raw_dir, "mentions.csv")
    mentions_df.to_csv(mentions_csv, index=False)
    print(f"  -> {len(mentions_df)} mention edges")

    # ══════════════════════════════════════════════════════════
    # Campaign ground truth
    # ══════════════════════════════════════════════════════════
    print("\nSaving campaign ground truth...")
    campaigns_data = []
    for c_idx, campaign in enumerate(CAMPAIGN_TEMPLATES):
        c_users = campaign_user_ids[c_idx]
        c_posts = campaign_post_ids[c_idx]
        if len(c_users) >= 2:
            campaigns_data.append({
                'campaign_id': f"C{c_idx:05d}",
                'name': campaign['name'],
                'num_users': len(c_users),
                'num_posts': len(c_posts),
                'user_ids': ';'.join(c_users),
                'post_ids': ';'.join(c_posts),
                'domain': campaign['domain'],
                'risk_label': 2,  # All synthetic campaigns are malicious
            })
    campaigns_df = pd.DataFrame(campaigns_data)
    campaigns_csv = os.path.join(raw_dir, "campaigns.csv")
    campaigns_df.to_csv(campaigns_csv, index=False)
    print(f"  -> {len(campaigns_df)} campaigns saved to {campaigns_csv}")

    print("\n[DONE] Data generation complete!")
    print(f"   Users:     {num_users}")
    print(f"   Posts:     {len(posts_df)}")
    print(f"   URLs:      {len(urls_extracted)}")
    print(f"   Follows:   {len(follows_df)}")
    print(f"   Shares:    {len(shares_df)}")
    print(f"   Mentions:  {len(mentions_df)}")
    print(f"   Campaigns: {len(campaigns_df)}")


if __name__ == "__main__":
    generate_synthetic_data()
