import pandas as pd
import urllib.parse
import whois
import socket
import ssl
from datetime import datetime
from tqdm import tqdm

def extract_url_features(url):
    features = {
        'url_length': len(url),
        'domain': '',
        'domain_age_days': -1,
        'has_ssl': 0
    }
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        features['domain'] = domain
    except Exception:
        return pd.Series(features)
        
    # WHOIS Lookup (Domain Age)
    try:
        w = whois.whois(domain)
        if w.creation_date:
            creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            age = (datetime.now() - creation_date).days
            features['domain_age_days'] = age
    except Exception:
        pass
        
    # SSL Check
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(2.0)
            s.connect((domain, 443))
            features['has_ssl'] = 1
    except Exception:
        features['has_ssl'] = 0
        
    return pd.Series(features)

def process_url_data(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return None
        
    if 'url' not in df.columns:
        print("Column 'url' not found in dataset.")
        return None
        
    print("Extracting URL features. This might take a while due to network requests...")
    tqdm.pandas(desc="Processing URLs")
    features_df = df['url'].progress_apply(extract_url_features)
    
    df = pd.concat([df, features_df], axis=1)
    
    df.to_csv(output_csv, index=False)
    print(f"Processed URL data saved to {output_csv}")
    return df
