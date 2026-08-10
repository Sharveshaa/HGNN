import pandas as pd
import urllib.parse
import whois
import socket
import ssl
import math
import tldextract
import dns.resolver
import requests
from datetime import datetime
from tqdm import tqdm

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy

def extract_url_features(url):
    features = {
        'url_length': len(url),
        'url_entropy': shannon_entropy(url),
        'is_shortened': 0,
        'has_suspicious_tld': 0,
        'path_depth': 0,
        'subdomain_count': 0,
        'domain': '',
        'domain_age_days': -1,
        'days_to_expiry': -1,
        'has_mx_records': 0,
        'has_ssl': 0,
        'ip': '',
        'asn': '',
        'registrar': ''
    }
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        ext = tldextract.extract(url)
        features['domain'] = f"{ext.domain}.{ext.suffix}"
        
        if ext.subdomain:
            features['subdomain_count'] = len(ext.subdomain.split('.'))
        
        # Path depth
        features['path_depth'] = len([p for p in parsed.path.split('/') if p])
        
        # Shorteners
        shorteners = ['bit.ly', 'goo.gl', 't.co', 'tinyurl', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly']
        if any(s in domain for s in shorteners):
            features['is_shortened'] = 1
            
        # Suspicious TLDs
        suspicious_tlds = ['xyz', 'top', 'club', 'online', 'site', 'vip', 'click', 'info', 'tk']
        if ext.suffix in suspicious_tlds:
            features['has_suspicious_tld'] = 1
            
    except Exception:
        return pd.Series(features)
        
    actual_domain = features['domain']
    if not actual_domain:
        return pd.Series(features)

    # DNS Resolution
    try:
        answers = dns.resolver.resolve(actual_domain, 'A')
        features['ip'] = str(answers[0])
    except Exception:
        features['ip'] = 'unknown_ip'

    # MX Records
    try:
        mx_answers = dns.resolver.resolve(actual_domain, 'MX')
        if len(mx_answers) > 0:
            features['has_mx_records'] = 1
    except Exception:
        features['has_mx_records'] = 0

    # WHOIS Lookup
    try:
        w = whois.whois(actual_domain)
        if w.creation_date:
            creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            age = (datetime.now() - creation_date).days
            features['domain_age_days'] = age
        if w.expiration_date:
            exp_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
            features['days_to_expiry'] = (exp_date - datetime.now()).days
        if w.registrar:
            features['registrar'] = str(w.registrar).split()[0].lower() # Simplify
    except Exception:
        pass
        
    # SSL Check
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=actual_domain) as s:
            s.settimeout(2.0)
            s.connect((actual_domain, 443))
            features['has_ssl'] = 1
    except Exception:
        features['has_ssl'] = 0
        
    # ASN Lookup (via IP)
    if features['ip'] != 'unknown_ip':
        try:
            resp = requests.get(f"http://ip-api.com/json/{features['ip']}?fields=as", timeout=2)
            if resp.status_code == 200:
                features['asn'] = resp.json().get('as', 'unknown_asn').split()[0]
        except Exception:
            features['asn'] = 'unknown_asn'
            
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
        
    print("Extracting rich URL & Infrastructure features. This will take a while due to DNS/WHOIS/ASN lookups...")
    tqdm.pandas(desc="Processing URLs")
    features_df = df['url'].progress_apply(extract_url_features)
    
    df = pd.concat([df, features_df], axis=1)
    
    # Save the giant flat CSV (preprocess.py will break it down into node types)
    df.to_csv(output_csv, index=False)
    print(f"Processed infrastructure data saved to {output_csv}")
    return df
