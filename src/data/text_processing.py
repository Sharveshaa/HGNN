import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s#]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def process_text_data(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return None
    
    if 'text' not in df.columns:
        print("Column 'text' not found in dataset.")
        return None
    
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove duplicates
    original_len = len(df)
    df.drop_duplicates(subset=['cleaned_text'], inplace=True)
    print(f"Removed {original_len - len(df)} duplicate text entries.")
    
    # Basic tokenization (split by space) - this will be refined later with HuggingFace
    df['tokens'] = df['cleaned_text'].apply(lambda x: x.split())
    
    df.to_csv(output_csv, index=False)
    print(f"Processed text data saved to {output_csv}")
    return df
