import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.construct_graph import build_hetero_graph
from models.hgnn import PhishingHGNN

def plot_embeddings(embeddings, labels, node_type, method='TSNE'):
    print(f"Generating {method} plot for {node_type} nodes...")
    
    if method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'TSNE' or 'UMAP'")
        
    reduced_embeds = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # 0 is Benign, 1 is Phishing
    class_names = ['Benign', 'Phishing']
    palette = sns.color_palette("husl", 2)
    
    scatter = sns.scatterplot(
        x=reduced_embeds[:, 0], y=reduced_embeds[:, 1],
        hue=[class_names[l] for l in labels],
        palette=palette,
        s=100, alpha=0.7
    )
    
    plt.title(f'{method} Visualization of {node_type.capitalize()} Node Embeddings')
    plt.xlabel(f'{method} Dimension 1')
    plt.ylabel(f'{method} Dimension 2')
    
    # Save the plot
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{node_type}_{method.lower()}.png')
    plt.savefig(file_path, bbox_inches='tight')
    print(f"Saved {method} plot to {file_path}")
    plt.close()

def main():
    print("Loading Graph...")
    res = build_hetero_graph()
    data = res[0]
    
    # Normalize features (same as in training)
    for node_type in data.node_types:
        x = data[node_type].x
        data[node_type].x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        
    print("Loading Model...")
    model_path = os.path.join(os.path.dirname(__file__), 'hgnn.pth')
    model = PhishingHGNN(hidden_channels=16, out_channels=2, metadata=data.metadata())
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
        
    print("Extracting Embeddings...")
    with torch.no_grad():
        out_dict, embeds_dict = model(data.x_dict, data.edge_index_dict, return_embeds=True)
        
    for node_type in ['user', 'post', 'url']:
        embeddings = embeds_dict[node_type].cpu().numpy()
        labels = data[node_type].y.cpu().numpy()
        
        plot_embeddings(embeddings, labels, node_type, method='TSNE')
        plot_embeddings(embeddings, labels, node_type, method='UMAP')

if __name__ == "__main__":
    main()
