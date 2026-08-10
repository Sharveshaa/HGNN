import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Adjust path to import from src/graph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.construct_graph import build_hetero_graph
from models.hgnn import PhishingHGNN

def create_masks(data, node_type, train_ratio=0.8):
    num_nodes = data[node_type].num_nodes
    indices = torch.randperm(num_nodes)
    train_idx = indices[:int(train_ratio * num_nodes)]
    test_idx = indices[int(train_ratio * num_nodes):]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    data[node_type].train_mask = train_mask
    data[node_type].test_mask = test_mask

def train():
    print("Loading Graph...")
    data = build_hetero_graph()[0]
    
    # Normalize features
    for node_type in data.node_types:
        x = data[node_type].x
        data[node_type].x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    
    # Setup masks for Users, Posts, and URLs
    create_masks(data, 'user')
    create_masks(data, 'post')
    create_masks(data, 'url')

    # 2 output classes: Benign (0) and Phishing (1)
    model = PhishingHGNN(hidden_channels=16, out_channels=2, metadata=data.metadata())
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print("\nStarting Training...")
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        
        # Calculate loss for all 3 node types
        loss_user = F.cross_entropy(out['user'][data['user'].train_mask], data['user'].y[data['user'].train_mask])
        loss_post = F.cross_entropy(out['post'][data['post'].train_mask], data['post'].y[data['post'].train_mask])
        loss_url = F.cross_entropy(out['url'][data['url'].train_mask], data['url'].y[data['url'].train_mask])
        
        # Total loss
        loss = loss_user + loss_post + loss_url
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Total Loss: {loss:.4f} (User: {loss_user:.4f}, Post: {loss_post:.4f}, URL: {loss_url:.4f})')
            
    print("\nEvaluating Model...")
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        
        for node_type in ['user', 'post', 'url']:
            pred = out[node_type].argmax(dim=-1)
            test_mask = data[node_type].test_mask
            correct = (pred[test_mask] == data[node_type].y[test_mask]).sum()
            acc = int(correct) / int(test_mask.sum())
            print(f'{node_type.capitalize()} Test Accuracy: {acc:.4f}')
            
    # Save Model
    model_path = os.path.join(os.path.dirname(__file__), 'hgnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train()
