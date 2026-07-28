import sys
import os
import torch
import uvicorn
import random
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.construct_graph import load_node_features, build_hetero_graph
from models.hgnn import PhishingHGNN

# Global variables to hold model and graph
graph_data = None
model = None
node_mappings = {} 

FEATURE_NAMES = {
    'user': ['Followers Count', 'Following Count', 'Account Age (Days)'],
    'post': ['Text Length'],
    'url': ['URL Length', 'Domain Age (Days)', 'Has SSL Certificate']
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_data, model, node_mappings
    print("Loading Graph for API...")
    
    # Load mappings
    user_x, user_y, user_mapping, post_x, post_y, post_mapping, url_x, url_y, url_mapping, posts_df = load_node_features()
    
    node_mappings['user_to_idx'] = user_mapping
    node_mappings['post_to_idx'] = post_mapping
    node_mappings['url_to_idx'] = url_mapping
    
    # Build graph
    graph_data = build_hetero_graph()
    
    # Normalize features (same as in train.py)
    for node_type in graph_data.node_types:
        x = graph_data[node_type].x
        graph_data[node_type].x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    
    # Load Model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'hgnn.pth'))
    model = PhishingHGNN(hidden_channels=16, out_channels=2, metadata=graph_data.metadata())
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded successfully from {model_path}.")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using uninitialized weights.")
        
    yield
    
    # Cleanup if needed
    print("Shutting down API...")

app = FastAPI(title="HGNN Phishing Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/nodes")
def get_nodes():
    # Return a sample of nodes for the UI
    def sample_dict(d, k=50):
        keys = list(d.keys())
        return random.sample(keys, min(k, len(keys)))
        
    return {
        "users": sample_dict(node_mappings['user_to_idx']),
        "posts": sample_dict(node_mappings['post_to_idx']),
        "urls": sample_dict(node_mappings['url_to_idx'])
    }

@app.get("/api/predict")
def predict(node_type: str, node_id: str):
    global graph_data, model, node_mappings
    
    if node_type not in ['user', 'post', 'url']:
        return JSONResponse(status_code=400, content={"error": "Invalid node type"})
        
    mapping_key = f"{node_type}_to_idx"
    if node_id not in node_mappings[mapping_key]:
        return JSONResponse(status_code=404, content={"error": f"Node ID {node_id} not found"})
        
    idx = node_mappings[mapping_key][node_id]
    
    # 1. Enable gradients for input features for XAI
    for ntype in graph_data.node_types:
        graph_data[ntype].x.requires_grad = True
    
    # 2. Forward pass
    out = model(graph_data.x_dict, graph_data.edge_index_dict)
    logits = out[node_type][idx]
    probs = torch.softmax(logits, dim=0)
    
    phishing_prob = float(probs[1])
    benign_prob = float(probs[0])
    pred_class = 1 if phishing_prob > 0.5 else 0
    
    # 3. Backward pass to get saliency (gradients)
    model.zero_grad()
    logits[pred_class].backward()
    
    # 4. Extract feature importance for the target node
    grad = graph_data[node_type].x.grad[idx].abs()
    
    # Normalize importance to sum to 1
    grad_sum = grad.sum() + 1e-9
    importance_scores = (grad / grad_sum).tolist()
    
    # Map to feature names
    names = FEATURE_NAMES[node_type]
    feature_importance = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(names, importance_scores)
    ]
    
    # Sort by importance descending
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    # Reset requires_grad
    for ntype in graph_data.node_types:
        graph_data[ntype].x.requires_grad = False
        
    return {
        "node_type": node_type,
        "node_id": node_id,
        "phishing_probability": phishing_prob,
        "benign_probability": benign_prob,
        "prediction": "Malicious" if phishing_prob > 0.5 else "Benign",
        "confidence": max(phishing_prob, benign_prob),
        "explanations": feature_importance
    }

# Mount static files for the frontend
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
