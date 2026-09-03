import sys
import os
import torch
import uvicorn
import random
import requests
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.construct_graph import build_hetero_graph
from models.hgnn import PhishingHGNN

# Load environment variables
load_dotenv()

# Global variables to hold model and graph
graph_data = None
model = None
node_mappings = {} 
raw_features = {}

FEATURE_NAMES = {
    'user': ['Followers Count', 'Following Count', 'Account Age (Days)'],
    'post': ['Text Length'],
    'url': ['URL Length', 'Entropy', 'Is Shortened', 'Has Suspicious TLD', 'Path Depth', 'Subdomain Count'],
    'domain': ['Domain Age (Days)', 'Days to Expiry', 'Has MX Records', 'Has SSL'],
    'ip': ['Dummy'],
    'asn': ['Dummy'],
    'registrar': ['Dummy']
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_data, model, node_mappings
    print("Loading Graph for API...")
    
    # Build graph
    res = build_hetero_graph()
    graph_data = res[0]
    
    # Load mappings
    node_mappings['user_to_idx'] = res[1]
    node_mappings['post_to_idx'] = res[2]
    node_mappings['url_to_idx'] = res[3]
    if res[4] is not None:
        node_mappings['domain_to_idx'] = res[4]
        node_mappings['ip_to_idx'] = res[5]
        node_mappings['asn_to_idx'] = res[6]
        node_mappings['registrar_to_idx'] = res[7]
    
    # Store raw features before normalization
    for node_type in graph_data.node_types:
        raw_features[node_type] = graph_data[node_type].x.clone().tolist()
        
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
        
    out = {}
    for k, v in node_mappings.items():
        out[k.replace('_to_idx', 's')] = sample_dict(v)
    return out

@app.get("/api/batch_predict")
def batch_predict(node_type: str):
    global graph_data, model, node_mappings
    
    if node_type not in ['user', 'post', 'url']:
        return JSONResponse(status_code=400, content={"error": "Invalid node type"})
        
    mapping_key = f"{node_type}_to_idx"
    nodes = list(node_mappings[mapping_key].keys())
    
    if len(nodes) > 50:
        nodes = random.sample(nodes, 50)
        
    # Forward pass without gradients for speed
    with torch.no_grad():
        out = model(graph_data.x_dict, graph_data.edge_index_dict)
        logits = out[node_type]
        probs = torch.softmax(logits, dim=1)
    
    benign_nodes = []
    malicious_nodes = []
    
    for node_id in nodes:
        idx = node_mappings[mapping_key][node_id]
        node_probs = probs[idx]
        phishing_prob = float(node_probs[1])
        if phishing_prob > 0.5:
            malicious_nodes.append({"id": node_id, "confidence": phishing_prob})
        else:
            benign_nodes.append({"id": node_id, "confidence": float(node_probs[0])})
            
    malicious_nodes.sort(key=lambda x: x["confidence"], reverse=True)
    benign_nodes.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "node_type": node_type,
        "malicious": malicious_nodes,
        "benign": benign_nodes
    }

@app.get("/api/graph_data")
def get_graph_data():
    global graph_data, node_mappings
    
    nodes = []
    edges = []
    
    # Invert mappings
    idx_to_id = {}
    for ntype in graph_data.node_types:
        mapping = node_mappings[f"{ntype}_to_idx"]
        inv_map = {v: k for k, v in mapping.items()}
        idx_to_id[ntype] = inv_map
        
        # Add nodes
        for idx in inv_map.keys():
            node_id = inv_map[idx]
            nodes.append({
                "id": f"{ntype}_{idx}",
                "label": node_id,
                "group": ntype
            })
            
    # Add edges
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()
        
        for src, dst in zip(src_indices, dst_indices):
            # Only add edge if both nodes exist (sanity check)
            if src in idx_to_id[src_type] and dst in idx_to_id[dst_type]:
                edges.append({
                    "from": f"{src_type}_{src}",
                    "to": f"{dst_type}_{dst}",
                    "label": rel
                })
            
    return {"nodes": nodes, "edges": edges}

@app.get("/api/subgraph")
def get_subgraph(node_type: str, node_id: str):
    global graph_data, model, node_mappings
    
    if node_type not in ['user', 'post', 'url']:
        return JSONResponse(status_code=400, content={"error": "Invalid node type"})
        
    mapping_key = f"{node_type}_to_idx"
    if node_id not in node_mappings[mapping_key]:
        return JSONResponse(status_code=404, content={"error": f"Node ID {node_id} not found"})
        
    target_idx = node_mappings[mapping_key][node_id]
    
    import torch
    
    # 1. Enable gradients for XAI
    for ntype in graph_data.node_types:
        graph_data[ntype].x.requires_grad = True
        
    # 2. Forward pass
    out = model(graph_data.x_dict, graph_data.edge_index_dict)
    logits = out[node_type][target_idx]
    probs = torch.softmax(logits, dim=0)
    pred_class = 1 if float(probs[1]) > 0.5 else 0
    
    # 3. Backward pass
    model.zero_grad()
    logits[pred_class].backward()
    
    # 4. Collect saliency scores for all nodes
    node_saliency = {}
    for ntype in graph_data.node_types:
        if graph_data[ntype].x.grad is not None:
            grads = graph_data[ntype].x.grad.abs().sum(dim=1)
            for i, score in enumerate(grads.tolist()):
                node_saliency[(ntype, i)] = score
                
    edges_list = {et: graph_data[et].edge_index.tolist() for et in graph_data.edge_types}
    
    # Build adjacency list for undirected exploration
    adj = {}
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        u_list, v_list = edges_list[edge_type]
        for u, v in zip(u_list, v_list):
            adj.setdefault((src_type, u), []).append((dst_type, v))
            adj.setdefault((dst_type, v), []).append((src_type, u))
            
    # 5. DFS guided by gradient saliency
    visited = set()
    visited.add((node_type, target_idx))
    
    def dfs(node):
        if len(visited) >= 50:
            return
        neighbors = adj.get(node, [])
        neighbors = [n for n in neighbors if n not in visited]
        # Sort neighbors by their saliency (highest first)
        neighbors.sort(key=lambda n: node_saliency.get(n, 0.0), reverse=True)
        
        for n in neighbors:
            if len(visited) >= 50:
                break
            # Only explore paths that actually contributed (non-zero gradient)
            if node_saliency.get(n, 0.0) > 1e-6:
                visited.add(n)
                dfs(n)
                
    dfs((node_type, target_idx))
        
    # Format for UI
    idx_to_id = {}
    for ntype in graph_data.node_types:
        idx_to_id[ntype] = {v: k for k, v in node_mappings[f"{ntype}_to_idx"].items()}
        
    nodes = []
    for ntype, idx in visited:
        logits = out[ntype][idx]
        probs = torch.softmax(logits, dim=0)
        risk_score = float(probs[1]) * 100
        status = "Malicious" if risk_score > 50 else "Benign"
        
        feat_names = FEATURE_NAMES[ntype]
        feat_vals = raw_features[ntype][idx]
        features_dict = {name: float(val) for name, val in zip(feat_names, feat_vals)}
        
        nodes.append({
            "id": f"{ntype}_{idx}",
            "label": idx_to_id[ntype][idx],
            "group": ntype,
            "risk_score": risk_score,
            "status": status,
            "features": features_dict
        })
        
    edges = []
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        u_list, v_list = edges_list[edge_type]
        for u, v in zip(u_list, v_list):
            if (src_type, u) in visited and (dst_type, v) in visited:
                edges.append({
                    "from": f"{src_type}_{u}",
                    "to": f"{dst_type}_{v}",
                    "label": rel
                })
                
    return {"nodes": nodes, "edges": edges}

def generate_dynamic_xai(feature_data, prediction, confidence):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {item["feature"]: "API key not set. Skipping dynamic explanation." for item in feature_data}
        
    prompt = f"You are an AI explainer for a Phishing Detection Graph Neural Network (HGNN). The model predicted this node as '{prediction}' (Confidence: {confidence:.2f}).\n"
    prompt += "Note on Graph Connections: A connection to a Malicious node indicates guilt by association. If a Malicious node connects to a Benign node, it indicates the malicious account is likely targeting or mimicking the benign user to build false legitimacy and evade detection.\n"
    prompt += "CRITICAL INSTRUCTION: Do NOT explain each feature in isolation. You MUST look at all the features provided below as a whole, and explain each feature in the context of the others. For example, if a user has a high 'Following Count' but a low 'Account Age', use that impossible velocity to explain both features as evidence of botting. Connect the dots for the reader so they understand how the combination of these features proves the classification.\n"
    prompt += "Here are the top features, their actual values for this node, and their contribution to the decision:\n"
    for item in feature_data:
        val = item.get('value', 'N/A')
        if isinstance(val, float) and not val.is_integer():
            val = f"{val:.2f}"
        prompt += f"- {item['feature']}: {val} (Contribution: {item['importance']*100:.1f}%)\n"
        
    prompt += "\nFor each feature, provide a 1-2 sentence explanation of why it matters for this prediction. CRITICAL RULES:\n"
    prompt += "1. Tell the analyst WHY a connection matters (e.g. 'User U001 is predicted Malicious, increasing the likelihood of coordinated spam').\n"
    prompt += "2. Compare related features instead of explaining them in isolation (e.g. Followers vs Following ratio).\n"
    prompt += "3. If explaining a graph connection, use graph terminology like 'meta-path' or 'shared interaction patterns' (e.g. 'The meta-path User -> Post -> URL strongly contributed because...').\n"
    prompt += "4. State exactly how much it contributed (e.g., 'This feature contributed X%...').\n"
    prompt += "5. If a connected URL is a well-known legitimate site (like ineos.com), do not call the domain itself phishing; explain its co-occurrence with suspicious entities.\n"
    prompt += "Return ONLY a valid JSON object mapping the exact feature name to its explanation. Do NOT include markdown formatting or backticks (e.g. no ```json)."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen/qwen3.6-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=8)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"LLM API Error: {e}\nResponse: {e.response.text}")
        else:
            print(f"LLM API Error: {e}")
        return {item["feature"]: f"Dynamic explanation unavailable." for item in feature_data}

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
    pred_text = "Malicious" if pred_class == 1 else "Benign"
    conf_val = max(phishing_prob, benign_prob)
    
    # 3. Backward pass to get saliency (gradients)
    model.zero_grad()
    logits[pred_class].backward()
    
    # 4. Extract feature importance for the target node
    grad = graph_data[node_type].x.grad[idx].abs()
    
    names = FEATURE_NAMES[node_type]
    raw_vals = raw_features[node_type][idx]
    
    local_features = []
    for name, score, val in zip(names, grad.tolist(), raw_vals):
        local_features.append({"feature": name, "grad": float(score), "value": float(val)})
        
    # 5. Extract neighborhood importance (Guilt by Association)
    neighbor_features = []
    idx_to_id = {}
    for ntype in graph_data.node_types:
        inv_map = {v: k for k, v in node_mappings[f"{ntype}_to_idx"].items()}
        idx_to_id[ntype] = inv_map
        
        if graph_data[ntype].x.grad is None:
            continue
            
        all_grads = graph_data[ntype].x.grad.abs().sum(dim=1)
        if ntype == node_type:
            all_grads[idx] = 0.0 # Exclude the target node itself
            
        top_k = min(2, len(all_grads))
        if top_k > 0:
            top_vals, top_indices = torch.topk(all_grads, top_k)
            for val, n_idx in zip(top_vals.tolist(), top_indices.tolist()):
                if val > 1e-6: # Only include if it actually contributed
                    original_id = idx_to_id[ntype][n_idx]
                    
                    n_logits = out[ntype][n_idx]
                    n_probs = torch.softmax(n_logits, dim=0)
                    n_phishing_prob = float(n_probs[1])
                    n_status = "Malicious" if n_phishing_prob > 0.5 else "Benign"
                    n_conf = max(n_phishing_prob, 1.0 - n_phishing_prob) * 100
                    
                    neighbor_features.append({
                        "feature": f"Connection to {ntype.capitalize()} {original_id}",
                        "grad": float(val),
                        "value": f"Linked (Neighbor is predicted {n_status} with {n_conf:.1f}% confidence)"
                    })
                    
    # Combine and normalize
    all_combined = local_features + neighbor_features
    total_grad = sum(item["grad"] for item in all_combined) + 1e-9
    
    feature_importance = []
    for item in all_combined:
        item["importance"] = item["grad"] / total_grad
        del item["grad"]
        feature_importance.append(item)
    
    # Sort by importance descending
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    # Generate dynamic LLM explanations
    explanations_dict = generate_dynamic_xai(feature_importance, pred_text, conf_val)
    for item in feature_importance:
        item["description"] = explanations_dict.get(item["feature"], "Explanation not found.")
    
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
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
