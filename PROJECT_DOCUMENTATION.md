# HGNN Phishing Detection System
## Complete Technical Specification & Implementation Guide

This document serves as the exhaustive architectural and operational blueprint for the Heterogeneous Graph Neural Network (HGNN) Phishing Detection System. By reading this document, a developer will understand the exact features extracted, the neural network architecture used, the XAI gradient math, and the end-to-end steps to build the project from scratch.

---

## 1. Project Overview & Philosophy
The HGNN Phishing Detection System is a multimodal, graph-based artificial intelligence application designed to identify malicious actors, posts, and URLs on a social network. 

Traditional ML models (like Random Forests or MLPs) evaluate entities in a vacuum (e.g., "Does this user have low followers?"). This system instead builds a **Heterogeneous Graph** (a graph with multiple node types and edge types). It uses PyTorch Geometric to propagate risk signals through the network using **Message Passing**. If a Benign user shares a Malicious URL, the URL's embedding flows backward into the Post, and then into the User, staining them by association.

---

## 2. Dataset Schema & Feature Extraction (`src/data/`)

The synthetic dataset mimics a real-world social network. The generation scripts (`generate_synthetic_data.py`, `generate_edges.py`) build the raw CSV files, which are then normalized by `preprocess.py`.

### A. Node Types & Features
1. **User Nodes (100 nodes)**:
   - `followers_count` (int): Number of followers. Malicious bots typically have very few.
   - `following_count` (int): Number of people followed. Bots often have heavily inflated following counts (farming).
   - `account_age_days` (int): Days since creation. Malicious accounts are often freshly minted (1-30 days).
2. **Post Nodes (200 nodes)**:
   - Text is processed to extract `length`, `contains_urgent_keywords`, and `sentiment_score`.
   - Phishing posts contain urgent keyword templates (e.g., "URGENT", "compromised", "verify").
3. **URL Nodes**:
   - Extracted features: `url_length`, `url_entropy`, `is_shortened`, `has_suspicious_tld`, `path_depth`, `subdomain_count`.
4. **Infrastructure Nodes**:
   - **Domain**: Features include `domain_age_days`, `days_to_expiry`, `has_mx_records`, `has_ssl`.
   - **IP**, **ASN**, and **Registrar**: These act as structural hubs to mathematically link coordinated phishing rings that share the same hosting or registration infrastructure.

### B. Edge Types (Topology)
The graph connects these nodes using seven distinct directed edge relationships:
1. `(user, follows, user)`: Captures social hierarchy.
2. `(user, shares, post)`: Captures content generation.
3. `(post, contains, url)`: Captures payload delivery.
4. `(url, hosted_on, domain)`: Connects payloads to their registered domain.
5. `(domain, resolves_to, ip)`: Connects domains to physical servers.
6. `(domain, registered_via, registrar)`: Groups domains by their registrar.
7. `(ip, belongs_to, asn)`: Groups servers into broad ISP/hosting subnetworks.

---

## 3. PyTorch Geometric Model Architecture (`src/models/hgnn.py`)

The neural network is built using PyTorch Geometric (PyG).

### The Architecture
1. **Base Homogeneous GNN**: 
   - Uses `SAGEConv` (GraphSAGE) layers. GraphSAGE aggregates features from a node's local neighborhood (mean aggregation) and concatenates them with the node's own features.
   - **Layer 1**: `SAGEConv((-1, -1), hidden_channels=64)` with a ReLU activation. The `(-1, -1)` enables lazy initialization based on input feature dimensions.
   - **Layer 2**: `SAGEConv((-1, -1), out_channels=2)`. This outputs a 2-dimensional tensor (Logits for Benign vs Malicious).
2. **Heterogeneous Conversion**:
   - The Base GNN is wrapped in `torch_geometric.nn.to_hetero(base_model, metadata, aggr='mean')`.
   - This automatically clones the `SAGEConv` layers for every specific edge type (e.g., a separate weight matrix for `user-shares-post` vs `post-contains-url`). The signals from different edge types are then averaged (`aggr='mean'`) before being passed to the node.

---

## 4. Backend API & Explainable AI (XAI) (`src/api/app.py`)

The backend is a FastAPI server that mounts the PyTorch model into memory and serves predictions. The most complex workflow in the backend is the XAI engine.

### How XAI Works (Feature Saliency)
When the `/api/predict` endpoint is hit:
1. **Gradient Computation**: The server enables `requires_grad=True` on the input feature tensors.
2. **Forward Pass**: The model predicts the node's class.
3. **Backward Pass**: The server calculates the gradients of the predicted class with respect to the input features and the connected neighbor nodes (`loss.backward()`).
4. **Saliency Scoring**: The absolute values of the gradients (`x.grad.abs()`) are extracted. A high gradient means that slightly changing that feature would drastically change the model's prediction. The gradients are normalized into percentages (e.g., Account Age: 40%, Follows U0002: 15%).
5. **LLM Translation**: The top 5 features and their percentages are formatted into a strict prompt. The prompt is sent to the **Groq Llama 3 API**, which converts the mathematical gradients into a human-readable explanation.
   - *Note*: The LLM prompt contains specific domain knowledge instructions (e.g., "A connection to a Benign user indicates the malicious account is farming legitimate connections to evade detection").

### Local Subgraph Extraction
The `/api/subgraph` endpoint extracts a 4-hop local graph around the target node using Breadth-First Search (BFS). 
- It maintains a hard limit of 50 nodes (checked inside the innermost BFS loop) to prevent the frontend UI from rendering a massive "hairball", while remaining deep enough to traverse into the physical infrastructure nodes (Domain, IP, ASN).
- It runs a lightweight forward pass to inject **Risk Scores** into every neighbor node payload, allowing the frontend to color-code the cluster.

---

## 5. Frontend Visualizations (`src/api/static/index.html`)

The UI is built purely with HTML, Vanilla JS, CSS, and `vis-network`.

### Core Features
- **Global Graph Explorer**: Renders all generated nodes. Users (Blue), Posts (Purple), URLs (Green), Domains (Orange Hexagons), IPs (Pink Diamonds), ASNs (Purple Stars), Registrars (Slate Squares).
- **Batch Analysis**: Fetches `/api/batch_predict` and filters for all nodes where the Phishing Probability > 50%.
- **XAI Modal**: 
  - **Dynamic Explanations**: Renders the LLM text alongside progress bars representing the gradient percentages.
  - **Insightful Subgraph**: Renders the 25-node local neighborhood. The target node is mathematically pinned to `(0, 0)` forcing the physics engine to arrange the neighbors in a beautiful orbital star. Neighbors are colored Red (Malicious) or Green (Benign) based on the backend risk scores. Rich HTML tooltips show raw metrics on hover.

---

## 6. End-to-End Build Guide

Follow these exact steps to rebuild and run the project:

### Step 1: Environment Setup
```bash
# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies (requires torch, torch_geometric, fastapi, pandas, groq)
pip install -r requirements.txt
```

### Step 2: Data Generation
Generate the synthetic network and extract multimodal features into numerical matrices.
```bash
python src/data/generate_synthetic_data.py
python src/data/generate_edges.py
python src/data/preprocess.py
```
*Expected Output*: The `data/processed/` directory will now contain `.npy` files for `x_dict` and `edge_index_dict`.

### Step 3: Model Training
Train the PyTorch Geometric HeteroConv model.
```bash
python src/models/train.py
```
*Expected Output*: The model trains for ~200 epochs and saves the weights to `src/models/hgnn.pth`.

### Step 4: Run the Application
Start the FastAPI backend server. Ensure you have your Groq API key set for the XAI feature.
```bash
# Export the Groq API key (Required for XAI explanations)
export GROQ_API_KEY="gsk_your_key_here"

# Start the server on port 8000
python src/api/app.py
```

### Step 5: Explore the UI
Open your browser and navigate to `http://localhost:8000`. Click "Analyze Nodes", select any node marked as "Malicious", and watch the XAI gradient calculations, LLM text generation, and local subgraph rendering occur in real-time.
