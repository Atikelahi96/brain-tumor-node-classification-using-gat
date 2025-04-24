import os
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors  # k-NN for synthetic edges

# Improved GAT model with multiple layers, residual connections, and LeakyReLU activation
class ImprovedGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.3):
        super(ImprovedGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Initial GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Intermediate GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Final GAT layer (to output classes)
        self.gat_final = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = self.leaky_relu(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)

        # Intermediate GAT layers with Leaky ReLU and residual connections
        for gat_layer, batch_norm in zip(self.gat_layers, self.batch_norms):
            residual = x  # Residual connection
            x = self.leaky_relu(batch_norm(gat_layer(x, edge_index)))
            x = self.dropout(x) + residual  # Add the residual connection after applying dropout

        # Final GAT layer (output layer)
        x = self.gat_final(x, edge_index)
        return x

# Load graphs from RAG files

def load_graphs(graph_dir):
    graphs, labels = [], []
    for file in os.listdir(graph_dir):
        if file.endswith(".gpickle"):
            with open(os.path.join(graph_dir, file), 'rb') as f:
                g = pickle.load(f)
            graphs.append(g)
            labels.append([d[1]['label'] for d in g.nodes(data=True)])
    return graphs, labels

# Convert networkx graph + features/labels to PyTorch Geometric Data

def graph_to_pytorch_data(graph, features, labels):
    edges = np.array(list(graph.edges)).T
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# Advanced Graph-SMOTE: oversample node features & labels, enhance via GraphSAGE, add k-NN edges

def advanced_graph_smote(graphs, labels, k=5):
    smote = SMOTE()
    synthetic_graphs = []
    all_smote_features = []
    all_smote_labels = []

    for graph, label in zip(graphs, labels):
        features = [graph.nodes[node]['features'] for node in graph.nodes]
        features = np.array(features)
        label = np.array(label)

        # Apply SMOTE to each graph separately
        smote_features, smote_labels = smote.fit_resample(features, label)

        # Collect resampled features and labels
        all_smote_features.extend(smote_features)
        all_smote_labels.extend(smote_labels)

     
        
        # Create a synthetic graph with the SMOTE features
        synthetic_graph = graph.copy()
        for i, node in enumerate(synthetic_graph.nodes):
            synthetic_graph.nodes[node]['features'] = smote_features[i]
            synthetic_graph.nodes[node]['label'] = smote_labels[i]

        # Add k-NN edges for synthetic nodes
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(smote_features)
        knn_edges = nbrs.kneighbors_graph(smote_features).tocoo()
        
        for i, j in zip(knn_edges.row, knn_edges.col):
            synthetic_graph.add_edge(i, j)

        synthetic_graphs.append(synthetic_graph)

    # Convert lists to numpy arrays
    all_smote_features = np.array(all_smote_features)
    all_smote_labels = np.array(all_smote_labels)

    return all_smote_features, all_smote_labels, synthetic_graphs

# Prepare train/test split of Data objects

def prepare_data(graphs, feats, labs, test_size=0.2):
    data_list, idx = [], 0
    for g in graphs:
        n = len(g.nodes)
        subset_feats = feats[idx:idx+n]
        subset_labs = labs[idx:idx+n]
        data = graph_to_pytorch_data(g, subset_feats, subset_labs)
        data_list.append(data)
        idx += n
    return train_test_split(data_list, test_size=test_size)

# Training loop

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1).cpu().numpy()
        lbl = data.y.cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(lbl)

    print(f"Train Distribution: {Counter(all_labels)}, Pred: {Counter(all_preds)}")
    return total_loss/len(loader), all_preds, all_labels

# Evaluation loop

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1).cpu().numpy()
            lbl = data.y.cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(lbl)
    print(f"Test Distribution: {Counter(all_labels)}, Pred: {Counter(all_preds)}")
    return total_loss/len(loader), all_preds, all_labels

# Metrics

def calculate_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

# Main execution

if __name__ == '__main__':
    graph_dir = 'results/rags'
    graphs, labels = load_graphs(graph_dir)

    feats, labs, graphs_sm = advanced_graph_smote(graphs, labels)
    train_data, test_data = prepare_data(graphs_sm, feats, labs)

    # Hyperparameters
    in_ch = feats.shape[1]
    hid_ch = 32
    out_ch = len(np.unique(labs))
    epochs, bs, lr = 15, 8, 0.001

    model = ImprovedGATModel(in_channels=in_ch, hidden_channels=hid_ch, out_channels=out_ch)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=bs)

    best_loss = float('inf')
    for ep in range(epochs):
        tr_loss, tr_p, tr_l = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_p, te_l = eval_epoch(model, test_loader, criterion)
        tr_acc, tr_f1 = calculate_metrics(tr_p, tr_l)
        te_acc, te_f1 = calculate_metrics(te_p, te_l)
        print(f"Epoch {ep+1}/{epochs} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, F1: {tr_f1:.4f} |" \
              f" Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}, F1: {te_f1:.4f}")
        if te_loss < best_loss:
            best_loss = te_loss
            torch.save(model.state_dict(), 'best_gat_model.pth')
            print(f"Saved best model epoch {ep+1}")
