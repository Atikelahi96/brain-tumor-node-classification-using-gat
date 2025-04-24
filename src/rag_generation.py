import os
import pickle
import numpy as np
import networkx as nx
from scipy.ndimage import binary_dilation
from sklearn.neighbors import KDTree

# Import preprocessing and feature functions
from data_preprocessing import print_memory_usage
from segmentation_and_features import perform_superpixel_segmentation, extract_features_binary_detection

# ----------------------------------------
# Edge Weight & Adjacency Helper Functions
# ----------------------------------------

def compute_edge_weight_vectorized(segment_features, neighbor_features, weight_type='absolute'):
    if weight_type == 'absolute':
        return np.abs(segment_features - neighbor_features).sum(axis=1)
    elif weight_type == 'squared':
        return np.square(segment_features - neighbor_features).sum(axis=1)
    else:
        raise ValueError("Invalid weight_type. Choose 'absolute' or 'squared'.")


def get_adjacent_segments_optimized(segment_label, segment_labels, radius=1):
    mask = (segment_labels == segment_label)
    struct = np.ones((2*radius+1,)*3, dtype=bool)
    dil = binary_dilation(mask, structure=struct)
    neigh = np.unique(segment_labels[dil])
    return neigh[neigh != segment_label]

# -------------------------
# RAG Generation
# -------------------------

def generate_rag_optimized(batch_features, batch_segment_labels, superpixel_segmentations,
                            batch_idx, save_dir, superpixel_labels=None,
                            weight_type='absolute', radius=1):
    print(f"Generating RAG for batch {batch_idx}...")
    print_memory_usage("before RAG generation")

    rag = nx.Graph()
    label_map = {lbl: idx for idx, lbl in enumerate(batch_segment_labels)}
    current_index = 0

    # Iterate over each image's superpixels
    for segs in superpixel_segmentations:
        unique_segs = np.unique(segs)
        n = len(unique_segs)
        feats = np.array(batch_features[current_index:current_index + n])
        labs = None
        if superpixel_labels is not None:
            labs = np.array(superpixel_labels[current_index:current_index + n])

        # Add nodes
        for i, seg_id in enumerate(unique_segs):
            node_idx = current_index + i
            if labs is not None:
                rag.add_node(node_idx, features=feats[i], label=int(labs[i]))
            else:
                rag.add_node(node_idx, features=feats[i])

        # Add edges based on spatial adjacency
        for i, seg_id in enumerate(unique_segs):
            node_idx = current_index + i
            neigh_ids = get_adjacent_segments_optimized(seg_id, segs, radius)
            neigh_idxs = [label_map[nid] for nid in neigh_ids if nid in label_map]
            if neigh_idxs:
                neigh_feats = np.array([batch_features[j] for j in neigh_idxs])
                weights = compute_edge_weight_vectorized(feats[i].reshape(1, -1), neigh_feats, weight_type)
                for j, w in zip(neigh_idxs, weights):
                    if not rag.has_edge(node_idx, j):
                        rag.add_edge(node_idx, j, weight=float(w))

        current_index += n

    # Summary
    nodes, edges = rag.number_of_nodes(), rag.number_of_edges()
    density = nx.density(rag)
    print(f"Batch {batch_idx} RAG: {nodes} nodes, {edges} edges, density={density:.4f}")
    print_memory_usage("after RAG generation")

    # Save to file
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"rag_batch_{batch_idx}.gpickle")
    with open(filepath, 'wb') as f:
        pickle.dump(rag, f)
    print(f"Saved RAG to {filepath}")
    return rag

# -------------------------
# Main Execution
# -------------------------
if __name__ == '__main__':
    import os
    import random
    from data_preprocessing import load_mri_data_batch, preprocess_mri_data, combine_modalities

    # Configuration
    data_dir = 'dataset'
    subs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    random.shuffle(subs)
    modalities = ['t1','t1ce','t2','flair']
    batch_size = 16
    save_dir = 'results/rags'

    # Process batches
    for batch_idx, (mri_batch, gt_batch) in enumerate(load_mri_data_batch(data_dir, subs, modalities, batch_size)):
        pre_data, pre_labels = preprocess_mri_data(mri_batch, gt_batch)
        combined = combine_modalities(pre_data)
        segs = perform_superpixel_segmentation(combined)
        feats, seg_labels, sp_labels, segs = extract_features_binary_detection(combined, pre_labels, segs, is_training=True)
        generate_rag_optimized(feats, seg_labels, segs, batch_idx, save_dir, superpixel_labels=sp_labels)
