import os
import pickle
import numpy as np
import networkx as nx
from scipy.ndimage import binary_dilation

# Import functions from local modules
from data-preprocess import print_memory_usage, load_mri_data_batch, preprocess_mri_data, combine_modalities
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


def get_adjacent_segments_optimized(segment_label, segmentation, radius=1):
    mask = (segmentation == segment_label)
    struct = np.ones((2*radius+1,)*3, dtype=bool)
    dil = binary_dilation(mask, structure=struct)
    neigh = np.unique(segmentation[dil])
    return neigh[neigh != segment_label]

# ----------------------------------------
# RAG Generation
# ----------------------------------------

def generate_rag_optimized(features, segment_labels, superpixel_segmentations,
                            batch_idx, save_dir, sp_labels=None,
                            weight_type='absolute', radius=1):
    """
    Builds and saves a Region Adjacency Graph for one batch.
    - features: list of feature vectors for all superpixels across the batch
    - segment_labels: corresponding superpixel IDs
    - superpixel_segmentations: list of 2D/3D label maps per image
    - sp_labels: (optional) list of binary labels per superpixel
    """
    print(f"Generating RAG for batch {batch_idx}...")
    print_memory_usage("before RAG generation")

    rag = nx.Graph()
    label_map = {lbl: idx for idx, lbl in enumerate(segment_labels)}
    idx_ptr = 0

    # Add nodes
    for seg_map in superpixel_segmentations:
        unique_segs = np.unique(seg_map)
        n = len(unique_segs)
        feats = np.array(features[idx_ptr:idx_ptr+n])
        labs = None
        if sp_labels is not None:
            labs = np.array(sp_labels[idx_ptr:idx_ptr+n])

        for i, seg_id in enumerate(unique_segs):
            node_idx = idx_ptr + i
            if labs is not None:
                rag.add_node(node_idx, features=feats[i], label=int(labs[i]))
            else:
                rag.add_node(node_idx, features=feats[i])

        idx_ptr += n

    # Add edges using spatial adjacency
    idx_ptr = 0
    for seg_map in superpixel_segmentations:
        unique_segs = np.unique(seg_map)
        for seg_id in unique_segs:
            i = label_map[seg_id]
            neigh_ids = get_adjacent_segments_optimized(seg_id, seg_map, radius)
            neigh_idxs = [label_map[nid] for nid in neigh_ids if nid in label_map]
            if neigh_idxs:
                f_i = np.array(features[i]).reshape(1, -1)
                neigh_feats = np.array([features[j] for j in neigh_idxs])
                weights = compute_edge_weight_vectorized(f_i, neigh_feats, weight_type)
                for j, w in zip(neigh_idxs, weights):
                    if not rag.has_edge(i, j):
                        rag.add_edge(i, j, weight=float(w))
        idx_ptr += len(unique_segs)

    # Log and save
    n_nodes, n_edges = rag.number_of_nodes(), rag.number_of_edges()
    print(f"Batch {batch_idx} RAG: {n_nodes} nodes, {n_edges} edges, density={nx.density(rag):.4f}")
    print_memory_usage("after RAG generation")

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"rag_batch_{batch_idx}.gpickle")
    with open(filepath, 'wb') as f:
        pickle.dump(rag, f)
    print(f"Saved RAG to {filepath}")
    return rag

# ----------------------------------------
# Main Execution
# ----------------------------------------
if __name__ == '__main__':
    # Configuration
    data_dir = 'dataset'
    subs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    random.shuffle(subs)
    modalities = ['t1', 't1ce', 't2', 'flair']
    batch_size = 16
    save_dir = 'results/rags'

    # Generate RAGs
    for batch_idx, (mri_batch, gt_batch) in enumerate(
            load_mri_data_batch(data_dir, subs, modalities, batch_size)):
        pre_data, pre_labels = preprocess_mri_data(mri_batch, gt_batch)
        combined = combine_modalities(pre_data)
        segs = perform_superpixel_segmentation(combined)
        feats, seg_lbls, sp_lbls, _ = extract_features_binary_detection(
            combined, pre_labels, segs, is_training=True)
        generate_rag_optimized(feats, seg_lbls, segs, batch_idx, save_dir, sp_labels=sp_lbls)
