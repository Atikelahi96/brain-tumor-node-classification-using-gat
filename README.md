# Brain Tumor Node Classification with GAT

## Project Overview
This repository implements a graph-based approach to classify brain tumor regions using MRI scans from the BraTS2021 dataset. It includes:
- MRI data loading and batch preprocessing
- Superpixel segmentation (SLIC)
- Feature extraction per superpixel
- Region Adjacency Graph (RAG) generation and saving
- Synthetic graph generation with SMOTE
- Graph Attention Network (GAT) model training and evaluation

## Dataset
Download the BraTS2021 Training Data from Kaggle:
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

Place each subject folder (e.g., `BraTS2021_Training_001`) into the `dataset/` directory.

## Repository Structure
```
brain_tumor_node_classification_complete/
├── dataset/
│   └── README.md
├── results/
├── src/
│   ├── data_preprocessing.py
│   ├── segmentation_and_features.py
│   ├── rag_generation.py
│   ├── synthetic_graph.py
│   └── train_gat.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Preprocess & Generate RAGs**  
   ```bash
   python src/data_preprocessing.py
   python src/segmentation_and_features.py
   python src/rag_generation.py
   ```

2. **Train & Evaluate GAT**  
   ```bash
   python src/train_gat.py
   ```
3. **Results**  
   Saved graphs, synthetic graphs, and model checkpoints/logs appear in `results/`.  
