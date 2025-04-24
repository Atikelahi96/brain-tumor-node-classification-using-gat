import os
import random
import nibabel as nib
import numpy as np
from skimage.segmentation import slic
import networkx as nx
import torch
import psutil
from sklearn.neighbors import KDTree
from scipy.ndimage import binary_dilation
from scipy.stats import scoreatpercentile

def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    print(f"Memory usage at {stage}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

def load_mri_data_batch(data_dir, subdirectories, modalities, batch_size):
    print("Loading MRI data...")
    num_subjects = len(subdirectories)
    num_batches = (num_subjects + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_subjects)
        batch_subdirectories = subdirectories[start_idx:end_idx]

        batch_mri_data = []
        batch_ground_truth_labels = []

        for subdir in batch_subdirectories:
            mri_data = []
            ground_truth_labels = load_ground_truth_labels(data_dir, subdir)
            if ground_truth_labels is None:
                continue

            for modality in modalities:
                modality_path = os.path.join(data_dir, subdir, f'{subdir}_{modality}.nii.gz')
                modality_data = nib.load(modality_path).get_fdata()
                mri_data.append(modality_data)

            if len(mri_data) == len(modalities):
                batch_mri_data.append(mri_data)
                batch_ground_truth_labels.append(ground_truth_labels)

        batch_ground_truth_labels = np.array(batch_ground_truth_labels)

        yield batch_mri_data, batch_ground_truth_labels

def load_ground_truth_labels(data_dir, subdirectory):
    ground_truth_path = os.path.join(data_dir, subdirectory, f'{subdirectory}_seg.nii.gz')
    if os.path.exists(ground_truth_path):
        ground_truth_labels = nib.load(ground_truth_path).get_fdata()
        return ground_truth_labels
    else:
        print(f"Ground truth labels not found for {subdirectory}")
        return None

def preprocess_mri_data(mri_data, ground_truth_labels, visualize=False):
    print("Preprocessing MRI data...")
    preprocessed_data = []
    preprocessed_labels = []

    # Cropping and rescaling
    for idx, (mri, gt) in enumerate(zip(mri_data, ground_truth_labels)):
        preprocessed_modalities = []
        min_coords, max_coords = None, None
        for modality in mri:
            non_zero_coords = np.array(np.nonzero(modality))
            min_coords = np.min(non_zero_coords, axis=1)
            max_coords = np.max(non_zero_coords, axis=1)
            cropped_data = modality[min_coords[0]:max_coords[0],
                                    min_coords[1]:max_coords[1],
                                    min_coords[2]:max_coords[2]]

            rescaled_img = cropped_data / np.percentile(cropped_data, 99.5)
            preprocessed_modalities.append(rescaled_img)

        preprocessed_data.append(preprocessed_modalities)

        cropped_gt = gt[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2]]
        preprocessed_labels.append(cropped_gt)

    max_shape = np.max([modality.shape for sample in preprocessed_data for modality in sample], axis=0)

    padded_data = []
    padded_labels = []
    for sample, gt in zip(preprocessed_data, preprocessed_labels):
        padded_modalities = []
        for modality in sample:
            padded_modality = np.pad(modality, [(0, max_shape[i] - modality.shape[i]) for i in range(3)], mode='constant')
            padded_modalities.append(padded_modality)
        padded_data.append(padded_modalities)

        padded_gt = np.pad(gt, [(0, max_shape[i] - gt.shape[i]) for i in range(3)], mode='constant')
        padded_labels.append(padded_gt)

    modality_means = []
    modality_stds = []

    for i in range(len(mri_data[0])):
        modality_data = [sample[i] for sample in padded_data]
        non_zero_voxels = np.concatenate([modality[modality > 0] for modality in modality_data])
        modality_mean = np.mean(non_zero_voxels)
        modality_std = np.std(non_zero_voxels)
        modality_means.append(modality_mean)
        modality_stds.append(modality_std)

    for i, sample in enumerate(padded_data):
        for j, modality in enumerate(sample):
            normalized_img = (modality - modality_means[j]) / modality_stds[j]
            padded_data[i][j] = normalized_img

    return padded_data, padded_labels

def combine_modalities(all_mri_data):
    print("Combining MRI modalities...")
    combined_images = []
    for mri_data in all_mri_data:
        combined_img = np.stack(mri_data, axis=-1)
        combined_images.append(combined_img)
    return combined_images
