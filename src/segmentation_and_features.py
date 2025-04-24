import numpy as np
from skimage.segmentation import slic
from scipy.stats import scoreatpercentile

def perform_superpixel_segmentation(combined_images, n_segments=500, compactness=.1, sigma=1):
    print("Performing superpixel segmentation using SLIC...")
    superpixel_segmentations = []
    for idx, combined_image in enumerate(combined_images):
        segments_slic = slic(combined_image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        superpixel_segmentations.append(segments_slic)
        
    return superpixel_segmentations

def extract_features_binary_detection(combined_images, padded_ground_truth_labels, superpixel_segmentations, is_training):
    print("Extracting features for binary detection from superpixels...")
    print_memory_usage("start of feature extraction")

    all_features = []
    all_segment_labels = []
    all_superpixel_labels = []

    for idx, (image, ground_truth_labels, superpixel_seg) in enumerate(zip(combined_images, padded_ground_truth_labels, superpixel_segmentations), start=1):
        unique_segments = np.unique(superpixel_seg)

        for segment_label in unique_segments:
            segment_mask = (superpixel_seg == segment_label)
            segment_features = []

            for modality in range(image.shape[-1]):
                modality_values = image[..., modality][segment_mask]
                percentile_features = [scoreatpercentile(modality_values, perc) for perc in [10, 25, 50, 75, 90]]
                segment_features.extend(percentile_features)

            all_features.append(segment_features)
            all_segment_labels.append(segment_label)

            if is_training:
                segment_gt_labels = ground_truth_labels[segment_mask]

                if np.any(np.isin(segment_gt_labels, [1, 2, 4])):
                    tumor_label = 1
                else:
                    tumor_label = 0

                all_superpixel_labels.append(tumor_label)

        
    print_memory_usage("end of feature extraction")
    return all_features, all_segment_labels, all_superpixel_labels, superpixel_segmentations