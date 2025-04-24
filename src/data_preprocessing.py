import os
import psutil
import random
import nibabel as nib
import numpy as np

def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    print(f"Memory usage at {stage}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

def load_mri_data_batch(data_dir, subdirectories, modalities, batch_size):
    print("Loading MRI data...")
    num_subjects = len(subdirectories)
    num_batches = (num_subjects + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start, end = batch_idx*batch_size, min((batch_idx+1)*batch_size, num_subjects)
        batch = subdirectories[start:end]
        batch_data, batch_labels = [], []
        for sub in batch:
            gt = load_ground_truth_labels(data_dir, sub)
            if gt is None: continue
            imgs = []
            for m in modalities:
                path = os.path.join(data_dir, sub, f'{sub}_{m}.nii.gz')
                imgs.append(nib.load(path).get_fdata())
            if len(imgs)==len(modalities):
                batch_data.append(imgs)
                batch_labels.append(gt)
        yield batch_idx, batch_data, np.array(batch_labels)

def load_ground_truth_labels(data_dir, subdirectory):
    path = os.path.join(data_dir, subdirectory, f'{subdirectory}_seg.nii.gz')
    if os.path.exists(path):
        return nib.load(path).get_fdata()
    else:
        print(f"Missing GT for {subdirectory}")
        return None

def preprocess_mri_data(mri_data, ground_truth_labels):
    print("Preprocessing MRI data...")
    pre_data, pre_labels = [], []
    for imgs, gt in zip(mri_data, ground_truth_labels):
        min_c, max_c = None, None
        modalities = []
        for img in imgs:
            nz = np.array(np.nonzero(img))
            low, high = np.min(nz,1), np.max(nz,1)
            cropped = img[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            scaled = cropped / np.percentile(cropped, 99.5)
            modalities.append(scaled)
            min_c, max_c = low, high
        pre_data.append(modalities)
        gt_crop = gt[min_c[0]:max_c[0], low[1]:high[1], low[2]:high[2]]
        pre_labels.append(gt_crop)
    # Padding to same shape
    shapes = [m.shape for sample in pre_data for m in sample]
    max_shape = np.max(shapes, axis=0)
    padded_data, padded_labels = [], []
    for sample, gt in zip(pre_data, pre_labels):
        pd_modalities = [np.pad(m, [(0, max_shape[i]-m.shape[i]) for i in range(3)], mode='constant') for m in sample]
        padded_data.append(pd_modalities)
        padded_labels.append(np.pad(gt, [(0, max_shape[i]-gt.shape[i]) for i in range(3)], mode='constant'))
    # Normalize
    means, stds = [], []
    for i in range(len(padded_data[0])):
        vox = np.concatenate([s[i][s[i]>0] for s in padded_data])
        means.append(np.mean(vox)); stds.append(np.std(vox))
    for s in padded_data:
        for i in range(len(s)):
            s[i] = (s[i] - means[i]) / stds[i]
    return padded_data, padded_labels

def combine_modalities(all_mri_data):
    print("Combining modalities...")
    return [np.stack(sample, axis=-1) for sample in all_mri_data]

if __name__ == "__main__":
    data_dir = 'dataset'
    modalities = ['t1','t1ce','t2','flair']
    subs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    random.shuffle(subs)
    for batch_idx, mri, gt in load_mri_data_batch(data_dir, subs, modalities, batch_size=16):
        pd, pl = preprocess_mri_data(mri, gt)
        # Save or pass to next steps
