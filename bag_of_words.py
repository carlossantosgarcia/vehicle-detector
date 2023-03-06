import pickle

import cv2
import numpy as np
import pandas as pd
from cv2 import SIFT_create
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset import CarDetectorDataset


def build_vocabulary(patches_list: list[np.ndarray], vocab_size: int = 200) -> np.ndarray:
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Args:
        patches_list (list[np.ndarray]): List of image patches.
        vocab_size (int, optional): Size of the vocabulary to build. Defaults to 200.

    Returns:
        np.ndarray: Vocabulary of size (vocab_size, 128). Each row is a visual word.
    """
    # SIFT Extractor
    sift = SIFT_create()

    # Extract features
    features = []
    keypoints = []
    for patch in tqdm.tqdm(patches_list):
        # Patches are 64x64
        img = cv2.resize(patch, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        # Computes SIFT features
        keys, descriptors = sift.detectAndCompute(img, None)

        if descriptors is not None:
            indices = np.arange(len(descriptors))
            keypoints.append([keys[i] for i in indices])
            features.append(descriptors[indices])

    all_features = np.concatenate(features, axis=0)

    # Clusters features
    kmeans = MiniBatchKMeans(n_clusters=vocab_size)
    kmeans.fit(all_features)

    # Gets centroids
    vocab = kmeans.cluster_centers_

    return vocab


if __name__ == "__main__":
    # Loads and creates dataset
    df_data = pd.read_csv("data/train.csv")
    dataset = CarDetectorDataset(
        df_data=df_data,
        min_h=30,
        min_w=30,
        window_size=64,
        date="",
    )
    dataset.add_positive_samples()
    dataset.add_negative_samples()

    # Builds vocabulary
    vocab = build_vocabulary(dataset.patches)

    # Saves vocabulary
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
