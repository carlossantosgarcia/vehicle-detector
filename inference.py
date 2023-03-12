import argparse
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tqdm
from cv2 import SIFT_create
from scipy.ndimage import label
from skimage.io import imread
from skimage.measure import regionprops
from sklearn.metrics import pairwise_distances

from dataset import HOGExtractor
from utils import bounding_boxes_to_mask, run_length_encoding


def get_bags_of_sifts(sift_extractor, vocab, img):
    """
    Computes Bag-of-SIFT Features.
    """
    features = []
    _, descriptors = sift_extractor.detectAndCompute(img, None)
    dists = pairwise_distances(vocab, descriptors)
    argmin = np.argmin(dists, 0)
    hist = np.bincount(argmin, minlength=200)
    hist = hist / hist.sum()
    features.append(hist)
    return np.hstack(features)


def get_hist_features(img):
    """
    Computes colour histogram features.
    """
    hist_features = np.array([])
    for channel in range(img.shape[2]):
        channel_hist = np.histogram(img[:, :, channel], bins=64, range=(0, 255))[0]
        hist_features = np.hstack((hist_features, channel_hist))
    return hist_features


def get_spatial_features(img):
    """
    Computes spatial features (a resized version of the patch).
    """
    return cv2.resize(img, dsize=(16, 16), interpolation=cv2.INTER_CUBIC).reshape(-1)


def compute_features(
    patch: np.ndarray,
    hog_extractor: callable,
    sift_extractor: callable,
    vocab: np.ndarray,
    bow: bool,
    hist: bool,
    spatial: bool,
) -> np.ndarray:
    """Computes features from a given patch.

    Args:
        patch (np.ndarray): Input patch.
        hog_extractor (callable): HOG Features extractor.
        sift_extractor (callable): SIFT Features extractor.
        vocab (np.ndarray): Vocabulary for Bag-of-SIFT features.
        bow (bool): If True, computes Bag-of-SIFT features.
        hist (bool): If True, computes colour hitogram features.
        spatial (bool): If True, computes spatial features.

    Returns:
        np.ndarray: Returns the feature vector for a given patch.
    """
    resized_patch = cv2.resize(patch, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    fd = hog_extractor(resized_patch)
    if bow:
        sift_bags = get_bags_of_sifts(sift_extractor, vocab, resized_patch)
        fd = np.concatenate((fd, sift_bags), axis=0)
    if hist:
        hist_features = get_hist_features(resized_patch)
        fd = np.concatenate((fd, hist_features), axis=0)
    if spatial:
        spatial_features = get_spatial_features(resized_patch)
        fd = np.concatenate((fd, spatial_features), axis=0)
    return fd


def sliding_window(
    yrange: tuple[int],
    scale: float,
    overlap: float,
    H: int = 720,
    W: int = 1280,
) -> list:
    """
    Creates a list of windows coordinates on which to extract patches to run the binary classifier.

    Args:
        yrange (tuple[int]): Range of image heights from which to extract the patches.
        scale (float): Scaling coefficient for perspective purposes. Patches closer to the car get bigger.
        overlap (float): X-axis overlap coefficient (between 0 and 1).
        H (int, optional): Height of the image. Defaults to 720.
        W (int, optional): Width of the image. Defaults to 1280.

    Returns:
        list: List of tuples (x, y, win_width, win_height) containing the windows to look at.
    """
    assert (0 <= overlap) and (overlap <= 1), "Overlap coefficient needs to be between 0 and 1."
    h0, w0 = 64, 64
    ymin, ymax = int(yrange[0] * H), int(yrange[1] * H)
    windows = []
    for y in range(ymin, ymax, 2):
        win_width = int(w0 + (scale * (y - (yrange[0] * H))))
        win_height = int(h0 + (scale * (y - (yrange[0] * H))))
        if y + win_height > ymax:
            break
        xstep = int((1 - overlap) * win_width)
        for x in range(0, W, xstep):
            if x + win_width > W:
                break
            else:
                windows.append((x, y, win_width, win_height))

    return windows


def run_inference(
    test_paths: list[str],
    clf_path: str,
    scaler_path: str,
    vocab_path: str,
    thr: float = 0.63,
    H: int = 720,
    W: int = 1280,
    save_csv_submission=False,
):
    """Runs inference pipeline on test images:
            - Patch selection through sliding window
            - Feature extraction
            - Probability computation and aggregation
            - Binarization of the regions and bounding box creation

    Args:
        test_paths (list[str]): List of test image paths.
        clf_path (str): Path to the classifer.
        scaler_path (str): Path to the data scaler.
        thr (float, optional): Threshold used to binarize probability heatmap. Defaults to 0.63.
        save_csv_submission (bool, optional): If True, creates a .csv submission file for the challenge. Defaults to False.
    """
    rows = []

    # Loads classifier and scaler
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Extractors
    hog_extractor = HOGExtractor()
    sift_extractor = SIFT_create()
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Sliding windows for inference
    windows = sliding_window(yrange=(0.2, 0.8), scale=1.5, overlap=0.75)

    for img_path in tqdm.tqdm(test_paths):
        # Reads image
        test_img = imread(img_path)

        # Predicted probabilities
        preds = np.zeros((720, 1280))

        for x, y, w, h in windows:
            # Extracts patch
            patch = test_img[y : y + h, x : x + w]

            # Computes features
            fd = compute_features(
                patch=patch,
                hog_extractor=hog_extractor,
                sift_extractor=sift_extractor,
                vocab=vocab,
                bow=True,
                hist=True,
                spatial=True,
            )

            # Scales the data
            fd = scaler.transform(np.array(fd).reshape(1, -1))

            # Predicts probabilities of containing vehicles
            p = clf.predict_proba(fd)[0][1]

            # Saves probability
            preds[y : y + h, x : x + w] += p

        # Binarization
        heatmap = preds / np.max(preds)
        norm_preds = np.where(heatmap > thr, 1, 0)

        # Connected components of mask
        cc_preds = np.zeros_like(preds)
        N = label(norm_preds, output=cc_preds)

        # Bounding boxes computation
        bboxes = np.zeros_like(cc_preds)
        list_of_boxes = []
        for i in range(1, N + 1):
            blob = np.where(cc_preds == i, 1, 0)
            min_row, min_col, max_row, max_col = regionprops(blob)[0].bbox
            if (max_row - min_row) * (max_col - min_col) > 30 * 30:
                bboxes[min_row:max_row, min_col:max_col] = 1
                x, y, w, h = min_col, min_row, max_col - min_col, max_row - min_row
                list_of_boxes.append([x, y, w, h])

        # Compute submission format
        rle = run_length_encoding(bounding_boxes_to_mask(list_of_boxes, H, W))
        rows.append([img_path, rle])

    if save_csv_submission:
        df_prediction = pd.DataFrame(columns=["Id", "Predicted"], data=rows).set_index("Id")
        df_prediction.to_csv("submission.csv")
        print("Submission saved to submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches inference on test images.")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test images folder")
    parser.add_argument("--clf_path", type=str, default="models/gradient_boosting.pkl", help="Path to the classifier")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl", help="Path to the data scaler")
    parser.add_argument("--vocab_path", type=str, default="models/vocab.pkl", help="Path to the learned vocabulary")
    parser.add_argument("--submission", dest="submission", action="store_true", help="Save Kaggle submission")
    parser.add_argument("--no-submission", dest="submission", action="store_false")
    parser.set_defaults(spatial=False)
    args = parser.parse_args()

    run_inference(
        test_paths=[os.path.join(args.test_dir, file) for file in os.listdir(args.test_dir)],
        clf_path=args.clf_path,
        scaler_path=args.scaler_path,
        vocab_path=args.vocab_path,
        save_csv_submission=args.submission,
    )
