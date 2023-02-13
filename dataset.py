import os
import pickle
import random

import cv2
import numpy as np
from cv2 import SIFT_create
import pandas as pd
from skimage.io import imread
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import annotations_for_frame, bounding_boxes_to_mask


class HOGExtractor:
    """
    Wrapper for HOG Extractor with hardcoded parameters.
    """

    def __init__(self):
        self.win_size = (64, 64)
        self.block_size = (16, 16)
        self.cell_size = (8, 8)
        self.block_stride = (8, 8)
        self.nbins = 9
        self.deriv_aperture = 1
        self.win_sigma = -1.0
        self.histogram_norm_type = 0
        self.L2_his_threshold = 0.2
        self.gamma_correction = 1
        self.nlevels = 64
        self.signed_gradients = False

        self.hog = cv2.HOGDescriptor(
            self.win_size,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.nbins,
            self.deriv_aperture,
            self.win_sigma,
            self.histogram_norm_type,
            self.L2_his_threshold,
            self.gamma_correction,
            self.nlevels,
            self.signed_gradients,
        )

    def __call__(self, img):
        return self.hog.compute(img)


class CarDetectorDataset:
    """
    Builds the training dataset extracting the features from the images in the train dataset.
    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        min_h: int,
        min_w: int,
        date: str,
        window_size: int = 64,
        bag_of_words: bool = False,
        hist: bool = False,
        spatial: bool = False,
        split: bool = False,
    ):
        """
        Args:
            df_data (pd.DataFrame): Dataframe containing path to images and bounding boxes.
            min_h (int): Minimal height to consider a bounding box for training.
            min_w (int): Minimal width to consider a bounding box for training.
            date (str): String containing creation date information.
            window_size (int, optional): Squared patch size used for training. Defaults to 64.
            bag_of_words (bool, optional): If True, uses Bag-of-SIFT features. Defaults to False.
            hist (bool, optional): If True, uses color histogram features. Defaults to False.
            spatial (bool, optional): If True, uses spatial features. Defaults to False.
            split (bool, optional): If True, creates test/train splits. Defaults to True.
        """
        self.df_data = df_data
        self.min_h = min_h
        self.min_w = min_w
        self.date = date
        self.window_size = window_size
        self.bag_of_words = bag_of_words
        if self.bag_of_words:
            self.sift = SIFT_create()
            with open("vocab.pkl", "rb") as f:
                self.vocab = pickle.load(f)
        self.hist = hist
        self.spatial = spatial
        self.split = split
        self.patches = []
        self.labels = []
        self.N = len(self.df_data)
        self.hog = HOGExtractor()

    def add_positive_samples(self):
        """
        Adds patches containing vehicles to the dataset.
        """
        for idx in tqdm(range(self.N)):
            img = imread(self.df_data.frame_id[idx])
            for x, y, dx, dy in annotations_for_frame(self.df_data, idx):
                if dx > self.min_w and dy > self.min_h:
                    # Patch is added if bigger than minimum size
                    self.patches.append(img[y : y + dy, x : x + dx])
                    self.labels.append(1)

    def add_negative_samples(self):
        """
        Adds negative patches to the dataset in a balanced way.
        These patches need not to intersect with positive patches.
        """
        imgs_per_sample = len(self.patches) // self.N
        r = len(self.patches) / self.N - imgs_per_sample
        for idx in tqdm(range(self.N)):
            img = imread(self.df_data.frame_id[idx])
            H, W, _ = img.shape
            bboxes_list = annotations_for_frame(self.df_data, idx)
            bboxes_mask = bounding_boxes_to_mask(bboxes_list, H, W)
            tmp_patches = []

            # Adds imgs_per_sample from each image
            while len(tmp_patches) < imgs_per_sample:
                y0, x0 = random.randint(int(0.2 * H), int(0.9 * H) - self.window_size), random.randint(
                    0, W - self.window_size
                )
                patch = img[y0 : y0 + self.window_size, x0 : x0 + self.window_size]
                if np.sum(bboxes_mask[y0 : y0 + self.window_size, x0 : x0 + self.window_size]) == 0:
                    tmp_patches.append(patch)

            # Randomly adds one more patch to end up with a balanced dataset
            if r > random.random():
                while len(tmp_patches) < imgs_per_sample + 1:
                    y0, x0 = random.randint(int(0.2 * H), int(0.9 * H) - self.window_size), random.randint(
                        0, W - self.window_size
                    )
                    patch = img[y0 : y0 + self.window_size, x0 : x0 + self.window_size]
                    if np.sum(bboxes_mask[y0 : y0 + self.window_size, x0 : x0 + self.window_size]) == 0:
                        tmp_patches.append(patch)

            # Adds the patches to the dataset
            self.patches += tmp_patches
            self.labels += [0] * len(tmp_patches)

    def get_bags_of_sifts(self, img: np.ndarray):
        """Computes Bag-of-SIFT features using the given vocabulary.

        Args:
            img (np.ndarray): Image to extract bag-of-words features from.

        Returns:
            np.ndarray: Histogram containing Bag-of-SIFT features.
        """
        features = []
        _, descriptors = self.sift.detectAndCompute(img, None)
        dists = pairwise_distances(self.vocab, descriptors)
        argmin = np.argmin(dists, 0)
        hist = np.bincount(argmin, minlength=200)
        hist = hist / hist.sum()
        features.append(hist)
        return np.hstack(features)

    def get_hist_features(self, img: np.ndarray):
        """Computes histogram of colours of a given image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Features containing colours histogram information.
        """
        hist_features = np.array([])
        for channel in range(img.shape[2]):
            channel_hist = np.histogram(img[:, :, channel], bins=64, range=(0, 255))[0]
            hist_features = np.hstack((hist_features, channel_hist))
        return hist_features

    def get_spatial_features(self, img: np.ndarray):
        """Computes spatial features from the image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Returns a 16x16 version of the image.
        """
        return cv2.resize(img, dsize=(16, 16), interpolation=cv2.INTER_CUBIC).reshape(-1)

    def create_dataset(self):
        """Creates the dataset for training:
            - Creates an array of positive and negative patches
            - Extracts the selected features from the patches
            - Normalizes the extracted features
            - Splits the dataset if needed

        Returns:
            tuple: Different data splits.
        """
        # Extracts positive and negative patches
        self.add_positive_samples()
        self.add_negative_samples()

        # Creates features
        self.features = []
        for patch in tqdm(self.patches):
            img = cv2.resize(
                patch,
                dsize=(self.window_size, self.window_size),
                interpolation=cv2.INTER_CUBIC,
            )
            feat_vect = self.hog(img)
            if self.bag_of_words:
                sift_bags = self.get_bags_of_sifts(img)
                feat_vect = np.concatenate((feat_vect, sift_bags), axis=0)
            if self.hist:
                hist_features = self.get_hist_features(img)
                feat_vect = np.concatenate((feat_vect, hist_features), axis=0)
            if self.spatial:
                spatial_features = self.get_spatial_features(img)
                feat_vect = np.concatenate((feat_vect, spatial_features), axis=0)
            self.features.append(feat_vect)

        Xdata = np.array(self.features)
        Ydata = np.array(self.labels)
        print(f"Xdata: {Xdata.shape}")
        print(f"Ydata: {Ydata.shape}")

        # Scaler
        self.scaler = StandardScaler()

        if self.split:
            # Splitting the data
            X_train, X_test, y_train, y_test = train_test_split(
                Xdata,
                Ydata,
                test_size=0.1,
                random_state=42,
                stratify=Ydata,
            )

            # Scaling the data
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        else:
            X_train = self.scaler.fit_transform(Xdata)
            y_train = Ydata
            X_test, y_test = None, None

        # Scaler saved
        filename = f"{self.date}_scaler.pkl"
        with open(os.path.join("models", filename), "wb") as f:
            pickle.dump(self.scaler, f)

        return X_train, X_test, y_train, y_test
