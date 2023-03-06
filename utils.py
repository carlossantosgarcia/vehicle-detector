from datetime import datetime

import numpy as np
import pandas as pd
from skimage.io import imread


def read_frame(df_annotation, frame):
    """Read frames and create integer frame_id-s"""
    file_path = df_annotation[df_annotation.index == frame]["frame_id"].values[0]
    return imread(file_path)


def annotations_for_frame(df_annotation, frame):
    assert frame in df_annotation.index
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    if pd.isna(bbs):  # some frames contain no vehicles
        return []

    bbs = list(map(lambda x: int(x), bbs.split(" ")))
    return np.array_split(bbs, len(bbs) / 4)


def bounding_boxes_to_mask(bounding_boxes, H, W):
    """
    Converts set of bounding boxes to a binary mask
    """
    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[y : y + dy, x : x + dx] = 1

    return mask


def run_length_encoding(mask):
    """
    Produces run length encoding for a given binary mask
    """

    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]
    padded = np.pad(non_zeros, pad_width=1, mode="edge")

    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return " ".join(["%d %d" % (s, l) for s, l in zip(starts, lengths)])


def date_to_string():
    """
    Creates a string with current date information, e.g.: "14_May_16h23"
    """
    now = datetime.now()
    current_day = now.day
    current_month = now.strftime("%h")
    current_hour = now.strftime("%H")
    current_min = now.strftime("%M")
    return f"{current_day}_{current_month}_{current_hour}h{current_min}"
