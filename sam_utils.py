import random
from random import randint
import cv2
import matplotlib.pyplot as plt
import numpy as np

def filter_masks(masks):
    """
    check if any mask contain another mask and remove the smaller one frm the bigger one.
    return the filtered masks.
    :param masks (list): list of masks output from sam.
    """
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i == j:
                continue
            # check if mask i contain mask j by anding them
            anded = np.logical_and(masks[i]["segmentation"], masks[j]["segmentation"])
            if np.all(anded == masks[j]["segmentation"]):
                # subtract mask j from mask i
                masks[i]["segmentation"] = np.logical_xor(masks[i]["segmentation"], masks[j]["segmentation"])
                masks[i]["area"] = np.sum(masks[i]["segmentation"])

    # remove empty masks
    masks = [mask for mask in masks if np.sum(mask["segmentation"]) > 0]
    return masks

