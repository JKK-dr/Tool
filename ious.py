import numpy as np
import torch
import open3d as o3d

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap

def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou
    
def compute_3d_giou(bbox1, bbox2):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())
    
    # Intersection
    intersec_min = np.maximum(bbox1_min, bbox2_min)
    intersec_max = np.minimum(bbox1_max, bbox2_max)
    intersec_size = np.maximum(intersec_max - intersec_min, 0.0)
    intersec_volume = np.prod(intersec_size)

    # Union
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    union_volume = bbox1_volume + bbox2_volume - intersec_volume
    
    iou = intersec_volume / union_volume
    
    # Enclosing box
    enclosing_min = np.minimum(bbox1_min, bbox2_min)
    enclosing_max = np.maximum(bbox1_max, bbox2_max)
    enclosing_size = np.maximum(enclosing_max - enclosing_min, 0.0)
    enclosing_volume = np.prod(enclosing_size)
    
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume
    
    return giou

def compute_giou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute the generalized IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, D)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, D)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, D)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, D)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, D)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, D)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, D)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, D)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, D)
    
    # to obtain the coordinates of enclosing box
    enclosing_min = torch.min(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    enclosing_max = torch.max(bbox1_max, bbox2_max)  # Shape: (M, N, D)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)
    enclosing_vol = torch.prod(enclosing_max - enclosing_min, dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)
    union_vol = bbox1_vol + bbox2_vol - inter_vol

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (union_vol + 1e-10)
    giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + 1e-10)

    return giou
