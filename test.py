import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_metrics(threshold, gt_boxes_all, pred_boxes_all, scores_all):
    tp_count = 0
    fp_count = 0
    fn_count = 0
    ious = []

    for gt_boxes, pred_boxes, scores in zip(gt_boxes_all, pred_boxes_all, scores_all):
        # Filter boxes by threshold
        filtered_boxes = [box for box, score in zip(pred_boxes, scores) if score >= threshold]
        
        # Calculate IoU for filtered predictions against ground truth boxes
        iou_values = compare_predictions(gt_boxes, filtered_boxes)
        ious.extend(iou_values)

        # Calculate true positives, false positives, and false negatives
        tp = sum(iou >= 0.5 for iou in iou_values)  # Matches with IoU >= 0.5 are considered TP
        fp = len(filtered_boxes) - tp  # Remaining predictions are false positives
        fn = len(gt_boxes) - tp  # Remaining ground truths not matched are false negatives

        tp_count += tp
        fp_count += fp
        fn_count += fn

    # Calculate mean IoU, precision, and recall
    mean_iou = sum(ious) / len(ious) if ious else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

    return mean_iou, precision, recall
