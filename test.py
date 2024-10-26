import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from sklearn.metrics import precision_recall_fscore_support
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def main():
    # Paths to your dataset and annotations
    test_annotations = 'data/annotations/test_annotations.json'
    test_img_dir = 'data/images/test'
    plot_dir = 'data/plots'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load the trained model weights
    model_weights_path = 'weights/model_epoch_1.pth'
    model.load_state_dict(torch.load(model_weights_path))

    # Create dataset instance
    test_dataset = COCODataset(test_annotations, test_img_dir, processor)

    # Use the custom collate function with DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Run inference and collect IoU values with a progress bar
    gt_boxes_all, pred_boxes_all, scores_all = run_inference(test_dataloader, model, processor, device)

    # Parallel processing to compute mIoU, TP, FP vs. thresholds with a progress bar
    thresholds = np.linspace(0, 1, 100)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_metrics, thresholds, [gt_boxes_all]*100, [pred_boxes_all]*100, [scores_all]*100), total=len(thresholds), desc="Computing metrics"))

    mious, tps, fps = zip(*results)

    # Plot the results
    plot_results(thresholds, mious, tps, fps, plot_dir)

def run_inference(dataloader, model, processor, device):
    gt_boxes_all = []
    pred_boxes_all = []
    scores_all = []

    with torch.no_grad():
        for batch_idx, (pixel_values, targets) in enumerate(tqdm(dataloader, desc="Running inference", leave=False)):
            pixel_values = pixel_values.to(device)

            # Run the model on the input
            outputs = model(pixel_values=pixel_values)

            # Post-process the output
            results = processor.post_process_object_detection(outputs, target_sizes=[pixel_values.shape[-2:]])

            # Collect ground truth and prediction data
            gt_boxes = [target['boxes'].tolist() for target in targets]
            pred_boxes = results[0]['boxes'].tolist()
            scores = results[0]['scores'].tolist()

            gt_boxes_all.append(gt_boxes[0])
            pred_boxes_all.append(pred_boxes)
            scores_all.append(scores)

    return gt_boxes_all, pred_boxes_all, scores_all

def compute_metrics(threshold, gt_boxes_all, pred_boxes_all, scores_all):
    tp_count = 0
    fp_count = 0
    ious = []

    for gt_boxes, pred_boxes, scores in zip(gt_boxes_all, pred_boxes_all, scores_all):
        filtered_boxes = [box for box, score in zip(pred_boxes, scores) if score >= threshold]
        iou_values = compare_predictions(gt_boxes, filtered_boxes)
        ious.extend(iou_values)

        tp = sum(iou >= 0.5 for iou in iou_values)
        fp = len(filtered_boxes) - tp

        tp_count += tp
        fp_count += fp

    mean_iou = sum(ious) / len(ious) if ious else 0
    return mean_iou, tp_count, fp_count

def compare_predictions(gt_boxes, pred_boxes):
    iou_values = []
    for gt_box in gt_boxes:
        max_iou = 0
        for pred_box in pred_boxes:
            iou = compute_iou(gt_box, pred_box)
            max_iou = max(max_iou, iou)
        iou_values.append(max_iou)
    return iou_values

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def plot_results(thresholds, mious, tps, fps, plot_dir):
    plt.figure()
    plt.plot(thresholds, mious, label='mIoU')
    plt.xlabel('Threshold')
    plt.ylabel('mIoU')
    plt.title('mIoU vs. Threshold')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'miou_vs_threshold.png'))
    plt.close()

    plt.figure()
    plt.plot(thresholds, tps, label='True Positives')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('TP vs. Threshold')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'tp_vs_threshold.png'))
    plt.close()

    plt.figure()
    plt.plot(thresholds, fps, label='False Positives')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('FP vs. Threshold')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'fp_vs_threshold.png'))
    plt.close()

if __name__ == "__main__":
    main()
