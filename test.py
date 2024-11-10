import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    # Parallel processing to compute mIoU, precision, and recall vs. thresholds with a progress bar
    thresholds = np.linspace(0, 1, 100)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_metrics, thresholds, [gt_boxes_all]*100, [pred_boxes_all]*100, [scores_all]*100), total=len(thresholds), desc="Computing metrics"))

    mious, precisions, recalls = zip(*results)

    # Plot the results
    plot_results(thresholds, mious, precisions, recalls, plot_dir)

def run_inference(dataloader, model, processor, device):
    gt_boxes_all = []
    pred_boxes_all = []
    scores_all = []

    with torch.no_grad():
        for batch_idx, (pixel_values, targets, orig_sizes) in enumerate(tqdm(dataloader, desc="Running inference", leave=False)):
            pixel_values = pixel_values.to(device)

            # Run the model on the input
            outputs = model(pixel_values=pixel_values)

            # Prepare target_sizes (original image sizes)
            target_sizes = torch.tensor([(size[1], size[0]) for size in orig_sizes]).to(device)  # (height, width)

            # Post-process the output
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)

            # Collect ground truth and prediction data
            for i in range(len(targets)):
                # Convert ground truth boxes from COCO format to [x_min, y_min, x_max, y_max]
                gt_boxes = convert_coco_to_xyxy(targets[i]['boxes']).tolist()
                pred_boxes = results[i]['boxes'].tolist()
                scores = results[i]['scores'].tolist()

                gt_boxes_all.append(gt_boxes)
                pred_boxes_all.append(pred_boxes)
                scores_all.append(scores)

    return gt_boxes_all, pred_boxes_all, scores_all

def compute_metrics(threshold, gt_boxes_all, pred_boxes_all, scores_all):
    tp_count = 0
    fp_count = 0
    fn_count = 0
    ious = []

    for gt_boxes, pred_boxes, scores in zip(gt_boxes_all, pred_boxes_all, scores_all):
        filtered_boxes = [box for box, score in zip(pred_boxes, scores) if score >= threshold]
        iou_values = compare_predictions(gt_boxes, filtered_boxes)
        ious.extend(iou_values)

        tp = sum(iou >= 0.5 for iou in iou_values)
        fp = len(filtered_boxes) - tp
        fn = len(gt_boxes) - tp

        tp_count += tp
        fp_count += fp
        fn_count += fn

    mean_iou = sum(ious) / len(ious) if ious else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

    return mean_iou, precision, recall

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
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    denominator = boxAArea + boxBArea - interArea
    if denominator == 0:
        return 0.0

    iou = interArea / denominator
    return iou

def plot_results(thresholds, mious, precisions, recalls, plot_dir):
    plt.figure()
    plt.plot(thresholds, mious, label='mIoU')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('mIoU, Precision, and Recall vs. Threshold')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'metrics_vs_threshold.png'))
    plt.close()

def convert_coco_to_xyxy(boxes):
    """
    Converts bounding boxes from COCO format (x_min, y_min, width, height) to [x_min, y_min, x_max, y_max]
    """
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0]  # x_min
    boxes_xyxy[:, 1] = boxes[:, 1]  # y_min
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x_min + width
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y_min + height
    return boxes_xyxy

if __name__ == "__main__":
    main()
