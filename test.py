import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from sklearn.metrics import precision_recall_fscore_support
import os
import json

def main():
    # Paths to your dataset and annotations
    test_annotations = 'data/annotations/test_annotations.json'
    test_img_dir = 'data/images/test'

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load the trained model weights
    model_weights_path = 'weights/model_epoch_1.pth'  # Adjust to the epoch/model you want to use
    model.load_state_dict(torch.load(model_weights_path))

    # Create dataset instance
    test_dataset = COCODataset(test_annotations, test_img_dir, processor)

    # Use the custom collate function with DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    iou_threshold = 0.5  # IoU threshold for True Positives
    ious = []
    gt_boxes_all = []
    pred_boxes_all = []

    # Run inference
    with torch.no_grad():
        for batch_idx, (pixel_values, targets) in enumerate(test_dataloader):
            pixel_values = pixel_values.to(device)

            # Run the model on the input
            outputs = model(pixel_values=pixel_values)

            # Post-process the output
            results = processor.post_process_object_detection(outputs, target_sizes=[pixel_values.shape[-2:]])

            # Convert tensors in the results to serializable format
            results_serializable = convert_to_serializable(results)

            # Collect ground truth and prediction data
            gt_boxes = [target['boxes'].tolist() for target in targets]
            pred_boxes = results_serializable[0]['boxes']

            # Save results to file
            save_results(batch_idx, results_serializable)

            # Compare predictions to ground truth
            iou_values = compare_predictions(gt_boxes[0], pred_boxes, iou_threshold)
            ious.extend(iou_values)

            gt_boxes_all.extend(gt_boxes[0])
            pred_boxes_all.extend(pred_boxes)

    # Summary of results
    mean_iou = sum(ious) / len(ious) if ious else 0
    print(f"Mean IoU over the dataset: {mean_iou:.4f}")

def convert_to_serializable(results):
    """
    Convert tensors in the results to serializable format (e.g., lists).
    """
    for i, result in enumerate(results):
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                results[i][key] = value.tolist()  # Convert tensor to list
    return results

def save_results(batch_idx, results):
    """
    Save or display the results of the inference.
    """
    output_path = f"results/result_{batch_idx}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

def compare_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Compare ground truth boxes with predicted boxes using IoU.
    
    Returns a list of IoU values.
    """
    iou_values = []
    for gt_box in gt_boxes:
        max_iou = 0
        for pred_box in pred_boxes:
            iou = compute_iou(gt_box, pred_box)
            max_iou = max(max_iou, iou)
        if max_iou >= iou_threshold:
            iou_values.append(max_iou)
    return iou_values

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two boxes.
    Each box is represented as [x_min, y_min, x_max, y_max].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground truth boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the Intersection over Union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

if __name__ == "__main__":
    main()
