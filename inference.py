import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn

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

    # Run inference
    with torch.no_grad():
        for batch_idx, (pixel_values, _) in enumerate(test_dataloader):
            pixel_values = pixel_values.to(device)

            # Run the model on the input
            outputs = model(pixel_values=pixel_values)

            # Post-process the output
            results = processor.post_process_object_detection(outputs, target_sizes=[pixel_values.shape[-2:]])

            # Save or display the results
            save_results(batch_idx, results)

def save_results(batch_idx, results):
    """
    Save or display the results of the inference.
    """
    output_path = f"results/result_{batch_idx}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
