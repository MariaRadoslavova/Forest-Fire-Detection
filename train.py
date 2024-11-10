import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.coco_dataset import COCODataset, collate_fn
from src.train_detr import train_model
from src.download_dataset import download_and_prepare_dataset
from src.produce_overlays import produce_overlays
from src.split_coco_json import split_coco_json
from src.organize_images import organize_images

import json

def main():
    
    # Prepare data pipeline (download, preprocess, organize)
    prepare_data()

    # Paths to your dataset and annotations
    train_annotations = 'data/annotations/train_annotations.json'
    test_annotations = 'data/annotations/test_annotations.json'
    train_img_dir = 'data/images/train'
    test_img_dir = 'data/images/test'

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Create dataset instances
    train_dataset = COCODataset(train_annotations, train_img_dir, processor)
    test_dataset = COCODataset(test_annotations, test_img_dir, processor)

    # Use this custom collate function with DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    train_model(train_dataloader, test_dataloader, model, optimizer, device)

def prepare_data():
    # Step 1: Download and prepare dataset
    dataset_url = "https://universe.roboflow.com/ds/vmof7PYopz?key=WdcJOXhGC3"
    download_and_prepare_dataset(dataset_url)

    # Step 2: Produce overlays on images and save to `data/overlays`
    annotation_file = 'data/annotations/_annotations.coco.json'
    produce_overlays(annotation_file, num_images=5)

    # Step 3: Split the dataset into training and testing sets
    split_coco_json(annotation_file, 'data/annotations', 'train_annotations.json', 'test_annotations.json')

    # Step 4: Organize images into train/test folders
    organize_images('data/annotations/train_annotations.json', 'data/annotations/test_annotations.json', 'data/images')

if __name__ == "__main__":
    main()


