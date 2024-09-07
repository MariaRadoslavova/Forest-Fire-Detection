import os
from scripts.download_dataset import download_dataset
from scripts.produce_overlays import produce_overlays
from scripts.organize_images import organize_images
from scripts.split_coco_json import split_coco_json

# Define paths
data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
overlays_dir = os.path.join(data_dir, 'overlays')
annotations_dir = os.path.join(data_dir, 'annotations')
input_json = 'path/to/your/input_coco.json'

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(overlays_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Run your scripts
download_dataset('dataset_url', images_dir)
produce_overlays(images_dir, overlays_dir)
organize_images(images_dir)
split_coco_json(input_json, annotations_dir)

