from src.analyze_annotations import plot_class_distribution, generate_statistics_report
from src.download_dataset import download_and_prepare_dataset
from src.produce_overlays import produce_overlays
from src.split_coco_json import split_coco_json
from src.organize_images import organize_images

import json

def main():
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

    # Step 5: Analyze and visualize annotations
    with open('data/annotations/train_annotations.json', 'r') as f:
        train_data = json.load(f)

    plot_class_distribution(train_data['annotations'])
    generate_statistics_report(train_data['annotations'])

if __name__ == "__main__":
    main()


