import os
import cv2
import matplotlib.pyplot as plt
import json

def produce_overlays(annotation_file, output_dir='data/train', num_images=5):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_dict = {img['id']: img['file_name'] for img in annotations['images']}

    for annotation in annotations['annotations'][:num_images]:
        image_id = annotation['image_id']
        bbox = annotation['bbox']

        img_path = os.path.join(output_dir, image_dict[image_id])
        img = cv2.imread(img_path)

        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

