import cv2
import matplotlib.pyplot as plt
import json
import os

def visualize_overlays(annotation_file, image_dir):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_dict = {img['id']: img['file_name'] for img in annotations['images']}
    for annotation in annotations['annotations'][:5]:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        img_path = os.path.join(image_dir, image_dict[image_id])
        img = cv2.imread(img_path)
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage
visualize_overlays('data/train/_annotations.coco.json', 'data/train')
