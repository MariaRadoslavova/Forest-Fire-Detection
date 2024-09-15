import os
import cv2
import json

import os
import cv2
import json

def produce_overlays(annotation_file, output_dir='data/images', overlay_dir='data/overlays', num_images=5):
    # Create the overlays directory if it doesn't exist
    os.makedirs(overlay_dir, exist_ok=True)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_dict = {img['id']: img['file_name'] for img in annotations['images']}

    for idx, annotation in enumerate(annotations['annotations'][:num_images]):
        image_id = annotation['image_id']
        bbox = annotation['bbox']

        img_path = os.path.join(output_dir, image_dict[image_id])
        img = cv2.imread(img_path)

        # Draw the bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the overlay image
        overlay_path = os.path.join(overlay_dir, f"overlay_{idx+1}.jpg")
        cv2.imwrite(overlay_path, img)

        print(f"Saved overlay image to {overlay_path}")

# Example usage
if __name__ == "__main__":
    produce_overlays('data/annotations/_annotations.coco.json', num_images=5)

