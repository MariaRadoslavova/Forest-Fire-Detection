import cv2
import matplotlib.pyplot as plt
import json
import os

def produce_overlays(annotation_file, images_dir, overlays_dir):
    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Dictionary to map image IDs to file names
    image_dict = {img['id']: img['file_name'] for img in annotations['images']}

    # Create the overlays directory if it doesn't exist
    os.makedirs(overlays_dir, exist_ok=True)

    # Process and save overlays for a few images
    for annotation in annotations['annotations'][:5]:  # Limiting to 5 images for display
        image_id = annotation['image_id']
        bbox = annotation['bbox']  # Bounding box: [x, y, width, height]

        # Load the image
        img_path = os.path.join(images_dir, image_dict[image_id])
        img = cv2.imread(img_path)

        # Draw the bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green bounding box

        # Save the image with the overlay
        overlay_path = os.path.join(overlays_dir, image_dict[image_id])
        cv2.imwrite(overlay_path, img)

    print(f'Overlays produced and saved to {overlays_dir}')

if __name__ == "__main__":
    annotation_file = 'data/train/_annotations.coco.json'
    images_dir = 'data/train'
    overlays_dir = 'data/overlays'
    produce_overlays(annotation_file, images_dir, overlays_dir)
