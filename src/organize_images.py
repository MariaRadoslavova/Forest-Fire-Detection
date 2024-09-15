import os
import shutil
import json

def organize_images(train_annotations, test_annotations, image_dir='data/images'):
    with open(train_annotations, 'r') as f:
        train_data = json.load(f)
    
    with open(test_annotations, 'r') as f:
        test_data = json.load(f)

    # Create directories for train and test images
    train_img_dir = os.path.join(image_dir, 'train')
    test_img_dir = os.path.join(image_dir, 'test')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # Move train images
    for img in train_data['images']:
        img_path = os.path.join(image_dir, img['file_name'])
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(train_img_dir, img['file_name']))

    # Move test images
    for img in test_data['images']:
        img_path = os.path.join(image_dir, img['file_name'])
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(test_img_dir, img['file_name']))

# Example usage
if __name__ == "__main__":
    organize_images('data/annotations/train_annotations.json', 'data/annotations/test_annotations.json', 'data/images')





