import json
import os
import random

def split_coco_json(annotation_file, output_dir='data/annotations', train_file='train_annotations.json', test_file='test_annotations.json', test_size=0.2):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    num_images = len(images)
    num_test_images = int(num_images * test_size)
    test_images = random.sample(images, num_test_images)
    train_images = [img for img in images if img not in test_images]

    def filter_annotations(img_ids):
        return [ann for ann in annotations if ann['image_id'] in img_ids]

    train_annotations = filter_annotations([img['id'] for img in train_images])
    test_annotations = filter_annotations([img['id'] for img in test_images])

    # Save train and test annotations
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': data['categories']
    }

    test_data = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': data['categories']
    }

    with open(os.path.join(output_dir, train_file), 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, test_file), 'w') as f:
        json.dump(test_data, f, indent=2)

# Example usage
if __name__ == "__main__":
    split_coco_json('data/annotations/_annotations.coco.json')


