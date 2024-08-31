import json
import random

def split_coco_json(coco_json_path, train_json_path, test_json_path, train_ratio=0.8, seed=42):
    random.seed(seed)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']
    random.shuffle(images)
    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    test_images = images[num_train:]
    train_image_ids = set(img['id'] for img in train_images)
    test_image_ids = set(img['id'] for img in test_images)
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
    train_coco = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
    test_coco = {'images': test_images, 'annotations': test_annotations, 'categories': categories}
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=4)
    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f, indent=4)

# Example usage
split_coco_json('data/train/_annotations.coco.json', 'train_annotations.json', 'test_annotations.json', 0.8, 42)
