import json
import os
import shutil
from sklearn.model_selection import train_test_split

def split_coco_json(input_json, output_dir, test_size=0.2):
    # Load the COCO JSON file
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Split the annotations
    annotations = data['annotations']
    train_annotations, test_annotations = train_test_split(annotations, test_size=test_size)

    # Prepare the train and test data
    train_data = data.copy()
    train_data['annotations'] = train_annotations

    test_data = data.copy()
    test_data['annotations'] = test_annotations

    # Save the split data
    train_json_path = os.path.join(output_dir, 'train_annotations.json')
    test_json_path = os.path.join(output_dir, 'test_annotations.json')

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)

    with open(test_json_path, 'w') as f:
        json.dump(test_data, f)

if __name__ == '__main__':
    input_json = 'path/to/your/input_coco.json'
    output_dir = 'data/annotations'  # Save to the annotations directory

    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    split_coco_json(input_json, output_dir)
