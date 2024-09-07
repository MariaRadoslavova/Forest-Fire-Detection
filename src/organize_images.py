import os
import shutil
import json

def organize_images(train_json_path, test_json_path, images_dir, output_dir='images'):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    def move_images(images, target_dir):
        for img in images:
            img_filename = img['file_name']
            src_path = os.path.join(images_dir, img_filename)
            dest_path = os.path.join(target_dir, img_filename)
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
            else:
                print(f"Warning: {src_path} does not exist.")

    move_images(train_data['images'], train_output_dir)
    move_images(test_data['images'], test_output_dir)

    print(f"Images organized into {train_output_dir} and {test_output_dir}")
