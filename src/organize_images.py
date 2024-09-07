import os
import shutil

def organize_dataset(raw_dir, target_dir):
    # Create the 'data/train' folder if it doesn't exist
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)

    # Move the contents of the dataset's 'train' folder into the 'data/train/' folder
    for item in os.listdir(os.path.join(raw_dir, 'train')):
        s = os.path.join(raw_dir, 'train', item)
        d = os.path.join(target_dir, 'train', item)
        shutil.move(s, d)

    # Clean up the extracted folder
    shutil.rmtree(raw_dir)

    print(f'Dataset organized into {target_dir}')

if __name__ == "__main__":
    raw_dir = 'data/raw'
    target_dir = 'data/train'
    organize_dataset(raw_dir, target_dir)


