import os
import shutil

def download_and_prepare_dataset(url, output_dir='data/train'):
    # Download the dataset
    os.system(f'curl -L "{url}" > roboflow.zip')
    os.system('unzip roboflow.zip -d roboflow_data')
    os.system('rm roboflow.zip')

    # Move the dataset into a folder called `data/train/`
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir('roboflow_data/train'):
        s = os.path.join('roboflow_data/train', item)
        d = os.path.join(output_dir, item)
        shutil.move(s, d)

    # Clean up the extracted folder
    shutil.rmtree('roboflow_data')
    print(f'Dataset downloaded and moved to {output_dir}')

