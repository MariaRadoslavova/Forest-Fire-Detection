import requests
import zipfile
import os

def download_dataset(url, save_dir):
    # Download the dataset
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    zip_path = os.path.join(save_dir, 'dataset.zip')
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)

    # Remove the ZIP file after extraction
    os.remove(zip_path)

    print(f'Dataset downloaded and extracted to {save_dir}')

if __name__ == "__main__":
    url = "https://universe.roboflow.com/ds/vmof7PYopz?key=WdcJOXhGC3"  # Update this URL if needed
    save_dir = 'data/raw'
    os.makedirs(save_dir, exist_ok=True)
    download_dataset(url, save_dir)

