import os
import requests
import zipfile

def download_and_prepare_dataset(dataset_url, output_dir='data'):
    dataset_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'annotations')

    # Create directories if they don't exist
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Download the dataset
    response = requests.get(dataset_url)
    zip_path = os.path.join(output_dir, 'dataset.zip')

    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Extract the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Move images to the correct folder
    extracted_img_dir = os.path.join(output_dir, 'train')
    for file_name in os.listdir(extracted_img_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            os.rename(os.path.join(extracted_img_dir, file_name), os.path.join(dataset_dir, file_name))

    # Move annotations file
    annotation_file = os.path.join(extracted_img_dir, '_annotations.coco.json')
    if os.path.exists(annotation_file):
        os.rename(annotation_file, os.path.join(annotations_dir, '_annotations.coco.json'))

    # Clean up extracted folder and zip file
    os.rmdir(extracted_img_dir)
    os.remove(zip_path)

# Example usage
if __name__ == "__main__":
    dataset_url = "https://universe.roboflow.com/ds/vmof7PYopz?key=WdcJOXhGC3"
    download_and_prepare_dataset(dataset_url)


