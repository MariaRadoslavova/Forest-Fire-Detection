import os
import subprocess

def download_and_unzip_data():
    # Define the data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Create the data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Define the file paths
    zip_file_path = os.path.join(data_dir, 'roboflow.zip')
    unzip_dir = os.path.join(data_dir, 'roboflow_data')
    
    # Download the dataset
    download_command = f'curl -L "https://universe.roboflow.com/ds/vmof7PYopz?key=WdcJOXhGC3" > {zip_file_path}'
    subprocess.run(download_command, shell=True, check=True)
    
    # Unzip the dataset
    unzip_command = f'unzip {zip_file_path} -d {unzip_dir}'
    subprocess.run(unzip_command, shell=True, check=True)
    
    # Remove the zip file
    os.remove(zip_file_path)

if __name__ == "__main__":
    download_and_unzip_data()
