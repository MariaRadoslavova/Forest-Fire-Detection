
# Forest-Fire Detection Project

## Overview

This project focuses on detecting wildfires using advanced object detection models. The repository provides tools for downloading datasets, preprocessing images, training detection models, and visualizing results. Key components include scripts for dataset preparation, training using the DEtection TRansformer (DETR), and evaluating model performance.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [download_dataset.py](#download_datasetpy)
  - [train_detr.py](#train_detrpy)
  - [inference.py](#inferencepy)
  - [visualization.py](#visualizationpy)
- [Project Structure](#project-structure)
- [References](#references)

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:
- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/Forest-Fire-Detection.git
cd Forest-Fire-Detection
pip install -r requirements.txt
```

## Usage

### download_dataset.py

This script downloads and organizes the dataset needed for training. The dataset will be structured into directories with images and COCO-style annotations.

- **Usage**:
  ```bash
  python src/download_dataset.py
  ```

### train_detr.py

`train_detr.py` is responsible for training the DEtection TRansformer (DETR) model on the wildfire dataset. This script handles model setup, training, and validation.

- **Usage**:
  ```bash
  python src/train_detr.py
  ```

### inference.py

`inference.py` provides inference capabilities, allowing users to detect wildfires in new images using the trained model.

- **Usage**:
  ```bash
  python inference.py --input path/to/image_or_folder --output path/to/output_dir
  ```

### visualization.py

`visualization.py` visualizes model predictions and evaluation metrics, such as confusion matrices and performance metrics, to assess the model's accuracy.

- **Usage**:
  ```bash
  python visualization.py
  ```

## Project Structure

- `requirements.txt` - Lists all necessary packages and libraries.
- `train.py` - General training script (use `train_detr.py` for DETR-specific training).
- `inference.py` - Script for running inference on new data.
- `visualization.py` - Script for visualizing model performance and results.
- `src/` - Contains various helper scripts for data preprocessing and analysis:
  - `download_dataset.py` - Downloads and organizes the dataset.
  - `train_detr.py` - Trains the DETR model.
  - `analyze_annotations.py` - Analyzes dataset annotations.
  - `organize_images.py` - Structures images for model input.
  - `produce_overlays.py` - Produces overlays for visual analysis.

## References

This project builds on the DEtection TRansformer (DETR) model and uses COCO-style datasets for training and validation.
