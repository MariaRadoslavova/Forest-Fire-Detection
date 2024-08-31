
# Safety Tracking Training

This repository contains the code and configuration files for the Safety Tracking Training project. The project includes training scripts, testing scripts, configurations, and necessary dependencies for deploying and running safety tracking models.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Docker Building](#docker-building)
  - [Docker Running](#docker-running)
  - [On Start-Up](#on-start-up)
  - [CPU Monitoring](#cpu-monitoring)
  - [GPU Monitoring](#gpu-monitoring)
  - [Clone Repo and Install Requirements](#clone-repo-and-install-requirements)
  - [Virtual Environment](#virtual-environment)
  - [PyTorch and MMCV](#pytorch-and-mmcv)
    - [CUDA 12.1](#cuda-121)
    - [CUDA 11.3](#cuda-113)
  - [Install Requirements](#install-requirements)
  - [Weights from AWS](#weights-from-aws)
  - [Training](#training)
  - [Testing](#testing)
  - [Tid Bits](#tid-bits)
- [Project Structure](#project-structure)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MariaRadoslavova/Forest-Fire-Detection.git
   cd Forest-Fire-Detection
   ```

2. **Set up the virtual environment:**
   ```bash
   python3 -m venv training
   source training/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### On Start-Up

First, update your package list and install `python3-pip`:
```bash
sudo apt update
sudo apt install python3-pip
python3 -m pip install --upgrade pip
```

Next, install `protobuf-compiler` and `libprotobuf-dev`:
```bash
sudo apt-get install protobuf-compiler libprotobuf-dev
```

### CPU Monitoring

To install and run `bpytop` for CPU monitoring:
```bash
sudo apt install snapd
sudo snap install bpytop
bpytop
```

### GPU Monitoring

To install and run `nvitop` for GPU monitoring:
```bash
sudo apt install pipx
sudo apt install python3.8-venv
pipx run nvitop
```

### Virtual Environment

Set up and activate a virtual environment:
```bash
python3 -m venv training
source training/bin/activate
python3 -m pip install --upgrade pip
pip install --upgrade pip
```

### PyTorch and MMCV

#### CUDA 12.1

To install PyTorch and MMCV with CUDA 12.1 support:
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.10/index.html
```

#### CUDA 11.3

To install PyTorch and MMCV with CUDA 11.3 support:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

### Install Requirements

Install the project dependencies:
```bash
pip install -r requirements.txt
```

### Training

To start the training process:
```bash
./dist_train.sh configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_gh.py 1 gh
```

### Testing

To run tests on the trained model:
```bash
./dist_test.sh configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_gh.py gh/latest.pth 1 --eval bbox
```

### Tid Bits

Additional useful commands:
```bash
export PATH=$PATH:/snap/bin
conda config --set auto_activate_base false
```
