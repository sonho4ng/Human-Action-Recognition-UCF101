---

# Human Action Recognition using Deep Learning (UCF101 Dataset)

This project implements a deep learning-based model for human action recognition using the UCF101 dataset. The goal of this project is to classify different human actions captured in videos, leveraging state-of-the-art deep learning models and techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Overview

Human action recognition involves the automatic identification of actions in videos, such as walking, running, swimming, etc. In this project, we use deep learning models to classify actions from the UCF101 dataset. The approach focuses on:

- Data preprocessing and augmentation
- Model design and training
- Evaluation and performance metrics

The UCF101 dataset, which contains 101 action categories, is used for both training and testing the model. This project aims to leverage convolutional neural networks (CNNs), Long Short-Term Memory (LSTM) networks, or 3D CNNs for efficient temporal and spatial feature extraction from video frames.

---

## Dataset

This project uses the **UCF101 dataset**, which consists of 13,320 videos in 101 action categories, collected from YouTube. The actions range from sports and physical activities to everyday human actions.

### UCF101 Dataset Categories

The 101 categories include:
- **Sports:** Basketball, Cricket, Diving, Football, etc.
- **Human Activities:** Walking, Running, Sitting, etc.
- **Others:** Biking, Playing musical instruments, etc.

You can download the dataset from [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php).

### Dataset Structure

Once downloaded and extracted, the dataset should be structured as follows:

```
UCF101/
    train/
        class_1/
            video1.mp4
            video2.mp4
            ...
        class_2/
            video1.mp4
            video2.mp4
            ...
        ...
    test/
        class_1/
            video1.mp4
            video2.mp4
            ...
        ...
```

Ensure the videos are organized by class in the `train` and `test` directories.

---

## Requirements

The following libraries and frameworks are required for running the project:

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- h5py
- tqdm
- wandb
...

To install these dependencies, you can use the provided `requirements.txt` file.

### Example:

```
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/sonho4ng/Human-Action-Recognition-UCF101.git
   cd Human-Action-Recognition-UCF101
   ```

2. **Install dependencies:**

   If you're using a `requirements.txt` file, run:

   ```
   pip install -r requirements.txt
   ```

   Alternatively, install required libraries individually using `pip`.
   
3. **Download checkpoints**
   
   Since our model checkpoints are larger than 100MB, please download them from the following Google Drive link and place them in the *checkpoint* directory found in the root directory:
   [Checkpoint](https://drive.google.com/drive/folders/1GHyuOym3f6AzXiG8fcT6l5gtfjd1Nq65?usp=share_link)

---

## Usage

### Training the Model

1. Ensure that you have the UCF101 dataset downloaded and properly organized.
2. Run the training script to start training the model:

```
python -m script.train \
    --epochs 100 \
    --batch_size 16 \
    --device cuda \
    --learning_rate 0.0005 \
    --num_workers 4 \
    --videos_per_class 30 \
    --n_frames 20 \
    --model resnet-lstm \
    --dataset ucf101 \
    --dataset_path <path-to-your-data> \
    --use_wandb True
```

or:

```
python3 -m script.train \
    --epochs 100 \
    --batch_size 16 \
    --device cuda \
    --learning_rate 0.0005 \
    --num_workers 4 \
    --videos_per_class 30 \
    --n_frames 20 \
    --model resnet-lstm \
    --dataset ucf101 \
    --dataset_path <path-to-your-data> \
    --use_wandb True
```

or just simply run:

```
python3 -m script.train
```
and other parameters are set as default


## Parameters

- **`--epochs`** (int, default: 200):  
  Number of training epochs.

- **`--batch_size`** (int, default: 8):  
  Size of each batch during training.

- **`--device`** (str, choices: ['cuda', 'cpu'], default: 'cpu'):  
  Choose the device for training ('cuda' for GPU, 'cpu' for CPU).

- **`--learning_rate`** (float, default: 0.0001):  
  Learning rate for the optimizer.

- **`--num_workers`** (int, default: 0):  
  Number of worker processes for loading the data.

- **`--videos_per_class`** (int, default: 50):  
  Number of videos per class to use for training.

- **`--n_frames`** (int, default: 10):  
  Number of frames to sample from each video.

- **`--model`** (str, choices: ['resnet-lstm', 'residualSE', 'tsm', 'i3d', 'enhanced_r3d'], default: 'resnet-lstm'):  
  Select the model architecture to use.

- **`--dataset_path`** (str):  
  Path to the folder containing the dataset. Example format: `<Path>/UCF101/train/...`.

- **`--use_wandb`** (str, choices: ['True', 'False'], default: 'False'):  
  Specify whether to enable Weights & Biases (wandb) for experiment tracking (`True` or `False`).

   You can adjust the `--epochs`, `--batch_size`, and other parameters based on your machine and dataset size.

### Testing the Model

Once training is complete or if you simply want to test the pretrained model, run the inference script:
```
python3 -m script.infer \
    --infer_path <video-path-here> \
    --model resnet-lstm \
    --dataset ucf101
```
The model and dataset parameters are specified similarly to above, while `infer_path` requires the path of the video to be inferred from. You can use some provided clips in the `dataset` directory; just copy the path of a video and paste it after `--infer_path`.

You can also run the following for simplicity:
```
python3 -m script.infer --infer_path <video-path-here>
```

This will generate accuracy metrics of some top classes predicted by the model, showing how well the model can recognize actions from the UCF101 test set.

A GUI demo can also be explored by running:
```
streamlit run demo.py
```
You can select a video to load for inference using the ResNet-LSTM model and observe predictions frame-by-frame.

---

## Model Architecture

This project includes various models for human action recognition evaluated on the UCF101 dataset. Below are brief descriptions of each model:

1. **ResidualSE**: This model incorporates residual connections with a squeeze-and-excitation (SE) block to improve feature recalibration. It is designed to enhance the representational power of convolutional neural networks for action recognition.

2. **TSM (Temporal Shift Module)**: TSM introduces temporal shift operations to capture temporal dependencies in video data. This helps the model recognize actions over time by shifting and processing video frames efficiently.

3. **ResNet50 + FC (Fully Connected)**: This model combines ResNet50, a deep convolutional neural network, with fully connected layers for classification. It leverages the strengths of both deep learning and fully connected networks for accurate action recognition.

4. **ResNet50 + biLSTM (Bidirectional LSTM)**: The ResNet50 model is paired with a bidirectional LSTM layer to capture both past and future temporal dependencies in video data. This architecture allows the model to better understand complex motion patterns.

5. **Enhanced ResNet-3D Model**: The Enhanced ResNet-3D Model incorporates 3D convolutions to capture spatio-temporal features from video data. It enhances action recognition performance by effectively modeling the spatial and temporal information in video sequences.



---

## Evaluation Metrics

The model is evaluated using several metrics, including:

- **Accuracy:** The percentage of correctly classified actions.


---

## Results

| Model                          | Accuracy (%) |
|--------------------------------|--------------|
| ResidualSE                     | 69.34        |
| TSM                            | 70.93        |
| ResNet50 + FC                  | 75.44        |
| ResNet50 + biLSTM              | 85.05        |
| Enhanced ResNet-3D             | 92.24        |

These results demonstrate the effectiveness of different architectures, with the **Enhanced ResNet-3D** model achieving the highest accuracy on the UCF101 dataset.

---

## Acknowledgments

- **UCF101 Dataset:** [UCF101 Dataset Homepage](https://www.crcv.ucf.edu/data/UCF101.php)
- **Deep Learning Frameworks:** PyTorch
- **Research Papers:** [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767), [](https://arxiv.org/abs/1506.01826)

---
