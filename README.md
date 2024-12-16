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
   git clone https://github.com/xt2201/it3320e-human-action-recognition-ucf101.git
   cd it3320e-human-action-recognition-ucf101
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
   python train.py --epochs 200 --batch_size 8 --device cuda --learning_rate 0.0001 --num_workers 4 --videos_per_class 50 --n_frames 10 --model resnet-lstm --dataset ucf101 --dataset_path <path_to_dataset>
   ```
Parameters:

- --epochs (int, default: 200): Number of training epochs.

- --batch_size (int, default: 8): Size of each batch during training.

- --device (str, choices: ['cuda', 'cpu'], default: 'cuda'): Choose the device for training ('cuda' for GPU, 'cpu' for CPU).

- --learning_rate (float, default: 0.0001): Learning rate for the optimizer.

- --num_workers (int, default: 0): Number of worker processes for loading the data.

- --videos_per_class (int, default: 50): Number of videos per class to use for training.

- --n_frames (int, default: 10): Number of frames to extract from each video for training.

- --model (str, choices: ['resnet-lstm', 'resnet-fc], default: 'resnet-lstm'): Choose the model architecture.

- --dataset (str, choices: ['ucf101', 'ucf11'], default: 'ucf101'): Choose the dataset to use ('ucf101' or 'ucf11').

- --dataset_path (str): Path to the folder containing the dataset. Example: <Path>/UCF101/train/....

   You can adjust the `--epochs`, `--batch_size`, and other parameters based on your machine and dataset size.

### Testing the Model

Once training is complete, open the notebook demo.ipynb to evaluate the model's performance:

This will generate accuracy metrics and visualize some test results, showing how well the model can recognize actions from the UCF101 test set.

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
| ResidualSE                     | 59.34        |
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
