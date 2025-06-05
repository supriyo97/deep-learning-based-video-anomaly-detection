<div align="center">

# DEEP-LEARNING-BASED-VIDEO-ANOMALY-DETECTION

Detect anomalies, enhance security, computer vision.

![last commit](https://img.shields.io/github/last-commit/supriyo97/deep-learning-based-video-anomaly-detection?color=007EC6&style=flat-square) ![languages](https://img.shields.io/github/languages/top/supriyo97/deep-learning-based-video-anomaly-detection?color=F37626&style=flat-square)

Built with the tools and technologies.

</div>


## Table of Contents

- [Overview](#overview)
- [Implementation Details](#implementation-details)
  - [Environment Setup](#1-environment-setup)
  - [Model Architectures](#2-model-architectures)
    - [CNN-LSTM Autoencoder](#cnn-lstm-autoencoder)
    - [3D Convolutional Autoencoder (C3D)](#3d-convolutional-autoencoder-c3d)
  - [Loss Function](#4-loss-function)
  - [Data Preprocessing](#5-data-preprocessing)
  - [Model Training](#6-model-training)
  - [Inference and Reconstruction](#7-inference-and-reconstruction)
  - [Anomaly Detection](#8-anomaly-detection)
  - [Evaluation](#9-evaluation)
  - [Anomaly Localization](#10-anomaly-localization)
  - [Visualization](#11-visualization)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Notes](#notes)
- [References](#references)

## Overview

This repository provides end-to-end pipelines for video anomaly detection on the UCSD Pedestrian (Ped1/Ped2) datasets using two deep learning approaches:

- **CNN-LSTM Autoencoder**: Combines convolutional layers for spatial feature extraction and ConvLSTM layers for temporal modeling.
- **3D Convolutional Autoencoder (C3D)**: Uses 3D convolutions to jointly model spatial and temporal features.

Both implementations are based on TensorFlow/Keras and include all steps from data preprocessing to anomaly localization and evaluation.

Both models are trained in an unsupervised manner to reconstruct normal video sequences. Anomalies are detected based on high reconstruction errors.

## Implementation Details

### 1. Environment Setup

- All required libraries are imported at the beginning of each notebook/script, including TensorFlow, Keras, OpenCV, NumPy, Matplotlib, and others.
- Additional utilities for visualization and progress tracking are included.
- Multi-GPU training is supported via TensorFlow’s MirroredStrategy.

### 2. Model Architectures

#### CNN-LSTM Autoencoder

- **Encoder**: Stacked `TimeDistributed Conv2D` layers extract spatial features from each frame in the sequence.
- **Temporal Modeling**: Multiple `ConvLSTM2D` layers capture temporal dependencies across the sequence.
- **Decoder**: `TimeDistributed Conv2DTranspose` layers reconstruct the input sequence from the encoded representation.
- **Activation**: ReLU activations are used throughout, with a final Tanh activation for the output.
- **Normalization**: `BatchNormalization` is applied after each convolutional layer.

#### 3D Convolutional Autoencoder (C3D)

Two variants are provided:
- Lightweight version with fewer parameters.
- Deeper version with more parameters for potentially better performance.

- **Encoder**: Stacked `Conv3D`, `BatchNorm`, `MaxPooling`, and `Dropout` layers.
- **Decoder**: `Conv3DTranspose`, `BatchNorm`, and `Dropout` layers.
- **Output Layer**: Uses a `tanh` activation for reconstruction.

### 4. Loss Function

- The reconstruction loss is defined as the mean squared error (MSE) between the input and the reconstructed output.

### 5. Data Preprocessing

- Video frames are loaded and resized to 224x224.
- **CNN-LSTM**: Frames are grouped into clips of 8 consecutive frames.
- **C3D**: Frames are grouped into clips of 16 consecutive frames.
- All frames are normalized to the [0, 1] range.

### 6. Model Training

- The model is trained on the training set using the defined reconstruction loss.
- Model checkpoints and training logs are saved during training.
- Training supports both from-scratch and resume-from-checkpoint modes.

### 7. Inference and Reconstruction

- The trained model is used to reconstruct test video clips.
- Reconstructed frames are saved for further analysis and visualization.

### 8. Anomaly Detection

- Regularity scores are computed based on the reconstruction error for each frame sequence.
- Frames with regularity scores below a set threshold are flagged as anomalous.
- Regularity plots are generated for each test video.

### 9. Evaluation

- The Area Under the ROC Curve (AUC) is computed for each test video and averaged across the dataset.
- ROC curves are plotted and saved.

### 10. Anomaly Localization

- The spatial location of anomalies is estimated by computing the sum of squared errors in sliding windows over the frame.
- Detected anomaly regions are highlighted with rectangles and overlaid on the original frames.
- Localized anomaly frames are saved and can be compiled into videos.

### 11. Visualization

- Model architecture is visualized and saved.
- Training loss and regularity plots are generated.
- Animations of regularity scores alongside video frames are created for qualitative analysis.

## Getting started

### Prerequisites

This project requires the following dependencies:

- **Programming Language**: Jupyter Notebook

### Installation

Build deep-learning-based-video-anomaly-detection from the source and install dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/supriy09/deep-learning-based-video-anomaly-detection
   ```

2. Navigate to the project directory:
   ```bash
   cd deep-learning-based-video-anomaly-detection
   ```
3. Run the jupiter notebooks

## Notes

- All paths and parameters are configurable in the code.
- The code supports both training from scratch and loading pre-trained models.
- Multi-GPU training is supported.
- The repository is organized for reproducibility and extensibility.

## References

- [UCSD Anomaly Detection Dataset](http://svcl.ucsd.edu/projects/anomaly/dataset.htm)
- [ConvLSTM: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
- [C3D: Tran et al., "Learning Spatiotemporal Features with 3D Convolutional Networks"](https://arxiv.org/pdf/1412.0767)
