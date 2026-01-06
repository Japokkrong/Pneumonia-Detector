# Pneumonia Detector

A deep learning project using Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle.

- **Total Images**: 5,863 X-Ray images (JPEG)
- **Categories**: 2 (Pneumonia/Normal)
- **Organization**: 3 folders (train, test, val) with subfolders for each category
- **Source**: Chest X-ray images (anterior-posterior) from pediatric patients (1-5 years old) from Guangzhou Women and Children's Medical Center, Guangzhou

## Model Architecture

The model is based on a modified VGG-8 architecture with the following features:
- 5 convolutional layers with max pooling
- Dual pathway architecture combining two feature streams
- Fully connected layers for classification
- Output: 2 classes (Pneumonia/Normal)

## Model Improvements

The model has been enhanced with the following optimizations:
- **Weight Decay**: Added L2 regularization (weight_decay=1e-4) to prevent overfitting
- **Learning Rate**: Optimized to 3e-4
- **Learning Rate Scheduler**: ReduceLROnPlateau with improved parameters (patience=3, factor=0.5)
- **Class Weights**: Balanced loss function to handle class imbalance

## Results

- **Accuracy**: 88.3%

The improved model achieved 88.3% accuracy by incorporating weight decay regularization, which helps prevent overfitting and improves generalization on the test set.

### Additional Evaluation Metrics

For comprehensive model evaluation, the notebook includes:
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives
- **Sample Predictions**: Random samples showing both correct and incorrect classifications
- **Individual Image Predictions**: Visual evaluation of model predictions on test images

*See the notebook for detailed evaluation visualizations and metrics.*

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL

## Usage

1. Install the required dependencies
2. Download the dataset from Kaggle
3. Open `PneumoniaDetector.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells to train and evaluate the model

## Training Details

- **Optimizer**: Adam with weight decay
- **Loss Function**: CrossEntropyLoss with class weights
- **Batch Size**: 16
- **Image Size**: 224x224 pixels