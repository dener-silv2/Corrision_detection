# Corrosion Detection Pipeline for Bridge Assessment

## Overview

This repository contains a comprehensive Python pipeline for **image recognition and semantic segmentation** to detect and segment corrosion in images of bridges. The pipeline leverages **deep learning** techniques to automate the assessment of infrastructure health, with a particular focus on identifying and tracing specific features within bridge imagery.

## Usage

### Requirements

* Python 3.x

* Required Python packages (installable via `pip`):

  * `tensorflow`

  * `numpy`

  * `matplotlib`

  * `Pillow` (PIL)

  * `scikit-learn`

  * `opencv-python`

  * Additional packages for `utils` modules (e.g., `image_tiler`, `data_loader`, `data_visual`, `models`, `loss_and_metrics`) which are part of the project structure.

### Instructions

1. **Clone the repository:**

```

git clone [https://github.com/dener-silv2/Corrision_detection(https://www.google.com/search?q=https://github.com/dener-silv2/Corrision_detection.git)

```


2. **Navigate to the repository directory:**

```

cd .\\Corrision_detection

```


3. **Install required packages:**

```

pip install -r requirements.txt

```


## The Main Script

> [**main_pipeline.ipynb**](https://github.com/dener-silv2/Corrision_detection/blob/main/corrosion_notebook.py)

## Script Overview

The main script orchestrates the entire vegetation detection pipeline, performing the following key steps:

1. **Initial Preparations:**

* Checks Python version and configures GPU usage (or disables it).

* Imports necessary libraries including TensorFlow.

* Defines root and sub-directories for images, JSON annotations (e.g., LabelMe, SAM), and masks.

* Sets global data parameters like `target_size` for image processing.

2. **Data Loading and Preprocessing:**

* Loads and preprocesses image and mask data from specified directories (e.g., LabelMe, SAM datasets).

* Combines datasets and performs a train/validation/test split using `sklearn.model_selection.train_test_split`.

3. **Data Augmentation:**

* Applies various data augmentation techniques to the training dataset to enhance model robustness and generalisation.

4. **Model Building:**

* Utilises a modular approach to build and load various deep learning segmentation models.

* Supports architectures such as U-Net-like (e.g., AlexNet-based), Mask-CNN, and HRNet, among others.

5. **Training (with K-Fold Cross-Validation):**

* Trains models using **K-Fold Cross-Validation** to ensure robust evaluation.

* Integrates callbacks like `EarlyStopping` and `ReduceLROnPlateau` for efficient training.

* Logs training metrics to TensorBoard for visualisation.

6. **Execution and Model Selection:**

* Runs the training and evaluation process across defined folds.

* Selects the best-performing models based on validation metrics (e.g., F1-Score).

* Retrains the top models on the full training data for final deployment.

7. **Evaluation on Test Set:**

* Evaluates the retrained models on a held-out test set to report final performance metrics (loss, accuracy, F1-Score).

8. **Visualize Predictions:**

* Generates and displays visual comparisons of original images, ground truth masks, and model predictions on test samples.

9. **Remount Drone Images / Any Chosen Images:**

* Includes functionality to process and remount tiled drone images or any chosen images, allowing for full-scale prediction visualisation on large, stitched imagery.

## License

This project is licensed under the MIT License - see the [LICENSE.md]() file for details
```
