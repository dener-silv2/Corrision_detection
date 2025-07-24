# data_visual.py
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import data_loader
#import tensorflow as tf

def show_random_image_pair(images, masks):
    """
    Displays a random pair of an image and its corresponding mask.

    This function selects a random index from the given `images` and `masks` lists, 
    retrieves the image-mask pair, and visualizes them side by side. The function 
    supports both image arrays and file paths to images/masks.

    Args:
        images (list): A list of image data or file paths to images.
        masks (list): A list of corresponding mask data or file paths to masks.

    Returns:
        None: Displays the selected image-mask pair as side-by-side plots.
    """

    def plot_image_or_path(image, mask):
        """
        Handles plotting for either raw image/mask arrays or file paths.

        Args:
            image: Either an image array or a file path to an image.
            mask: Either a mask array or a file path to a mask.

        Returns:
            tuple: Processed (image, mask) arrays ready for visualization.
        """
        if isinstance(image, str):
            image = plt.imread(image)
            mask = plt.imread(mask)
            return image, mask
        else:
            return image, mask
        
    try:
        random_indices = random.randint(0, len(images) - 1)
        
        plt.figure(figsize=(10, 10))
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(plot_image_or_path(images[random_indices], 
                                        masks[random_indices])[i], 
                                        cmap='gray' if i == 1 else None)
            plt.title('Image' if i == 0 else 'Mask')
            plt.axis('off')
        plt.show()
    except Exception as e:
        print(f'\nAn exception was raised: {e}')
        print('No image to show!')

def plot_accuracy_and_loss(history, fold_num):
    """
    Plots accuracy, F1-score, and loss curves based on training history.

    This function dynamically adjusts plotting behavior based on whether it's used 
    for model retraining or cross-validation:
    - **Retraining (`fold_num` as a string)** → Plots accuracy, F1-score, and loss 
      on the same figure using dual y-axes.
    - **Cross-validation (`fold_num` as an integer)** → Creates two subplots, one 
      for accuracy/F1-score and one for loss.

    Args:
        history (tf.keras.callbacks.History): Training history object containing metrics.
        fold_num (Union[int, str]): Fold identifier (integer for cross-validation, 
                                    string for retraining).

    Returns:
        None: Displays the generated plots.

    Example Usage:
        plot_accuracy_and_loss(model_history, fold_num=3)
        plot_accuracy_and_loss(retrain_history, fold_num="Best Model Retraining")

    Plot Details:
    - Accuracy & F1-score (training & validation)
    - Loss (training & validation when available)
    - Dynamic y-axis scaling for better visualization

    """

    if isinstance(fold_num, str):  # Retraining case
        fig, ax1 = plt.subplots(figsize=(12, 4))

        # Plot accuracy and F1-score on the left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy, F1-Score', color=color)
        l1, = ax1.plot(history.history['accuracy'], label='Accuracy', color='tab:blue')
        l2, = ax1.plot(history.history['f1_score'], label='F1 Score', color='tab:green')
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for loss on the right
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Loss', color=color)
        l3, = ax2.plot(history.history['loss'], label='Loss', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Merge legends
        handles = [l1, l2, l3]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc='upper left')

        # Adjust layout and set title
        fig.tight_layout()
        plt.title(f'Fold {fold_num} - Accuracy, F1_score, and Loss')
        plt.show()

    else:  # Cross-validation case (use existing subplot logic)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)  # Subplot for accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['f1_score'], label='Training F1')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(history.history['val_f1_score'], label='Validation F1')
        plt.title(f'Fold {fold_num} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy, F1-Score')
        plt.legend()

        plt.subplot(1, 2, 2)  # Subplot for loss
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold_num} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

'''def load_and_preprocess_image_mask(image, mask, target_size=(128, 128), mask_extension='jpg'):
    """
    Loads and preprocesses an image and its corresponding segmentation mask.

    This function reads an image and its mask from disk, decodes them, converts 
    pixel values to float32, and resizes both to the specified target dimensions.
    
    Args:
        image (str): Path to the image file.
        mask (str): Path to the segmentation mask file.
        target_size (tuple): Desired (height, width) for resizing (default: (128, 128)).
        mask_extension (str): File extension of the mask ('jpg', 'png', etc.).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Preprocessed image and mask tensors.

    Example Usage:
        img, msk = load_and_preprocess_image_mask("image.jpg", "mask.png", target_size=(256, 256), mask_extension="png")
    """
    
    # Load and preprocess image
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)  # Assuming image is always JPG
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [target_size[0], target_size[1]])

    # Load and preprocess mask based on format
    if mask == 0:
        return image
    mask = tf.io.read_file(mask)
    try:
        if mask_extension.lower() == 'png':
            mask = tf.image.decode_png(mask, channels=1)
        else:
            mask = tf.image.decode_jpeg(mask, channels=1)
    except Exception as e:
        print(f"Mask decoder needs to be adjusted in the code: {e}")
    
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
    mask = tf.image.resize(mask, [target_size[0], target_size[1]])

    return image, mask
'''
def visualize_predictions(models_dict, images, masks, num_samples=3):
    """
    Visualizes model predictions on randomly selected sample images.

    This function selects a subset of images and masks, preprocesses them if needed,
    and displays the original image, ground truth mask, and predictions from multiple models.
    The predictions are shown as overlays on the original images.

    Args:
        models_dict (dict): Dictionary containing model names as keys and trained Keras models as values.
        images (list[str] or list[tf.Tensor]): List of image file paths or preloaded image tensors.
        masks (list[str] or list[tf.Tensor]): List of corresponding mask file paths or preloaded mask tensors.
        num_samples (int): Number of random samples to visualize (default: 3).

    Returns:
        None: Displays plots showing original images, ground truth masks, and predictions.

    Example Usage:
        visualize_predictions(trained_models, image_list, mask_list, num_samples=5)
    """
    target_size=list(models_dict.values())[0].input_shape[1:3]

    if num_samples > len(images):
        num_samples = len(images)

    if not isinstance(images[0], str):
        sample_indices = np.random.choice(len(images), num_samples, replace=False)
    else:
        image_paths = images
        mask_paths = masks
        sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)

    for i in sample_indices:
        if not isinstance(images[i], str):
            img = images[i]
            mask = masks[i]
        else:
            img, mask = data_loader.load_and_preprocess_image_mask(image_paths[i], mask_paths[i], target_size=target_size)

        # Plot original image and ground truth mask
        plt.figure(figsize=(20, 5))
        plt.subplot(1, len(models_dict) + 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.subplot(1, len(models_dict) + 2, 2)
        plt.imshow(np.array(mask).squeeze(), cmap="gray")
        plt.title("Ground Truth Mask")

        # Iterate over each model and get predictions
        for j, (model_name, model) in enumerate(models_dict.items()):
            img_j = img
            if not model.input_shape[1:4] == img_j.shape:
                print(model.input_shape[1:4] == img_j.shape)
                img_j = tf.image.resize(img, [model.input.shape[1], model.input.shape[2]])
                
            img_expanded = np.expand_dims(img_j, axis=0)
            predicted_mask = model.predict(img_expanded, verbose=0)[0]

            # Plot the predicted mask as an overlay on the original image
            plt.subplot(1, len(models_dict) + 2, j + 3)
            plt.imshow(img_j)
            plt.imshow(predicted_mask.squeeze(), cmap="jet", alpha=0.5)  
            plt.title(f"Predicted ({model_name})")

        plt.show()