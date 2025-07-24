# # 0. Initial preparations

# ## **0.1. Check Python Version and Define GPU On or Off**
# * GPU always off if the system does not support it (even if ```GPU_ON=True```)
import sys
print(sys.version)

# Set GPU usage
GPU_ON = False

# Define use of LabelMe and Segment Anything Model
LABELME, SAM = True, False

# ## **0.2. Import necessary libraries**
import os
import numpy as np
import subprocess

import matplotlib.pyplot as plt
from PIL import Image

try:
    import tensorflow as tf
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "tensorflow"])
    import tensorflow as tf
  
if not GPU_ON:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

else:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            #tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

    except RuntimeError as e:
        print(e)

# ### 0.2.2. Check if GPU is Available and Confirm Tensorflow Version
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is NOT available.")
print(f'Tensorflow version: {tf.__version__}')

from tensorflow.keras import mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

print("Mixed precision enabled:", mixed_precision.global_policy())


# ## **0.3. Define Root and Sub Directories**
root_of_all = './corrosion_detection'
root_dir = './data/images'
os.makedirs(root_dir, exist_ok=True)

if LABELME:
        labelMe_image_dir = f'{root_dir}/LabelMe_images'
        labelMe_json_dir = f'{root_dir}/LabelMe_json'
        labelMe_mask_dir = f'{root_dir}/LabelMe_masks'

        for dir in [labelMe_image_dir,
                labelMe_json_dir,
                labelMe_mask_dir
                ]:

                os.makedirs(dir, exist_ok=True)

if SAM:
        SAM_image_dir = f'{root_dir}/SAM_images'
        SAM_json_dir = f'{root_dir}/SAM_json'
        SAM_mask_dir = f'{root_dir}/SAM_masks'

        #selected_images = f'{root_dir}/selected_images_and_masks/images'
        #selected_masks = f'{root_dir}/selected_images_and_masks/masks'


        for dir in [SAM_image_dir,
                SAM_json_dir,
                SAM_mask_dir,
                #selected_images,
                #selected_masks
                ]:

                os.makedirs(dir, exist_ok=True)


# ## **0.4. Define Data Parameters**
# Target size is the height and width of images for training, it will be used all over the code
# Usually (64,64), (128,128) or (256,256)

target_size = (256, 256)
mask_extension = '.jpg'


# # 1. Data Loading and Preprocessing
from utils import data_visual as dv
from utils import data_loader as dl

# Load LabelMe Image Data
if LABELME:
    labelMe_images, labelMe_masks = dl.load_and_preprocess(labelMe_image_dir, labelMe_json_dir, labelMe_mask_dir,
                                                    mask_extension=mask_extension, target_size=target_size, load_images=True, folder_percent=1.0)

    dv.show_random_image_pair(labelMe_images, labelMe_masks)

# Load External SAM2 Dataset
if SAM:
    print('\n')
    SAM_images, SAM_masks = dl.load_and_preprocess(SAM_image_dir, SAM_json_dir, SAM_mask_dir,
                                                target_size=target_size, folder_percent=1.0,
                                                mask_extension=mask_extension, SAM=True, load_images=True)
    dv.show_random_image_pair(SAM_images, SAM_masks)


# ## **1.4. Load and Combine Data**
import gc

if LABELME and SAM:
    images = np.concatenate((labelMe_images, SAM_images), axis=0)
    masks = np.concatenate((labelMe_masks, SAM_masks), axis=0)
elif LABELME and not SAM:
    images = np.array(labelMe_images)
    masks = np.array(labelMe_masks)
else:
    images = np.array(SAM_images)
    masks = np.array(SAM_masks)

for var in ['labelMe_images', 'SAM_images', 'labelMe_masks', 'SAM_masks']:
    try:
        del globals()[var]
    except KeyError:
        pass 
gc.collect()

dv.show_random_image_pair(images, masks)

# Determine the size of test set 
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(images, masks, test_size=0.1, random_state=42)


# # 2. Data Augmentation

# Augment functiona are image preprocessor are defined in ```utils/data_loader.py```

# ## **2.2. Create TensorFlow Datasets**

# Create TensorFlow Datasets function is defined in ```utils/data_loader.py```

# # 3. Model Building
# 

# **Models are defined in ```models.py```**<br>
# Available models are:
# * Resnet50
# * InceptionV3
# * Unet AlexNet like
# * HrNet AlexNet like
# * EfficientNetB0
# * MobileNetV2
# * Mask-CNN like
# * Deeplabv3plus with Resnet50

# # 4. Loss Function and Training

# **Loss functions are defined in ```loss_and_metrics.py```**<br>
# Available functions are:
# * Dice_loss
# * Weighted_dice_loss
# * Combined_loss (Dice + Binary Cross Entropy)
# * Soft_f1_loss

# ## **4.2. Train and Evaluate Function (with K-Fold Cross-Validation)**
from utils import models as l_models
from utils.loss_and_metrics import combined_loss, F1Score
from sklearn.model_selection import KFold
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam, AdamW

def train_and_evaluate(X_train_val, y_train_val, input_shape,
                       epochs=5, batch_size=32, k_folds=5, 
                       used_models=None, num_augmentations=5, log_dir="logs/"):
    
    """
    Trains and evaluates different models using K-fold cross-validation, with TensorBoard integration.

    Args:
        X_train_val (np.array): Full dataset features for cross-validation.
        y_train_val (np.array): Full dataset labels for cross-validation.
        input_shape (tuple): Expected input shape for the models.
        epochs (int): Number of epochs per fold. Default is 5.
        batch_size (int): Batch size for training. Default is 32.
        k_folds (int): Number of folds for cross-validation. Default is 5.
        used_models (list): List of model functions to cycle through during folds.
        num_augmentations (int): Number of augmentations applied to training data.
        log_dir (str): Directory for storing TensorBoard logs.

    Returns:
        list: A list of tuples containing (loss, accuracy, model_name, f1_score) for each fold.
    """

    if used_models is None or len(used_models) == 0:
        raise ValueError("Error: `used_models` must contain at least one model function.")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
        print(f"\n============ Fold {fold + 1}/{k_folds} ============\n")
        X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
        y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]

        with tf.device("/CPU:0"):
            train_dataset = dl.create_dataset(X_train_fold, y_train_fold, batch_size, input_shape, augment_data=True, num_augmentations=num_augmentations, mask_extension=mask_extension)
            val_dataset = dl.create_dataset(X_val_fold, y_val_fold, batch_size, input_shape, augment_data=False, mask_extension=mask_extension)

        # Select model based on fold index
        model_function = used_models[fold % len(used_models)]
        model = l_models.get_model(model_function, input_shape)

        fold_log_dir = os.path.join(log_dir, f"fold_{fold + 1}")  
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fold_log_dir, histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3, verbose=1)
        steps_per_epoch = (len(X_train_fold) * (1 + num_augmentations)) // batch_size
        validation_steps = len(X_val_fold) // batch_size
        if len(X_val_fold) % batch_size != 0:
            validation_steps += 1
        callbacks = [tensorboard_callback, early_stopping, lr_schedule]

        print(f"\nTraining dataset size: {len(X_train_fold) * num_augmentations}")
        print(f"Validation dataset size: {len(X_val_fold)}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}\n")
        
        model.compile(optimizer=Adam(), loss=combined_loss, metrics=['accuracy', F1Score()])
        history = model.fit(train_dataset, validation_data=val_dataset,
                            epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                            verbose=1, callbacks=callbacks)
                            
        dv.plot_accuracy_and_loss(history, fold + 1)

        loss, accuracy, f1 = model.evaluate(val_dataset, verbose=1, steps=validation_steps)
        fold_metrics.append((loss, accuracy, model_function[6:],f1))

        print(f"\nFold {fold + 1} - {model_function[6:]} - Validation Loss: {loss:.4f}, Validation Accuracy: {(accuracy * 100):.2f}%, Validation F1 Score: {f1:.4f}\n")
        clear_session()
        #print(model.summary())

    return fold_metrics


# # 5. Execution and Model Selection
# 
# ## **5.1. Define Input Shape and Run Training**
from tensorflow.keras.backend import clear_session


augmented_copies = 2
epochs=10
batch_size=8
used_models = [ 'build_hrnet_alexnet',
                'build_unet_alexnet',
                #'build_unet_resnet',
                #'build_unet_inception',
                #'build_unet_efficientnet',
                #'build_unet_mobilenet',
                'build_mask_cnn',
                #'build_deeplabv3plus'
                #'build_hrnet_alexnet'
              ]
folds=len(used_models)

if GPU_ON:
    # Warm-up GPU: run several dummy operations
    for _ in range(10):
        _ = tf.matmul(tf.random.uniform([512, 512]), tf.random.uniform([512, 512]))

input_shape = (target_size[0], target_size[1], 3)
fold_metrics = train_and_evaluate(X_train_val,
                                  y_train_val,
                                  input_shape,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  k_folds=folds,
                                  used_models=used_models,
                                  num_augmentations=augmented_copies)


# ### 5.1.1. Create Tensorboardwith Information about the Cross-Validation Training and Validation 
# Once run this cell will create a tensorboard for info related to training and validation performed in the previous cell
# It should e displayed in extension server, it may ask to open an external resource
"""
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs/')
"""

# ## **5.2. Select Best Model Based on Validation Accuracy**
best_fold_index = np.argsort([-f1 for _, _, _, f1 in fold_metrics])[:3] 
for idx in best_fold_index:
    print(f"Best Fold: n° {idx + 1} - Model: {fold_metrics[idx][2]}\nF1: {(fold_metrics[idx][3] * 100):.2f}%\n")


# ## **5.3. Retrain 2 Best Model on Full Training Data (excluding chosen validation fold)**
from utils.loss_and_metrics import combined_loss, F1Score
from utils import models as l_models

model_b, history_b = [], []

def retrain_best_model(X_train_val, y_train_val, best_fold_index, input_shape, id,  epochs=100, batch_size=32):
    """
    Retrains the best-selected model using the full training dataset with augmentation.

    This function retrieves the best-performing model based on `best_fold_index`,
    compiles it with an optimizer and loss function, and trains it using an augmented dataset.
    It also integrates key callbacks for efficient model training.

    Args:
        X_train_val (np.array or tf.Tensor): Full training images dataset.
        y_train_val (np.array or tf.Tensor): Full training masks/labels dataset.
        best_fold_index (int): Index identifying the best model from previous cross-validation.
        input_shape (tuple): The expected input shape of the model.
        id (int or str): Identifier used for checkpoint naming.
        epochs (int): Number of training epochs (default: 100).
        batch_size (int): Number of samples per batch (default: 32).

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: 
        - The retrained best model.
        - Training history containing loss, accuracy, and F1-score progression.

    Training Details:
    - Loads the best model based on validation performance.
    - Applies data augmentation (`num_augmentations=5`) for increased robustness.
    - Uses Adam optimizer with a learning rate of `1e-2`.
    - Implements `EarlyStopping` to prevent overfitting.
    - Saves checkpoints dynamically to `"models/best_model_checkpoints_{i}.keras"`.
    - Reduces learning rate (`ReduceLROnPlateau`) when loss stagnates.

    Example Usage:
        model, history = retrain_best_model(X_train, y_train, best_fold_index=3, 
                                            input_shape=(128, 128, 3), i=3, epochs=50, batch_size=16)
    """
    
    best_used_model = l_models.get_model(used_models[best_fold_index], input_shape)
    best_used_model.name = id
    X_train_full = X_train_val
    y_train_full = y_train_val
    num_augmentations = 5
    train_dataset_full = dl.create_dataset(X_train_full, y_train_full, batch_size=batch_size, input_shape=input_shape, augment_data=True, num_augmentations=num_augmentations, mask_extension=mask_extension)

    #model_best = best_used_model(input_shape)
    model_best = best_used_model
    model_best.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                       loss=combined_loss, metrics=['accuracy', F1Score()])
    
    steps_per_epoch = (len(X_train_full) * (1 + num_augmentations)) // batch_size
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"models/best_model_checkpoints_{id}.keras", monitor="loss", save_best_only=True, verbose=1)
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=5, verbose=1)

    history = model_best.fit(train_dataset_full, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch,
                             callbacks=[early_stopping, lr_schedule, checkpoint])
    return model_best, history

for i in best_fold_index:
    print(f"Model no. {i+1}")
    best_model, history = retrain_best_model(X_train_val, y_train_val, best_fold_index=i, input_shape=input_shape, id=fold_metrics[i][2])
    model_b.append(best_model)
    history_b.append(history)
    dv.plot_accuracy_and_loss(history, fold_num="Retraining")
    clear_session()
    print("\n")


# ### 5.3.1. Print the Scheme of each of the Retrained Model Layers
from tensorflow.keras.utils import plot_model
try:
    import pydot
    import graphviz
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "pydot"])
    subprocess.run(["pip", "install", "graphviz"])
    import pydot
    import graphviz

for i in range(len(best_fold_index)):
    plot_model(model_b[i], to_file=f'{root_dir}/model_{model_b[i].name}.png', show_shapes=False, show_layer_names=False)


# ### 5.3.2. Plot Retraining Metrics over Epochs
for i in range(len(best_fold_index)):
    print(model_b[i].name)
    dv.plot_accuracy_and_loss(history_b[i], fold_num="Retraining")


# ## **5.4. Evaluate on Test Set**
def evaluate_test_set(model, X_test, y_test, batch_size=32):
    """
    Evaluates a trained model on a separate test set and prints key metrics.

    This function evaluates the model using a test dataset without augmentation, 
    calculates validation steps dynamically based on batch size, and reports 
    test loss, accuracy, and F1-score.

    Args:
        model (tf.keras.Model): Trained Keras model to be evaluated.
        X_test (np.array or tf.Tensor): Test dataset images.
        y_test (np.array or tf.Tensor): Corresponding ground truth masks or labels.
        batch_size (int): Number of samples per batch (default: 32).

    Returns:
        None: Prints evaluation results including test loss, accuracy, and F1-score.

    Example Usage:
        evaluate_test_set(trained_model, X_test, y_test)
    """

    test_dataset = dl.create_dataset(X_test, y_test, input_shape=input_shape, augment_data=False, batch_size=batch_size)
    validation_steps = len(X_test) // batch_size
    if len(X_test) % batch_size != 0:
            validation_steps += 1

    loss, accuracy, f1 = model.evaluate(test_dataset, steps=validation_steps, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {100*accuracy:.2f}%, Test F1_score: {100*f1:.2f}%")
    
#evaluate_test_set(best_model, X_test, y_test)
for i in range(len(best_fold_index)):
    print(model_b[i].name)
    evaluate_test_set(model_b[i], X_test, y_test)

# ## **5.5. Visualize Predictions**
from utils import data_visual as dv
from utils.loss_and_metrics import combined_loss, F1Score

model_names = [used_models[0][6:], used_models[1][6:]]
model_paths = [f"models/best_model_checkpoints_{name}.keras" for name in model_names]

best_models = {model_names[0]: tf.keras.models.load_model(model_paths[0], custom_objects={'dice_loss': combined_loss, 'F1Score': F1Score}),
               model_names[1]: tf.keras.models.load_model(model_paths[1], custom_objects={'dice_loss': combined_loss, 'F1Score': F1Score}),
               }

dv.visualize_predictions(best_models, X_test, y_test)


# # **6. Test the best models in a set of chosen images**

#X_test_selected = np.array([os.path.join(selected_images, img) for img in os.listdir(selected_images) if img.endswith('.jpg')])
#y_test_selected = np.array([os.path.join(selected_masks, mask) for mask in os.listdir(selected_masks) if mask.endswith(mask_extension)])

#visualize_predictions(best_models, X_test_selected, y_test_selected, num_samples=X_test_selected.shape[0])


# # **7. Remount Drone Images**
# Remount tiled drone images
import random

model_paths = [f"models/best_model_checkpoints_{name}.keras" for name in model_names]

best_models = {model_names[0]: tf.keras.models.load_model(model_paths[0], custom_objects={'dice_loss': combined_loss, 'F1Score': F1Score}),
               model_names[1]: tf.keras.models.load_model(model_paths[1], custom_objects={'dice_loss': combined_loss, 'F1Score': F1Score})
              }

tile_identifier = 'dji'
mask_dir = f'{root_dir}/LabelMe_masks/predicted'
os.makedirs(mask_dir, exist_ok=True)
files = [f for f in os.listdir(labelMe_image_dir) if f.startswith(tile_identifier) and f.endswith('.jpg') and 'tile' not in f]

for filename in random.sample(files, min(10, len(files))):
    for n_model in len(model_names):
        prediction_model = best_models[n_model]
        filename_path = os.path.join(labelMe_image_dir, filename)
        json_path = os.path.join(labelMe_image_dir, f'{filename[:-4]}_tile_info.json')
        original_mask = os.path.join(labelMe_mask_dir, filename)
        image_path = os.path.join(labelMe_mask_dir, filename)
        output_path = f'{mask_dir}/remounted_{prediction_model[n_model].name}_{filename}'

        for img in [f for f in os.listdir(labelMe_image_dir) if f.startswith(filename[:-4] + '_tile_') and f.endswith('.jpg')]:
            input_image = os.path.join(labelMe_image_dir, img)
            output_image = os.path.join(mask_dir, img)
            dl.predict_and_save(input_image, prediction_model, output_image)

        image_tiler.remount_masks(filename_path, json_path, original_mask, mask_dir, filename, output_path, plot_and_save=True)


# # **8. Remount Any Chosen Images**
from utils import image_tiler
from utils import data_loader as dl

prediction_model = tf.keras.models.load_model("models/best_model_checkpoints_mask_cnn_all.keras", custom_objects={'dice_loss': combined_loss, 'F1Score': F1Score})

set_image_path_new = ["/home/hrodrigues/Machine_Learning_Projects/bridge_corrosion_detection/images/bridge_images/dji_fly_20241119_135450_0174_1732024899748_timed.jpg",
                      "/mnt/d/Fotogrametria/Edifícios UA/Ponte crasto/Ponte Crasto photos/13-11/dji_fly_20241113_145346_0079_1731510996812_photo.jpg",
                      "/mnt/c/Users/hrodrigues/Downloads/WhatsApp Image 2025-05-21 at 12.13.31_61f44328.jpg",
                      "/mnt/c/Users/hrodrigues/Downloads/images.jpg"
                     ]
mask_dir = './data/images/Random_images'

dl.remount_any_image(set_image_path_new, mask_dir, prediction_model)
dl.remount_any_image(set_image_path_new, mask_dir, prediction_model, tiles=False)

