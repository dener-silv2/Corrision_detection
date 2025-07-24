# data_loader.py
import os
import json
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils import image_tiler
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
try:
  from tqdm import tqdm
except ModuleNotFoundError:
  subprocess.run(["pip", "install", "tqdm"]) # type: ignore
  from tqdm import tqdm

def _parse_labelme_core(json_path, image_path, mask_path):
    """Create and save a LabelMe mask, return PIL objects (no resizing)."""
    with open(json_path) as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    mask = Image.new("L", img.size, 0)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "polygon":
            pts = [tuple(p) for p in shape["points"]]
            ImageDraw.Draw(mask).polygon(pts, outline=255, fill=255)

    mask.save(mask_path)
    return img, mask

def load_all_coco(json_dir):
    """
    Read every .json in json_dir, merge 'images', 'annotations' & 'categories'
    into one dict, then instantiate a COCO object around it.
    """
    merged = {"images": [], "annotations": [], "categories": []}
    cat_idx = {}

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue
        data = json.load(open(os.path.join(json_dir, fname)))
        # Merge categories (dedupe by id)
        for c in data.get("categories", []):
            if c["id"] not in cat_idx:
                cat_idx[c["id"]] = c
        # Extend images + annotations
        merged["images"].extend(data.get("images", []))
        merged["annotations"].extend(data.get("annotations", []))

    merged["categories"] = list(cat_idx.values())

    coco = COCO()                 # empty COCO
    coco.dataset = merged         # inject our merged dict
    coco.createIndex()            # build internal indices
    return coco

def _parse_coco_core(json_path, image_path, mask_path):
    """Create and save a COCO mask, return PIL objects (no resizing)."""

    coco = json_path
    if coco is None:
        raise RuntimeError("You must call load_all_coco(...) before using COCO mode!")

    fn = os.path.basename(image_path)
    # find the image in COCO
    img_ids = coco.getImgIds(imgIds=[], file_name=fn)
    if not img_ids:
        raise ValueError(f"COCO annotation has no entry for {fn}")
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids[0]))

    img = Image.open(image_path).convert("RGB")
    mask = Image.new("L", img.size, 0)

    for ann in anns:
        ann_m = coco.annToMask(ann)
        ann_m = Image.fromarray(ann_m.astype("uint8"))  # still 0/1
        mask = Image.fromarray(
            np.maximum(np.array(mask), np.array(ann_m))
        )

    mask.save(mask_path)
    return img, mask

def _parse_sam2_core(json_path, image_path, mask_path):
    """Create and save a SAM2 RLE mask, return PIL objects (no resizing)."""
    with open(json_path) as f:
        data = json.load(f)

    annos = data.get("annotations", [])
    if not annos:
        # nothing to annotate
        return None, None

    img = Image.open(image_path).convert("RGB")
    mask = Image.new("L", img.size, 0)
    mask_arr = np.array(mask)

    for ann in annos:
        if "segmentation" in ann:
            rle = ann["segmentation"]
            dec = maskUtils.decode(rle).squeeze()  # H×W
            mask_arr[dec > 0] = 255

    mask = Image.fromarray(mask_arr.astype("uint8"))
    mask.save(mask_path)
    return img, mask
        

def process_file(filename, image_dir, json_dir, mask_dir, target_size, dataset_type, load_images, mask_extension, images, masks, image_paths, mask_paths, possible_image_extension):
    try:
        for ext in possible_image_extension:
            if not filename.endswith(ext):
                continue

            json_name = filename.replace(ext, '.json')
            mask_name = filename.replace(ext, mask_extension)

            image_path = os.path.join(image_dir, filename)
            if dataset_type == 'COCO' and not isinstance(json_dir, str):
                json_path = json_dir  
            else:
                json_path = os.path.join(json_dir, json_name)
                mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(image_path):
                print(f"Skipping {filename}: no image")
                return

            if not (os.path.exists(mask_path) or os.path.exists(json_path) or isinstance(json_path, str)):
                print(f"Skipping {filename}: no mask and no JSON")
                return

            img, mask_img = parse_json(json_path, image_path, mask_path, dataset_type, load_images)
            if img is not None and mask_img is not None:
                process_image_and_mask(img, mask_img, filename, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension)
    except Exception as e:
            print(f"Skipping {filename}: {e}")

PARSERS = {
    "LabelMe": _parse_labelme_core,
    "COCO":    _parse_coco_core,
    "SAM":     _parse_sam2_core,
}

def parse_json(json_path, image_path, mask_path, dataset_type, load_images, target_size):
    """
    1) If mask already exists, either return file-paths or load+return arrays.
    2) Otherwise dispatch to the correct core parser, save the mask, then return.
    """
    # If masks are already on disk
    if os.path.exists(mask_path):
        if not load_images:
            return image_path, mask_path
        else:
            img = Image.open(image_path).convert("RGB").resize(target_size)
            mask = Image.open(mask_path).convert("L").resize(target_size)
            return (np.array(img).astype("float32") / 255,
                    np.array(mask).astype("uint8"))

    # If they need to be generated 
    if dataset_type not in PARSERS:
        raise ValueError(f"Unknown dataset_type {dataset_type!r}")

    core = PARSERS[dataset_type]
    img_pil, mask_pil = core(json_path, image_path, mask_path)
    if img_pil is None or mask_pil is None:
        return None, None

    # Respect load_images flag
    if not load_images:
        return image_path, mask_path
    img_r = img_pil.resize(target_size)
    mask_r = mask_pil.resize(target_size)
    return (np.array(img_r).astype("float32") / 255,
            np.array(mask_r).astype("uint8"))

def process_image_and_mask(img, mask_img, filename, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension):
    if not 'dji' in filename:
        if load_images:
            img = Image.fromarray(img).resize(target_size)
            mask_img = Image.fromarray(mask_img).resize(target_size)
            images.append(np.array(img))
            masks.append(np.array(mask_img))
        else:
            image_paths.append(img)
            mask_paths.append(mask_img)
    else:
        tile_info_name = f"{filename[:-4]}_tile_info.json"
        tile_info_path = os.path.join(image_dir, tile_info_name)
        if not os.path.exists(tile_info_path):
            tiles, tiles_info = image_tiler.process_image(img, mask_img, target_size)
            save_tiles(tiles, tiles_info, filename, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension)
            with open(tile_info_path, 'w') as file:
                json.dump(tiles_info, file, indent=4)
        else:
            load_existing_tiles(tile_info_name, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension)

def save_tiles(tiles, tiles_info, filename, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension):
    for i, (image_tile, mask_tile) in enumerate(tiles):
        tiles_info[3]["Tile Coordinates"][i]["ID"] = i
        tile_image_name = f"{filename[:-4]}_tile_{i}.jpg"
        tile_image_path = os.path.join(image_dir, tile_image_name)
        if not os.path.exists(tile_image_path):
            Image.fromarray(image_tile).save(tile_image_path)
        tile_mask_name = f"{filename[:-4]}_tile_{i}{mask_extension}"
        tile_mask_path = os.path.join(mask_dir, tile_mask_name)
        if not os.path.exists(tile_mask_path):
            Image.fromarray(mask_tile).save(tile_mask_path)
        if load_images:
            image_tile = Image.fromarray(image_tile).resize(target_size)
            mask_tile = Image.fromarray(mask_tile).resize(target_size)
            images.append(np.array(image_tile))
            masks.append(np.array(mask_tile))
        else:
            image_paths.append(tile_image_path)
            mask_paths.append(tile_mask_path)

def load_existing_tiles(tile_info_name, image_dir, mask_dir, target_size, load_images, images, masks, image_paths, mask_paths, mask_extension):
    tiled_image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.startswith(tile_info_name[:-9]) and file.endswith(".jpg")]
    tiled_mask_paths = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.startswith(tile_info_name[:-9]) and file.endswith(mask_extension)]
    if load_images:
        for img, mask in zip(tiled_image_paths, tiled_mask_paths):
            image_tile = Image.open(img).convert('RGB').resize(target_size)
            mask_tile = Image.open(mask).convert('L').resize(target_size)
            mask_tile = np.expand_dims(mask_tile, axis=-1)
            images.append(np.array(image_tile))
            masks.append(np.array(mask_tile))
    else:
        image_paths.extend(tiled_image_paths)
        mask_paths.extend(tiled_mask_paths)

def finalize_output(images, masks, image_paths, mask_paths, load_images, image_dir):
    if load_images:
        images = np.array(images, dtype=np.float32) / 255.0
        masks = [np.expand_dims(mask, axis=-1) if len(mask.shape) == 2 else mask for mask in masks]
        masks = np.ceil(np.array(masks, dtype=np.float32) / 255.0)
        if masks[0].shape[-1] == 3:
            masks = np.mean(masks, axis=-1, keepdims=True)
        print(f"Loaded {len(images)} images and masks from {os.path.basename(image_dir)}.")
        return images, masks
    else:
        print(f"\nLoaded {len(image_paths)} paths of images and masks from {os.path.basename(image_dir)}.")
        return image_paths, mask_paths

def handle_error(load_images):
    if load_images:
        return np.array([]), np.array([])
    else:
        return [], []

def load_and_preprocess(image_dir, json_dir, mask_dir, target_size,
                        dataset_type, folder_percent=1.0,
                        load_images=False, mask_extension='.jpg'):
    """
    Loads and preprocesses images and masks from the specified directories.

    Args:
        image_dir (str): Directory containing the images.
        json_dir (str): Directory containing the JSON annotations.
        mask_dir (str): Directory to save the generated masks.
        target_size (tuple): Target size for resizing images and masks.
        dataset_type (str): Type of dataset format. Must be one of: 'SAM', 'LabelMe', 'COCO'.
        folder_percent (float, optional): Percentage of files to read from the directory. Default is 1.0.
        load_images (bool, optional): Whether to load image/mask arrays into memory. Default is False.
        mask_extension (str, optional): File extension for saved masks. Default is '.jpg'.

    Returns:
        tuple: Depending on the value of load_images:
            - If load_images is True:
                - images (numpy.ndarray): Array of loaded images.
                - masks (numpy.ndarray): Array of loaded masks.
            - If load_images is False:
                - image_paths (list): List of image file paths.
                - mask_paths (list): List of mask file paths.
    """

    if dataset_type not in PARSERS:
        raise ValueError("dataset_type must be one of: " + ", ".join(PARSERS))

    os.makedirs(mask_dir, exist_ok=True)
    images, masks, image_paths, mask_paths = [], [], [], []
    exts = (".jpg", ".JPG", ".jpeg", ".JPEG")

    if folder_percent <= 0:
        return handle_error(load_images)
    folder_percent = min(1.0, folder_percent)    

    try:
        all_files = [f for f in os.listdir(image_dir) if f.endswith(exts)]
        n_read = int(len(all_files) * folder_percent)
        to_read = random.sample(all_files, n_read)

        for fn in tqdm(to_read, desc=f"Loading {os.path.basename(image_dir)}"):
            process_file(fn,
                        image_dir, json_dir, mask_dir,
                        target_size, dataset_type,
                        load_images, mask_extension,
                        images, masks, image_paths, mask_paths,
                        exts)

        return finalize_output(images, masks, image_paths, mask_paths, load_images, image_dir)

    except Exception as e:
        print(f"Error while loading: {e}")
        return handle_error(load_images)

#================= After test split =====================

def load_and_preprocess_image_mask(image, mask, target_size=(128, 128), mask_extension='jpg'):
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


def augment(image, mask, num_augmentations=5):
    """
    Applies multiple random augmentations to an image and its corresponding mask.

    This function performs a variety of transformations to improve model generalization 
    and robustness, including flipping, rotation, brightness/contrast adjustments, 
    zooming, noise addition, and color modifications.

    Args:
        image (tf.Tensor): The input image tensor with shape (height, width, channels).
        mask (tf.Tensor): The corresponding segmentation mask tensor with shape (height, width, 1).
        num_augmentations (int): The number of augmented versions to generate (default: 5).

    Returns:
        Generator yielding tuples of (augmented_image, augmented_mask).

    Augmentations Applied:
    - Random Horizontal & Vertical Flips
    - Random 90-degree Rotation
    - Brightness & Contrast Adjustments
    - Random Zoom (Central Crop & Resize)
    - Gaussian Noise Injection
    - Random Hue & Saturation Modifications
    - Final Brightness & Contrast Adjustments
    - Clipping to Ensure Pixel Value Range (0.0 - 1.0)

    Example Usage:
        augmented_data = augment(image, mask, num_augmentations=3)
        for img, msk in augmented_data:
            process(img, msk)
    """

    augmentations = []
    for _ in range(num_augmentations):

        image_aug = image
        mask_aug = mask

        # Random Horizontal Flip:
        if tf.random.uniform(()) > 0.5:
            image_aug = tf.image.flip_left_right(image_aug)
            mask_aug = tf.image.flip_left_right(mask_aug)

        # Random Vertical Flip:
        if tf.random.uniform(()) > 0.5:
            image_aug = tf.image.flip_up_down(image_aug)
            mask_aug = tf.image.flip_up_down(mask_aug)

        # Random 90-degree Rotations:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image_aug = tf.image.rot90(image_aug, k=int(k))
        mask_aug = tf.image.rot90(mask_aug, k=int(k))

        # Random Brightness Adjustment:
        image_aug = tf.image.random_brightness(image_aug, max_delta=0.2)

        # Random Contrast Adjustment:
        image_aug = tf.image.random_contrast(image_aug, lower=0.8, upper=1.2)

        # Random Zoom (Central Crop and Resize):
        zoom_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
        zoom_factor = tf.clip_by_value(zoom_factor, 0.0, 1.0)  # Clamp to maximum of 1.0
        image_aug = tf.image.central_crop(image_aug, zoom_factor)
        mask_aug = tf.image.central_crop(mask_aug, zoom_factor)
        image_aug = tf.image.resize(image_aug, tf.shape(image)[:2])
        mask_aug = tf.image.resize(mask_aug, tf.shape(mask)[:2])

        # Random Gaussian Noise:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
        image_aug = tf.clip_by_value(image_aug + noise, 0.0, 1.0)

        # Random Hue Adjustment:
        image_aug = tf.image.random_hue(image_aug, max_delta=0.08)

        # Random Saturation Adjustment:
        image_aug = tf.image.random_saturation(image_aug, lower=0.6, upper=1.6)

        # Random Brightness Adjustment (again, after color changes):
        image_aug = tf.image.random_brightness(image_aug, max_delta=0.25)

        # Random Contrast Adjustment (again, after color changes):
        image_aug = tf.image.random_contrast(image_aug, lower=0.7, upper=1.3)

        # Clip Pixel Values:
        image_aug = tf.clip_by_value(image_aug, 0.0, 1.0)

        if not image_aug.dtype == "tf.float32":
            image_aug = tf.convert_to_tensor(image_aug, dtype=tf.float32)
        if not mask_aug.dtype == "tf.float32":
            mask_aug = tf.convert_to_tensor(mask_aug, dtype=tf.float32)

        augmentations.append((image_aug, mask_aug))

    for aug in augmentations:
        yield aug


def create_dataset(images, masks, batch_size, input_shape, augment_data=True, num_augmentations=5, mask_extension='jpg'):
    """
    Creates a TensorFlow dataset generator that loads, preprocesses, and optionally augments images and masks.

    This function dynamically generates a TensorFlow dataset using a Python generator. It handles image 
    and mask preprocessing, resizing, batching, and augmentation when enabled.

    Args:
        images (list[str] or list[tf.Tensor]): List of file paths or pre-loaded image tensors.
        masks (list[str] or list[tf.Tensor]): List of file paths or pre-loaded mask tensors.
        batch_size (int): Number of samples per batch.
        input_shape (list[int]): Shape of images for training.
        augment_data (bool): Whether to apply data augmentation (default: True).
        num_augmentations (int): Number of augmented samples per image-mask pair (default: 5).

    Returns:
        tf.data.Dataset: A TensorFlow dataset that yields batches of (image, mask) pairs.

    Example Usage:
        dataset = create_dataset(image_paths, mask_paths, batch_size=16, augment_data=True, num_augmentations=3)
        for batch in dataset:
            images_batch, masks_batch = batch
    """
    def data_generator():
        """
        Generates preprocessed and optionally augmented image-mask pairs for training.

        This function iterates through provided image and mask datasets, loading and 
        resizing them while applying augmentations if enabled.

        Yields:
            Tuple[tf.Tensor, tf.Tensor]: Resized (image, mask) pairs, including augmented versions when applicable.
        """
        while True:
            for image, mask in zip(images, masks):
                # Check if inputs are file paths or already-loaded tensors
                if isinstance(image, str):
                    image, mask = load_and_preprocess_image_mask(image, mask, target_size, mask_extension=mask_extension)
                else:
                    if not tf.is_tensor(image):
                        image = tf.convert_to_tensor(image, dtype=tf.float32)
                    if not tf.is_tensor(mask):
                        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

                # Yield the original image and mask
                yield tf.image.resize(image, target_size), tf.image.resize(mask, target_size)

                # Apply augmentations if required
                if augment_data:
                    for augmented_image, augmented_mask in augment(image, mask, num_augmentations):

                        yield augmented_image, augmented_mask

    # Create the dataset from the generator
    target_size=(input_shape[0], input_shape[1])
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32)
        )
    )

    # Batch and prefetch the dataset for efficiency
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    if augment_data:
        dataset = dataset.repeat()

    return dataset

#========================= Testing other images ===================

def predict_and_save(image, model, output_path, mask=0, target_size=(256,256)):
    """
    Predicts a mask from an input image using a trained deep learning model and saves the result.

    Args:
        image (str or np.array): The input image. If a string, it is assumed to be a file path and will be loaded.
        model (tf.keras.Model): The trained model used for mask prediction.
        output_path (str): The path where the predicted mask will be saved.

    Process:
        - If `image` is a file path (string), it loads and preprocesses the image using `dl.load_and_preprocess_image_mask()`.
        - Predicts the mask using the model.
        - Saves the predicted mask as a `.npy` file.

    Returns:
        None (Saves the predicted mask to `output_path[:-4]+'.npy'`)
    """

    if isinstance(image, str):
        image = load_and_preprocess_image_mask(image, mask, target_size)

    image = tf.image.resize(image, [model.input.shape[1], model.input.shape[2]])
        
    img_expanded = np.expand_dims(image, axis=0)
    predicted_mask = model.predict(img_expanded, verbose=0)[0]

    np.save(output_path[:-4]+'.npy', predicted_mask.squeeze())

def remount_any_image(images, mask_dir, model, tiles=True):
    """
    Processes images by either applying a tile-based prediction and mask remounting strategy or performing
    a whole-image prediction with visualization, depending on the 'tiles' flag.

    When tiles is True:
      - The function iterates over each image file in the provided collection.
      - Extracts the file name from the image path.
      - Generates image tiles by calling `image_tiler.process_image` with a target size derived from
        `model.input_shape` (using dimensions at indices 1 and 2).
      - For each tile, it applies predictions using `predict_and_save` and saves the resulting tile image.
      - Remounts the processed tiles into a composite final mask image using `image_tiler.remount_masks`.
      - Saves both the tile images and the final remounted image into the specified `mask_dir`.

    When tiles is False:
      - The function reads each image using `plt.imread`.
      - Resizes the image to the dimensions required by `model.input` (using dimensions at indices 1 and 2).
      - Expands dimensions to create a batch and generates a predicted mask via `model.predict`.
      - Displays the resized image with an overlaid predicted mask (using a jet colormap with 60% transparency)
        via matplotlib.

    Parameters:
      images : iterable
          Collection of image file paths.
      mask_dir : str
          Destination directory for saving the remounted full images and individual tile images (used when tiles is True).
      model : object
          The prediction model to be used for generating masks. It should have the attributes 'name' and either
          'input' or 'input_shape' to define the target size and for naming output files.
      tiles : bool, optional
          If True, perform tile-based processing and remounting; if False, perform a direct prediction on the resized image,
          displaying the result. Default is True.

    Returns:
      None
          The function either saves the processed images (when tiles is True) in `mask_dir` or displays the
          combined image and prediction overlay (when tiles is False).
    """
    if tiles:       
        os.makedirs(mask_dir, exist_ok=True)   
        for filename_path in images: 
            filename = os.path.basename(filename_path)
            output_path = f'{mask_dir}/remounted_{model.name}_{filename}'
            original_mask = None

            tiles, tiles_info = image_tiler.process_image(filename_path, mask=original_mask, target_size=model.input_shape[1:3])

            for i, (tile,_) in enumerate(tiles):
                output_image = os.path.join(mask_dir, filename[:-4] + f'_tile_{i}.jpg')
                tiles_info[3]["Tile Coordinates"][i]["ID"] = i
                predict_and_save(tile, model, output_image)
            json_path = tiles_info
            image_tiler.remount_masks(filename_path, json_path, original_mask, mask_dir, filename, output_path, plot_and_save=True)
    else:
        for filename_path in images:
            image = plt.imread(filename_path)
            image_resized = tf.image.resize(image, [model.input.shape[1], model.input.shape[2]])
            img_expanded = np.expand_dims(image_resized, axis=0)
            predicted_mask = model.predict(img_expanded, verbose=0)[0]
            plt.figure(figsize=(20, 20))
            plt.imshow(image_resized / 255.0)
            plt.imshow(predicted_mask, cmap="jet", alpha=0.6)
            plt.title("Resized Prediction Mask")
            plt.axis("off")
            plt.show()