# mask_vegetation
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Example vegetation index computations.
# Here, the image is assumed to be a multi-band array (H x W x 4) with:
# - Channel 0: Red
# - Channel 1: Green
# - Channel 2: Blue
# - Channel 3: Near Infrared (NIR)

def compute_ndvi(image):
    red = image[:, :, 0].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def compute_gndvi(image):
    green = image[:, :, 1].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    gndvi = (nir - green) / (nir + green + 1e-6)
    return gndvi

def compute_savi(image, L=0.5):
    red = image[:, :, 0].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return savi

# Example of additional index: VARI (Visible Atmospherically Resistant Index)
def compute_vari(image):
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)
    numerator = green - red
    denominator = green + red - blue + 1e-6
    vari = numerator / denominator
    return vari

# An adaptive thresholding function using Otsu's method.
def adaptive_threshold(index_map):
    scaled = 255 * (index_map - np.min(index_map)) / (np.ptp(index_map) + 1e-6)
    scaled = scaled.astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary / 255.).astype(np.uint8)

def ensemble_weak_mask(image, 
                       fixed_thresholds={'ndvi': 0.1, 'gndvi': 0.1, 'savi': 0.1, 'vari': 0.05},
                       weights={'ndvi': 0.25, 'gndvi': 0.25, 'savi': 0.25, 'vari': 0.25}, 
                       use_adaptive=False):
    """
    Computes weak vegetation masks using multiple indices and combines them.

    Parameters:
        image (np.array): Multi-band image with shape (H, W, 4) or more if additional indices are available.
        fixed_thresholds (dict): Fixed thresholds for each index (if adaptive thresholding is off).
        weights (dict): Weights for merging the corresponding binary masks.
        use_adaptive (bool): If True, use adaptive thresholding per index.
        
    Returns:
        ensemble_mask (np.array): Final binary weak mask for vegetation.
        individual_masks (dict): Individual binary masks for each index computed.
    """
    # Compute indices
    ndvi = compute_ndvi(np.array(image))
    gndvi = compute_gndvi(np.array(image))
    savi = compute_savi(np.array(image))
    vari = compute_vari(np.array(image))
    
    # Determine binary masks from each index
    if use_adaptive:
        ndvi_mask = adaptive_threshold(ndvi)
        gndvi_mask = adaptive_threshold(gndvi)
        savi_mask = adaptive_threshold(savi)
        vari_mask = adaptive_threshold(vari)
    else:
        ndvi_mask = (ndvi > fixed_thresholds['ndvi']).astype(np.uint8)
        gndvi_mask = (gndvi > fixed_thresholds['gndvi']).astype(np.uint8)
        savi_mask = (savi > fixed_thresholds['savi']).astype(np.uint8)
        vari_mask = (vari > fixed_thresholds['vari']).astype(np.uint8)
    
    # Combine using a weighted sum. You might use a threshold of 0.5 or adjust based on experiments.
    ensemble_score = (weights['ndvi'] * ndvi_mask +
                      weights['gndvi'] * gndvi_mask +
                      weights['savi'] * savi_mask +
                      weights['vari'] * vari_mask)
    ensemble_mask = (ensemble_score > 0.5).astype(np.uint8)
    
    individual_masks = {'ndvi': ndvi_mask, 
                        'gndvi': gndvi_mask, 
                        'savi': savi_mask,
                        'vari': vari_mask}
    
    return ensemble_mask, individual_masks

def crop_center(image, crop_size=(512, 512)):
    """ Crop the center of an image to the given size (H x W). """
    height, width = image.shape[:2]
    
    # Calculate center crop coordinates
    start_x = (width - crop_size[1]) // 2
    start_y = (height - crop_size[0]) // 2
    
    return image[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]
# Example usage:
# Load RGB image
rgb_image = plt.imread("/mnt/c/Users/hrodrigues/Downloads/image_rgb_1.tif", cv2.IMREAD_COLOR)  # Reads as (H, W, 3)
rgb_image = crop_center(rgb_image)
# Load NIR R G image
nir_rg_image = plt.imread("/mnt/c/Users/hrodrigues/Downloads/image_nir_1.tif", cv2.IMREAD_COLOR)  # Reads as (H, W, 3)
nir_rg_image = crop_center(nir_rg_image)
# Ensure both images have the same shape
height, width = rgb_image.shape[:2]

# Extract the NIR band from the NIR R G image (first channel)
nir_channel = nir_rg_image[:, :, 0]  # Assuming NIR is stored in the Red slot of NIR R G

# Expand dimensions so it matches the RGB image structure
nir_channel = nir_channel.reshape(height, width, 1)  # Convert to (H, W, 1)

# Merge RGB and NIR into a single 4-band image
image = np.concatenate((nir_channel, rgb_image), axis=-1)  # Shape (H, W, 4)
plt.imshow(image)
plt.axis('off')

# Compute an ensemble weak mask using fixed thresholds
ensemble_mask, masks = ensemble_weak_mask(image, use_adaptive=False)

# Optionally, try adaptive thresholding for each index
ensemble_mask_adapt, masks = ensemble_weak_mask(image, use_adaptive=True)

# Plot and compare the masks
alpha = 0.7
plt.figure(figsize=(24, 8))
plt.subplot(1, 6, 1)
plt.imshow(image[:, :, 1:4])
plt.imshow(ensemble_mask_adapt, cmap='gray', alpha=alpha)
plt.title("Ensemble Mask (adapt)")
plt.axis('off')

plt.subplot(1, 6, 2)
plt.imshow(image[:, :, 1:4])
plt.imshow(ensemble_mask, cmap='gray', alpha=alpha)
plt.title("Ensemble Mask (Fixed)")
plt.axis('off')

plt.subplot(1, 6, 3)
plt.imshow(image[:, :, 1:4])
plt.imshow(masks['ndvi'], cmap='gray', alpha=alpha)
plt.title("NDVI Mask")
plt.axis('off')

plt.subplot(1, 6, 4)
plt.imshow(image[:, :, 1:4])
plt.imshow(masks['gndvi'], cmap='gray', alpha=alpha)
plt.title("GNDVI Mask")
plt.axis('off')

plt.subplot(1, 6, 5)
plt.imshow(image[:, :, 1:4])
plt.imshow(masks['savi'], cmap='gray', alpha=alpha)
plt.title("SAVI Mask")
plt.axis('off')

plt.subplot(1, 6, 6)
plt.imshow(image[:, :, 1:4])
plt.imshow(masks['vari'], cmap='gray', alpha=alpha)
plt.title("VARI Mask")
plt.axis('off')

plt.show()
