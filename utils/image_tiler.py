# image_tiler.py

import os
import math
import json
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

try:
    import rasterio
    import geopandas as gpd
    from owslib.wms import WebMapService
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "rasterio"])
    subprocess.run(["pip", "install", "geopandas"])
    subprocess.run(["pip", "install", "owslib"])
    import rasterio
    import geopandas as gpd
    from owslib.wms import WebMapService
from rasterio import features
from rasterio.transform import from_origin

def calculate_min_max_coordinates(image, tile_size):
    """
    Calculates coordinates for dividing a standard image into tiles.

    Parameters:
      image : numpy array of the image.
      tile_size : tuple (width, height) for each tile.

    Returns: 
      Dictionary with top-left coordinates and number of rows and columns.
    """
    height, width = image.shape[:2]
    n_row = math.ceil(height / tile_size[1])
    n_col = math.ceil(width / tile_size[0])

    top_left_corner = {
        "left_x": 0,
        "top_y": 0,
        "n_row": n_row,
        "n_col": n_col,
        "tile_size": tile_size
    }

    return top_left_corner

def cut_mask(image, mask, tile_coordinates):
    """
    Cuts a tile from the image and mask.

    Parameters:
      image : numpy array of the image.
      mask : numpy array of the mask  or None if missing.
      tile_coordinates : dictionary with 'left_x', 'top_y', and 'tile_size'.

    Returns:
      Tuple ( (image_tile, mask_tile), tile_coordinates ) if any image is present;
      otherwise, (None, None).
    """
    left_x = tile_coordinates["left_x"]
    top_y = tile_coordinates["top_y"]
    tile_size = tile_coordinates["tile_size"]

    image_tile = image[top_y:top_y + tile_size[1], left_x:left_x + tile_size[0]]
    if not mask == None:
        mask_tile = mask[top_y:top_y + tile_size[1], left_x:left_x + tile_size[0]]
    else:
        mask_tile = mask

    if np.any(image_tile):
        return (image_tile, mask_tile), tile_coordinates
    else:
        return None, None

def process_image(image, mask=None, target_size=(128, 128), num_processes=4):
    """
    Processes the image and mask to extract valid tiles.

    Parameters:
      image : file path or numpy array.
      mask : file path or numpy array.
      target_size : (width, height) for each tile.
      num_processes : number of processes for concurrent execution.

    Returns:
      Tuple (tiles, tile_info) where:
        - tiles is a list of (image_tile, mask_tile) pairs.
        - tile_info is meta-information including original image size, tile size,
          total number of tiles, and tile coordinates.
    """
    # Reshape masks to 2D if it's 1D
    if isinstance(image, str):
        image = plt.imread(image)
        mask = plt.imread(mask) if mask is not None else None
    if mask is not None and mask.ndim == 1:
        mask = mask.reshape(image.shape[0], image.shape[1]) 
    height, width = image.shape[:2]
    new_height = math.floor(height / target_size[1]) * target_size[1]
    new_width = math.floor(width / target_size[0]) * target_size[0]
    cropped_image = image[:new_height, :new_width]
    cropped_mask = mask[:new_height, :new_width] if mask is not None else None

    top_left_corner = calculate_min_max_coordinates(cropped_image, target_size)

    tiles = []
    tile_coordinates_list = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(top_left_corner["n_row"]):
            for j in range(top_left_corner["n_col"]):
                tile_coordinates = {
                    "left_x": j * target_size[0],
                    "top_y": i * target_size[1],
                    "tile_size": target_size
                }
                futures.append(executor.submit(cut_mask, cropped_image, cropped_mask, tile_coordinates))
                #print(futures+"\n")
        for future in futures:
            tile_data, tile_coord = future.result()
            if tile_data:
                tiles.append(tile_data)
                #print(tile_coord+"\n")
                tile_coordinates_list.append(tile_coord)

        # Create tile information list
    tile_info = [
        {"Original Image Size": image.shape},
        {"Tile Size": target_size[0]},
        {"Total Number of Tiles": len(tiles)},
        {"Tile Coordinates": tile_coordinates_list}  # Include coordinates list
    ]
    #print(tile_info+"\n")
    return tiles, tile_info


def remount_masks(filename_path, json_path, original_mask, mask_dir, filename, output_path, plot_and_save=False):
    """
    Remounts individual mask tiles to reconstruct the full image mask and overlays it
    on the original image.

    Parameters:
      filename_path : path to the original image.
      json_path : JSON file that contains tiling information.
      original_mask : file path to the original mask.
      mask_dir : directory where mask tiles are stored.
      filename : identifier for naming.
      output_path : path to save the remounted image.
      plot_and_save : whether to plot and save the overlay.

    """
    original_image = plt.imread(filename_path)
    # Load the JSON data
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Get the original image size
        original_size = data[0]["Original Image Size"]
    except:
        data = json_path
        original_size = data[0]["Original Image Size"]

    try:
        original_mask = plt.imread(original_mask)
    except:
        original_mask = None
    remounted_image = np.zeros(original_size[:2], dtype=np.float32)

    # Iterate over the tile coordinates and IDs
    for tile in data[3]["Tile Coordinates"]:
        tile_id = tile["ID"]
        left_x = tile["left_x"]
        top_y = tile["top_y"]
        tile_size = tile["tile_size"]

        # Load the mask tile
        mask_tile_path = os.path.join(mask_dir, f"{filename[:-4]}_tile_{tile_id}.npy")
        mask_tile = np.load(mask_tile_path)

        if mask_tile.shape[:2] != tuple(tile_size):
            mask_tile = np.resize(mask_tile, (tile_size[1], tile_size[0]))

        # Place the mask tile back into the original image
        remounted_image[top_y:top_y + tile_size[1], left_x: left_x + tile_size[0]] = mask_tile
        try:
            os.remove(mask_tile_path)
        except Exception as e:
            print(f"Warning: Could not delete {mask_tile_path} - {e}")


    if plot_and_save:
        fig, axes = plt.subplots(2, 1, figsize=(20, 20))

        fig.suptitle(filename+'\n', fontsize=14, fontweight='bold')

        # First subplot: Overlayed remounted mask
        axes[0].imshow(original_image)
        axes[0].imshow(remounted_image, alpha=0.8, cmap='jet')
        axes[0].set_title("Remounted Mask Overlay")
        axes[0].axis('off')

        error_occurred = False
        try:
            # Second subplot: Overlayed original mask
            axes[1].imshow(original_image)
            axes[1].imshow(original_mask, alpha=0.8)
            axes[1].set_title("Original Mask Overlay")
            axes[1].axis('off')
        except Exception as e:
            error_occurred = True
            fig.delaxes(axes[1]) 

        plt.tight_layout()
        plt.show()

        if error_occurred:
            print(f"No Original Ground Truth Mask to Show")

        # Save the remounted image
        extent = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())  # Get first subplot area
        fig.savefig(output_path, bbox_inches=extent, dpi=300)
        print(f"Saved remounted prediction at {output_path}")

# Aerial Image & Geospatial Functions

def convert_transform_dim(bounds, size):
    """
    Computes a raster transform for a georeferenced tile based on bounds and tile size.

    Parameters:
      bounds : dict with "min_lon_x", "min_lat_y", "max_lon_x", "max_lat_y".
      size : tuple (width, height) for the tile.

    Returns:
      A rasterio transform object.
    """
    pixel_conversion_width = (bounds["max_lon_x"] - bounds["min_lon_x"]) / size[0]
    pixel_conversion_height = (bounds["max_lat_y"] - bounds["min_lat_y"]) / size[1]
    transform = from_origin(bounds["min_lon_x"], bounds["max_lat_y"],
                            pixel_conversion_width, pixel_conversion_height)
    return transform

def connect_service(year):
    """
    Connects to the orthophoto WMS service for a given year.

    Parameters:
      year : a string indicating the year of the imagery.
    
    Returns:
      An instance of WebMapService.
    """
    orthophoto_url = f'https://cartografia.dgterritorio.gov.pt/wms/ortos{year}?service=wms&request=getcapabilities'
    return WebMapService(orthophoto_url, version="1.3.0")

def cut_building(village, bounds, tile_id):
    """
    Cuts a building mask from an aerial image tile.

    Parameters:
      village : dict containing georeferenced information and paths. It must have:
                - "buildins_path": directory to save masks.
                - "size": tile size (width, height).
                - "image_buildings": a GeoDataFrame of building geometries.
      bounds : dict with "min_lon_x", "min_lat_y", "max_lon_x", "max_lat_y" for the tile.
      tile_id : unique identifier for the tile.
    
    Returns:
      The file path of the saved building mask GeoTIFF, or None if no building is detected.
    """
    b_path = os.path.join(village["buildins_path"], f'{tile_id}_label.tif')
    if os.path.exists(b_path):
        os.remove(b_path)
    transform = convert_transform_dim(bounds, village["size"])
    with rasterio.open(b_path, 'w', driver='GTiff',
                       height=int(village["size"][1]),
                       width=int(village["size"][0]),
                       count=1,
                       dtype='uint8',
                       crs='EPSG:4326',
                       nodata=0,
                       transform=transform) as dst:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.ShapeSkipWarning)
            shapes = ((geom, 255) for geom in village["image_buildings"].geometry)
            printed = features.rasterize(
                shapes=shapes,
                out_shape=(village["size"][1], village["size"][0]),
                fill=0,
                transform=dst.transform,
                all_touched=True
            )
        if np.all(printed == 0):
            dst.close()
            os.remove(b_path)
            return None
        else:
            dst.write(printed, 1)
    return b_path

def calculate_min_max_coordinates_geo(village):
    """
    Calculates georeferenced tiling parameters based on a village's external bounds.

    Parameters:
      village : dict with keys "size" and "external_bounds".
                external_bounds should contain 'min_x', 'min_y', 'max_x', 'max_y'.

    Returns:
      A dictionary with geospatial tiling parameters.
    """
    earth_radius = 6371000  # in meters
    displacement_fig = village["size"][0] / (2 * earth_radius)
    max_displacement_y = math.radians(village["external_bounds"]["max_y"] - village["external_bounds"]["min_y"])
    max_displacement_x = math.radians(village["external_bounds"]["max_x"] - village["external_bounds"]["min_x"])
    lat_rad = math.radians(village["external_bounds"]["min_y"]) + max_displacement_y / 2
    lon_rad = math.radians(village["external_bounds"]["min_x"]) + max_displacement_x / 2
    n_row = math.ceil(abs(max_displacement_y / (displacement_fig * 2)))
    n_col = math.ceil(abs(max_displacement_x / (displacement_fig * 2 / math.cos(max_displacement_y))))
    left_x = lon_rad - (displacement_fig * n_col / math.cos(lat_rad))
    top_y = lat_rad + (displacement_fig * n_row)
    return {
        "left_x": left_x,
        "top_y": top_y,
        "n_row": n_row,
        "n_col": n_col,
        "from_middle": displacement_fig
    }

def download_and_save_image(village, image_type_color, coordinates):
    """
    Downloads an aerial image tile via a WMS service and saves it.

    Parameters:
      village : dict with keys "year", "srs", "size", and "buildins_path".
      image_type_color : string, 'nir' to request NIR ('IRG') images; otherwise, RGB.
      coordinates : dict with tile bounds and an identifier ("min_lon_x",
                    "min_lat_y", "max_lon_x", "max_lat_y", "n_key").

    Returns:
      The file path to the downloaded image tile.
    """
    if image_type_color == 'nir':
        i_type = 'IRG'
    else:
        i_type = 'RGB'
    wms = connect_service(village["year"])
    bbox = [coordinates["min_lon_x"], coordinates["min_lat_y"],
            coordinates["max_lon_x"], coordinates["max_lat_y"]]
    in_image = wms.getmap(
        layers=[f'Ortos{village["year"]}-{i_type}'],
        size=tuple(map(int, village["size"])),
        srs=village["srs"],
        bbox=bbox,
        format="image/tiff"
    )
    image_path = os.path.join(village["buildins_path"], f'{coordinates["n_key"]}_image_{image_type_color}.tif')
    if os.path.exists(image_path):
        os.remove(image_path)
    with open(image_path, 'wb') as out:
        out.write(in_image.read())
    return image_path

def process_aerial_images(village):
    """
    Processes aerial images by creating a grid of georeferenced tiles,
    extracting building masks, and displaying the resulting grid.

    Parameters:
      village : dict with necessary geospatial information,
                including "size", "external_bounds", "buildins_path", and
                "image_buildings" (GeoDataFrame of building geometries).

    Returns:
      A list of dictionaries; each contains a tile ID ("n_key") and its coordinates.
    """
    geo_coords = calculate_min_max_coordinates_geo(village)
    matrix_ortho = []
    iter_id = 0
    rows = geo_coords["n_row"]
    cols = geo_coords["n_col"]
    figsize = (cols, rows) if cols > rows else (rows, cols)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    top_y = geo_coords["top_y"]
    for i in range(rows):
        max_lat = math.degrees(top_y)
        x = geo_coords["left_x"]
        for j in range(cols):
            min_lon = math.degrees(x)
            min_lat = math.degrees(math.radians(max_lat) - geo_coords["from_middle"] * 2)
            max_lon = math.degrees(math.radians(min_lon) + (geo_coords["from_middle"] * 2 / math.cos(math.radians(max_lat))))
            tile_bounds = {
                "min_lon_x": min_lon,
                "min_lat_y": min_lat,
                "max_lon_x": max_lon,
                "max_lat_y": max_lat,
                "n_key": iter_id
            }
            x = math.radians(max_lon)
            b_path = cut_building(village, tile_bounds, iter_id)
            if b_path and os.path.exists(b_path):
                img_tile = plt.imread(b_path)
                matrix_ortho.append({"n_key": iter_id, "coordinates": tile_bounds})
            else:
                img_tile = np.zeros((village["size"][1], village["size"][0]))
            ax[i, j].imshow(img_tile, cmap='gray')
            ax[i, j].axis('off')
            iter_id += 1
        top_y = math.radians(min_lat)
    plt.tight_layout()
    plt.show()
    return matrix_ortho

# --- End of image_tiler.py module ---