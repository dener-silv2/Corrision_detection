#models.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def decoder_block(x, skip, num_filters):
    """A single decoder block with skip connections."""
    skip_resized = layers.Resizing(x.shape[1], x.shape[2])(skip)
    x = layers.concatenate([x, skip_resized], axis=-1)
    x = layers.Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    return x


def build_unet_resnet(input_shape, num_filters_list=[512, 256, 128, 64], fine_tune_layers=20, verbose=True):
    """
    Builds a U-Net model using ResNet50 as the backbone.

    Args:
        input_shape: Tuple, input shape of the image (height, width, channels).
        num_filters_list: List, filter sizes for each decoder stage.
        fine_tune_layers: Int, number of layers to unfreeze in the ResNet50 backbone for fine-tuning.
        verbose: Boolean, whether to print model building information.

    Returns:
        A compiled Keras Model.
    """
    if verbose:
        print("Building U-Net with ResNet50 backbone...")
    # Input Layer
    inputs = layers.Input(input_shape)

    # ResNet50 Backbone
    base_model = applications.ResNet50(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone initially

    # Optionally fine-tune specific layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Skip Connections
    skip_connections = [
        base_model.get_layer("conv2_block3_out").output,
        base_model.get_layer("conv3_block4_out").output,
        base_model.get_layer("conv4_block6_out").output,
        base_model.get_layer("conv5_block3_out").output,
    ]
    skip_connections.reverse()

    # Bottleneck
    x = base_model.output
    x = layers.Conv2D(512, 3, dilation_rate=2, activation='relu', padding='same')(x)  # Atrous convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(num_filters_list[0], (2, 2), strides=(2, 2), padding='same')(x)

    # Decoder
    for i, skip in enumerate(skip_connections):
        num_filters = num_filters_list[i]
        x = decoder_block(x, skip, num_filters)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_inception(input_shape, num_filters_list=[512, 256, 128, 64], fine_tune_layers=20, verbose=True):
    """
    Builds a U-Net model using InceptionV3 as the backbone.
    
    Args:
        input_shape: Tuple, input shape of the image (height, width, channels).
        num_filters_list: List, filter sizes for each decoder stage.
        fine_tune_layers: Int, number of layers to unfreeze in the InceptionV3 backbone for fine-tuning.
        verbose: Boolean, whether to print model building information.

    Returns:
        A compiled Keras Model.
    """
    if verbose:
        print("Building U-Net with InceptionV3 backbone...")

    # Input Layer
    inputs = layers.Input(input_shape)

    # InceptionV3 Backbone
    base_model = applications.InceptionV3(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone initially

    # Optionally fine-tune specific layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Skip Connections
    skip_connections = [
        base_model.get_layer("mixed2").output,
        base_model.get_layer("mixed3").output,
        base_model.get_layer("mixed4").output,
        #base_model.get_layer("mixed5").output,
    ]
    skip_connections.reverse()

    # Bottleneck
    x = base_model.output
    x = layers.Conv2D(512, 3, dilation_rate=2, activation='relu', padding='same')(x)  # Atrous convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(num_filters_list[0], (2, 2), strides=(2, 2), padding='same')(x)

    # Decoder
    for i, skip in enumerate(skip_connections):
        num_filters = num_filters_list[i]  # Get corresponding filters for each skip
        x = decoder_block(x, skip, num_filters)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_efficientnet(input_shape, num_filters_list=[512, 256, 128, 64], fine_tune_layers=20, verbose=True):
    """
    Builds a U-Net model using EfficientNetB0 as the backbone.

    Args:
        input_shape: Tuple, input shape of the image (height, width, channels).
        num_filters_list: List, filter sizes for each decoder stage.
        fine_tune_layers: Int, number of layers to unfreeze in the EfficientNetB0 backbone for fine-tuning.
        verbose: Boolean, whether to print model building information.

    Returns:
        A compiled Keras Model.
    """
    if verbose:
        print("Building U-Net with EfficientNetB0 backbone...")

    # Input Layer
    inputs = layers.Input(input_shape)

    # EfficientNetB0 Backbone
    base_model = applications.EfficientNetB0(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone initially

    # Optionally fine-tune specific layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Skip Connections
    skip_connections = [
        base_model.get_layer("block2a_expand_activation").output,  # Shallow layer
        base_model.get_layer("block3a_expand_activation").output,
        base_model.get_layer("block4a_expand_activation").output,
        base_model.get_layer("block6a_expand_activation").output,  # Deep layer
    ]
    skip_connections.reverse()  # Process from the deepest layer up

    # Bottleneck
    x = base_model.output
    x = layers.Conv2D(512, 3, dilation_rate=2, activation='relu', padding='same')(x)  # Atrous convolution for receptive field
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(num_filters_list[0], (2, 2), strides=(2, 2), padding='same')(x)

    # Decoder
    for i, skip in enumerate(skip_connections):
        num_filters = num_filters_list[i]  # Get corresponding filters for each skip
        x = decoder_block(x, skip, num_filters)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_mobilenetv2(input_shape, num_filters_list=[512, 256, 128, 64], fine_tune_layers=20, verbose=True):
    """
    Builds a U-Net model using MobileNetV2 as the backbone.
    
    Args:
        input_shape: Tuple, input shape of the image (height, width, channels).
        num_filters_list: List, filter sizes for each decoder stage.
        fine_tune_layers: Int, number of layers to unfreeze in the MobileNetV2 backbone for fine-tuning.
        verbose: Boolean, whether to print model building information.

    Returns:
        A compiled Keras Model.
    """
    if verbose:
        print("Building U-Net with MobileNetV2 backbone...")

    # Input Layer
    inputs = layers.Input(input_shape)

    # MobileNetV2 Backbone
    base_model = applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze entire backbone initially

    # Optionally fine-tune specific layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Skip Connections
    skip_connections = [
        base_model.get_layer("block_1_expand_relu").output,
        base_model.get_layer("block_3_expand_relu").output,
        base_model.get_layer("block_6_expand_relu").output,
        base_model.get_layer("block_13_expand_relu").output,
    ]
    skip_connections.reverse()  # Decoder processes from the deepest layer up

    # Bottleneck
    x = base_model.output
    x = layers.Conv2D(512, 3, dilation_rate=2, activation='relu', padding='same')(x)  # Atrous convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(num_filters_list[0], (2, 2), strides=(2, 2), padding='same')(x)

    # Decoder
    for i, skip in enumerate(skip_connections):
        num_filters = num_filters_list[i]  # Get corresponding filters for each skip
        x = decoder_block(x, skip, num_filters)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model

def build_unet_alexnet(input_shape, num_filters_list=[512, 256, 128, 64, 32], verbose=True):
    """
    Builds a U-Net model using a custom AlexNet-like encoder.

    Args:
        input_shape: Tuple, input shape of the image (height, width, channels).
        num_filters_list: List, filter sizes for each decoder stage.
        verbose: Boolean, whether to print model building information.

    Returns:
        A compiled Keras Model.
    """
    if verbose:
        print("Building U-Net with AlexNet-like backbone...")

    # Input Layer
    inputs = layers.Input(input_shape)
    x = inputs

    # AlexNet-Like Encoder
    encoder_filters = [96, 256, 384, 384, 256]  # Sequential filter sizes
    skip_connections = []

    for filters in encoder_filters:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)  # First convolution
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)  # Second convolution
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # Downsampling
        x = layers.BatchNormalization()(x)
        skip_connections.append(x)  # Store skip connections

    skip_connections.reverse()  # Reverse to process from deepest to shallowest layer

    # Bottleneck
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

        # Ensure num_filters_list matches the number of skip connections
    assert len(num_filters_list) == len(skip_connections), \
        f"num_filters_list must have {len(skip_connections)} elements, but got {len(num_filters_list)}."

    # Decoder
    for i, skip in enumerate(skip_connections):
        num_filters = num_filters_list[i]  # Corresponding filters for each skip
        x = decoder_block(x, skip, num_filters)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_mask_cnn(input_shape, num_filters_list=[64, 128, 256], verbose=True):
    """
    Builds a simple mask-based CNN for segmentation tasks.

    Args:
        input_shape: Tuple, input dimensions of the images (height, width, channels).
        num_filters_list: List, number of filters for each encoder stage.
        verbose: Boolean, whether to print model building details.

    Returns:
        A Keras Model for mask segmentation.
    """
    if verbose:
        print("Building Mask-based CNN...")

    inputs = layers.Input(input_shape)

    # Encoder: Feature Extraction
    x = inputs
    skip_connections = []
    for filters in num_filters_list:
        x = layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        skip_connections.append(x)  # Save skip connections for decoding
        x = layers.MaxPooling2D((2, 2))(x)  # Downsampling

    # Bottleneck
    x = layers.Conv2D(num_filters_list[-1], kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(num_filters_list[-1], kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Decoder: Reconstruct Masks
    skip_connections.reverse()  # Process skip connections from deepest to shallowest layer
    for i, skip in enumerate(skip_connections):
        x = layers.Conv2DTranspose(num_filters_list[-(i + 1)], kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv2D(num_filters_list[-(i + 1)], kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(num_filters_list[-(i + 1)], kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Output Layer
    outputs = layers.Conv2D(1, kernel_size=(1, 1), activation="sigmoid", dtype=tf.float32)(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_deeplabv3plus(input_shape, backbone="resnet50", num_classes=1, verbose=True):
    """
    Builds a DeepLabV3+ segmentation model with a specified backbone.

    Args:
        input_shape: Tuple, input image shape (height, width, channels).
        backbone: String, backbone architecture ('resnet50' or 'mobilenetv2').
        num_classes: Int, number of output classes for segmentation.
        verbose: Boolean, whether to print model details.

    Returns:
        A Keras Model for DeepLabV3+.
    """
    if verbose:
        print(f"Building DeepLabV3+ with {backbone} backbone...")

    # Select Backbone
    if backbone == "resnet50":
        base_model = applications.ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    elif backbone == "mobilenetv2":
        base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False  # Freeze pretrained backbone

    # DeepLabV3+ Atrous Convolutions and Decoder
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Conv2D(256, 3, dilation_rate=2, activation="relu", padding="same")(x)  # Atrous convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, dilation_rate=2, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=(4, 4))(x)  # Decoder upsampling
    x = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)  # Output mask
    x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model

def fusion_block(high, low, num_filters):
    """
    Upsamples the low-resolution branch and fuses it with the high-resolution branch.
    
    Args:
        high: High-resolution feature map.
        low: Low-resolution feature map.
        num_filters: Number of filters for the 1x1 projection and the following conv.
        
    Returns:
        fused: The fused feature map.
    """
    # Upsample the lower resolution branch to the high branch resolution.
    low_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(low)
    low_up = layers.Conv2D(num_filters, (1, 1), activation='relu', padding='same')(low_up)
    
    # Concatenate the high-res branch with the upsampled low-res branch.
    fused = layers.concatenate([high, low_up], axis=-1)
    
    # Process the concatenated features.
    fused = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', dtype=tf.float32)(fused)
    fused = layers.BatchNormalization()(fused)
    return fused

def build_hrnet_alexnet(input_shape, num_classes=1, verbose=True):
    """
    Constructs an HRNet-based segmentation model with an AlexNet-like stem.
    
    The model consists of:
      - A stem block for initial feature extraction.
      - Two parallel branches:
          • Branch A: High-resolution stream (remains at half resolution).
          • Branch B: Low-resolution stream (further downsampled).
      - A fusion block that upsamples and fuses branch B into branch A.
      - A decoder that upsamples fused features back to the original resolution.
    
    Args:
        input_shape: Tuple, the input shape (height, width, channels).
        num_classes: Number of output classes (1 for binary segmentation).
        summary: Whether to print the Keras model summary.
    
    Returns:
        A Keras Model.
    """
    if verbose:
        print(f"Building HRNet-based AlexNet-like backbone...")
    inputs = layers.Input(input_shape)
    
    # ----- AlexNet-Inspired Stem Block -----
    # Using a large kernel (e.g., 11x11) and strides similar to AlexNet.
    x = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # After the stem, resolution is roughly reduced (e.g., H/2 x W/2).
    
    # ----- Parallel Branches for Multi-Resolution Processing -----
    # Branch A: Maintains the current resolution.
    branchA = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    branchA = layers.BatchNormalization()(branchA)
    
    # Branch B: Downsample further (e.g., to H/4 x W/4) to capture more global context.
    branchB = layers.MaxPooling2D((2, 2))(x)
    branchB = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(branchB)
    branchB = layers.BatchNormalization()(branchB)
    
    # ----- Fusion Block -----
    # Fuse branch B (after upsampling) with branch A.
    fused = fusion_block(branchA, branchB, num_filters=128)
    
    # ----- Decoder / Final Upsampling -----
    # Upsample fused features back to the original image resolution.
    up1 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(fused)
    up1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(up1)
    up1 = layers.BatchNormalization()(up1)
    
    # Final 1x1 convolution for segmentation mask.
    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid" if num_classes == 1 else "softmax", padding="same", dtype=tf.float32)(up1)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)
    
    model = models.Model(inputs, outputs)
    
    return model


MODEL_REGISTRY = {
    "build_unet_resnet": build_unet_resnet,
    "build_unet_inception": build_unet_inception,
    "build_unet_efficientnet": build_unet_efficientnet,
    "build_unet_alexnet": build_unet_alexnet,
    "build_unet_mobilenet": build_unet_mobilenetv2,
    "build_mask_cnn": build_mask_cnn,
    "build_deeplabv3plus": build_deeplabv3plus,
    "build_hrnet_alexnet": build_hrnet_alexnet
}

def get_model(model_name, input_shape, verbose=True):
    """Retrieves a registered model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](input_shape, verbose=verbose)