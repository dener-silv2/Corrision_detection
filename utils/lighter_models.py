#models.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def decoder_block(x, skip, num_filters):
    """Ensures input tensors match spatial dimensions before concatenation."""
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:  
        skip = layers.Resizing(x.shape[1], x.shape[2])(skip)

    x = layers.concatenate([x, skip], axis=-1)
    x = layers.SeparableConv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    
    return x


def build_unet_resnet(input_shape, num_filters_list=[128, 64, 32], fine_tune_layers=0, verbose=True):
    """
    Builds a simplified U-Net model using ResNet50 as the backbone.
    """
    if verbose:
        print("Building Optimized U-Net with ResNet50 backbone...")

    # Input Layer
    inputs = layers.Input(input_shape)

    # ResNet50 Backbone (Feature Extractor)
    base_model = applications.ResNet50(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone initially

    # Fine-tune specific layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Extract skip connections
    skip_layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    skip_connections = [base_model.get_layer(layer).output for layer in skip_layers]
    skip_connections.reverse()

    # Bottleneck
    x = base_model.output
    x = layers.Conv2DTranspose(num_filters_list[0], (2, 2), strides=(2, 2), padding="same")(x)

    # Decoder
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    model = models.Model(inputs, outputs)
    return model


def build_unet_inception(input_shape, num_filters_list=[128, 64, 32], fine_tune_layers=10, verbose=True):
    """
    Builds a low-memory U-Net model using InceptionV3 as the backbone with the modular decoder function.
    """
    if verbose:
        print("Building simplified U-Net with InceptionV3 backbone...")

    inputs = layers.Input(shape=input_shape)

    # InceptionV3 Backbone
    base_model = applications.InceptionV3(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False

    # Fine-tune fewer layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Extract skip connections from earlier InceptionV3 layers
    skip_layers = ["mixed1", "mixed2", "mixed3"]  # Earlier layers reduce memory usage
    skip_connections = [base_model.get_layer(layer).output for layer in skip_layers]
    skip_connections.reverse()  # Reverse order for decoding

    # Bottleneck with SeparableConv2D
    x = base_model.output
    x = layers.SeparableConv2D(num_filters_list[0], (3, 3), activation="relu", padding="same")(x)

    # Decoder using `decoder_block`
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_efficientnet(input_shape, num_filters_list=[128, 64, 32], fine_tune_layers=10, verbose=True):
    """
    Builds a low-memory U-Net model using EfficientNetB0 as the backbone.
    """
    if verbose:
        print("Building simplified U-Net with EfficientNetB0 backbone...")

    inputs = layers.Input(shape=input_shape)

    # EfficientNetB0 Backbone (Feature Extractor)
    base_model = applications.EfficientNetB0(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone initially

    # Fine-tune fewer layers for efficiency
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Extract skip connections from earlier EfficientNetB0 layers
    skip_layers = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation"]
    skip_connections = [base_model.get_layer(layer).output for layer in skip_layers]
    skip_connections.reverse()  # Reverse order for decoding

    # Bottleneck with SeparableConv2D
    x = base_model.output
    x = layers.SeparableConv2D(num_filters_list[0], (3, 3), activation="relu", padding="same")(x)

    # Decoder using `decoder_block`
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_mobilenetv2(input_shape, num_filters_list=[128, 64, 32], fine_tune_layers=10, verbose=True):
    """
    Builds a low-memory U-Net model using MobileNetV2 as the backbone.
    """
    if verbose:
        print("Building simplified U-Net with MobileNetV2 backbone...")

    inputs = layers.Input(shape=input_shape)

    # MobileNetV2 Backbone (Feature Extractor)
    base_model = applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False  

    # Fine-tune fewer layers for efficiency
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Extract skip connections from earlier MobileNetV2 layers
    skip_layers = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    skip_connections = [base_model.get_layer(layer).output for layer in skip_layers]
    skip_connections.reverse()  

    # Bottleneck with SeparableConv2D
    x = base_model.output
    x = layers.SeparableConv2D(num_filters_list[0], (3, 3), activation="relu", padding="same")(x)

    # Decoder using `decoder_block`
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    # Compile the Model
    model = models.Model(inputs, outputs)
    return model


def build_unet_alexnet(input_shape, num_filters_list=[64, 32, 16], verbose=True):
    """
    Builds a fully modular U-Net model using an AlexNet-like encoder.
    """
    if verbose:
        print("Building Optimized U-Net with AlexNet-like backbone...")

    # Input Layer
    inputs = layers.Input(shape=input_shape)

    # Encoder: AlexNet-like structure using SeparableConv2D
    encoder_filters = [32, 64, 32]
    skip_connections = []

    x = inputs
    for filters in encoder_filters:
        x = layers.SeparableConv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)  # Downsampling
        x = layers.BatchNormalization()(x)
        skip_connections.append(x)

    skip_connections.reverse()  # Reverse for decoding

    # Bottleneck with SeparableConv2D
    x = layers.SeparableConv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Ensure num_filters_list matches skip connections count
    assert len(num_filters_list) == len(skip_connections), \
        f"num_filters_list must have {len(skip_connections)} elements, but got {len(num_filters_list)}."

    # Decoder using `decoder_block`
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    model = models.Model(inputs, outputs)

    return model


def build_mask_cnn(input_shape, num_filters_list=[64, 32, 16], verbose=True):
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
        print("Building simplified Mask-based CNN...")

    inputs = layers.Input(input_shape)

    # Encoder: Feature Extraction
    x = inputs
    for filters in num_filters_list:
        x = layers.SeparableConv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)  # Downsampling

    # Bottleneck
    x = layers.SeparableConv2D(num_filters_list[-1], kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Decoder: Reconstruct Masks
    for filters in reversed(num_filters_list):
            x = layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
            x = layers.SeparableConv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Output Layer
    outputs = layers.Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def build_deeplabv3plus(input_shape, backbone="resnet50", num_classes=1, verbose=True):
    """
    Builds a DeepLabV3+ segmentation model with a specified backbone, now using a modular decoder function.
    """
    if verbose:
        print(f"Building Optimized DeepLabV3+ with {backbone} backbone...")

    # Select Backbone
    if backbone == "resnet50":
        base_model = applications.ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        skip_layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    elif backbone == "mobilenetv2":
        base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        skip_layers = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False  # Freeze pretrained backbone

    # Extract skip connections
    skip_connections = [base_model.get_layer(layer).output for layer in skip_layers]
    skip_connections.reverse()  # Reverse order for decoding

    # Bottleneck with Atrous Convolutions
    x = base_model.output
    x = layers.Conv2D(256, 3, dilation_rate=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Decoder using `decoder_block`
    num_filters_list = [64, 32, 16] 
    for i, skip in enumerate(skip_connections):
        x = decoder_block(x, skip, num_filters_list[i])

    # Output Layer for Binary Segmentation
    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid", padding="same")(x)
    outputs = layers.Resizing(input_shape[0], input_shape[1])(outputs)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


MODEL_REGISTRY = {
    "build_unet_resnet": build_unet_resnet,
    "build_unet_inception": build_unet_inception,
    "build_unet_efficientnet": build_unet_efficientnet,
    "build_unet_alexnet": build_unet_alexnet,
    "build_unet_mobilenet": build_unet_mobilenetv2,
    "build_mask_cnn": build_mask_cnn,
    "build_deeplabv3plus": build_deeplabv3plus
}

def get_model(model_name, input_shape, verbose=True):
    """Retrieves a registered model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](input_shape, verbose=verbose)
