# loss_and_metrics.py

import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculates the Dice coefficient, a measure of overlap between the true and predicted masks.

    Args:
        y_true (tf.Tensor): Ground truth tensor with binary values (0 or 1).
        y_pred (tf.Tensor): Predicted tensor with values between 0 and 1.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1.

    Returns:
        tf.Tensor: Dice coefficient, with values between 0 (no overlap) and 1 (perfect overlap).
    """
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.reduce_sum(tf.square(y_true), -1) + tf.reduce_sum(tf.square(y_pred), -1) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    """
    Calculates the Dice loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor with binary values (0 or 1).
        y_pred (tf.Tensor): Predicted tensor with values between 0 and 1.

    Returns:
        tf.Tensor: Dice loss value.
    """
    return 1 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def weighted_dice_loss(y_true, y_pred, weight_background=1.0, weight_foreground=1.0):
    """
    Calculates a weighted Dice loss, accounting for imbalances between foreground and background.

    Args:
        y_true (tf.Tensor): Ground truth tensor with binary values (0 or 1).
        y_pred (tf.Tensor): Predicted tensor with values between 0 and 1.
        weight_background (float): Weight for background pixels. Default is 1.0.
        weight_foreground (float): Weight for foreground pixels. Default is 1.0.

    Returns:
        tf.Tensor: Weighted Dice loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    weights = y_true * weight_foreground + (1 - y_true) * weight_background
    dice = (2. * intersection + 1e-7) / (tf.reduce_sum(weights * (y_true + y_pred)) + 1e-7)
    return 1 - dice

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred, from_logits=False):
    """
    Combines Binary Cross-Entropy (BCE) and Dice Loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor with binary values (0 or 1).
        y_pred (tf.Tensor): Predicted tensor with values between 0 and 1 (or logits if `from_logits=True`).
        from_logits (bool): Whether `y_pred` is a tensor of logits. Default is False.

    Returns:
        tf.Tensor: Combined loss value (BCE + Dice Loss).
    """
    # Binary Cross-Entropy Loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)(y_true, y_pred)

    # Dice Loss
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = 1 - numerator / tf.maximum(denominator, tf.keras.backend.epsilon())
    dice_loss = tf.reduce_mean(dice)

    # Combine BCE and Dice Loss
    return bce + dice_loss

@tf.keras.utils.register_keras_serializable()
def soft_f1_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    return 1 - (2 * precision * recall) / (precision + recall + 1e-7)


# F1 score for imbalanced datasets, combines Precision with Recall
@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.3, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        #y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        #y_true = tf.cast(y_true, tf.float32)
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
