import numpy as np
import tensorflow as tf


def normalize(images):
    """
    Normalize images to [-1,1]
    """
    images = tf.cast(images, tf.float32)
    images /= 255.
    images -= 0.5
    images *= 2
    return images


def transform_train(images, labels):
    """
    Apply transformations to MNIST data for use in training.

    To images: random zoom and crop to 28x28, then normalize to [-1, 1]
    To labels: the class.
    """
    # zoom = 0.9 + np.random.random() * 0.2  # random between 0.9-1.1
    # size = int(round(zoom * 28))
    # images = tf.expand_dims(images, 3)  # [B, 28, 28, 1]
    # images = tf.image.resize_bilinear(images, (size, size))
    # images = tf.image.resize_image_with_crop_or_pad(images, 28, 28)
    images = normalize(images)
    images = images - tf.reduce_mean(images, axis=[1, 2], keepdims=True)
    return images, labels


def transform_val(images, labels):
    """
    Normalize MNIST images.
    """
    images = normalize(images)
    images = images - tf.reduce_mean(images, keepdims=True)
    return images, labels
