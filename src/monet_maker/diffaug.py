"""Differentiable image augmentations

Defines differentiable functions for augmenting images as introduced in
"Differentiable Augmentation for Data-Efficient GAN Training" by Shengyu
Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han available at
https://arxiv.org/pdf/2006.10738.

Attributes:
    AVAILABLE_AUGMENTATIONS (Tuple[str]): tuple of names for augmentation
        procedures fully defined and tested. Each item of the tuple will
        correspond to the name of a member function of the Augmentor class
        which implements the augmentation in a differentiable manner.

Todo:
    Redesign cutout augmentation to allow multiple cutout sizes per batch
     - would be easy with for loop... detrimental to GPU / TPU computation?
    Redesign translation augmentation to be more efficient
     - possible solution: register the gradient of the tensorflow roll function
    Separate augmentation functionality from class to remove need for
    instantiation.
"""

from typing import List, Tuple, Union, Iterable, Callable
import tensorflow as tf

AVAILABLE_AUGMENTATIONS = (
    'brightness',
    'color',
    'contrast',
    'saturation',
    'cutout',
    'translation',
)


class Augmentor(tf.keras.layers.Layer):
    """Encapsulates augmentation process

    Args:
        augmentations (Union[str, Iterable[str]]): List of augmentations to be
            implemented. Can either be provided as an iterable of augmentation
            names, or as a comma-separated-value string. Defaults to
            AVAILABLE_AUGMENTATIONS.
        max_brightness_adjustment (float): Maximum brightness adjustment value.
            Must be in range [0, 1]. Defaults to 0.5
        max_saturation_adjustment (float): Maximum saturation adjustment value.
            Must be in range [0, 1]. Defaults to 0.5
        max_contrast_adjustment (float): Maximum contrast adjustment value.
            Must be in range [0, 1]. Defaults to 0.5
        max_color_adjustment (float): Maximum color adjustment value. Must be
            in range [0, 1]. Defaults to 0.1
        max_translation_adjustment (float): Maximum translation adjustment
            value as fraction of image size. Must be in range [0, 1]. Defaults to 0.125
        max_cutout_size (float): Maximum size of cutout as fraction of image
            size. Must be in range [0, 1]. Defaults to 0.5
    """
    augmentations: List[Callable]

    def __init__(
            self,
            augmentations: Union[str, Iterable[str]] = AVAILABLE_AUGMENTATIONS,
            max_brightness_adjustment: float = 0.5,
            max_saturation_adjustment: float = 0.5,
            max_contrast_adjustment: float = 0.5,
            max_color_adjustment: float = 0.1,
            max_translation_adjustment: float = 0.125,
            max_cutout_size: float = 0.5,
            clip_values: bool = True,
    ):
        super().__init__()

        self.max_brightness_adjustment = max_brightness_adjustment
        self.max_saturation_adjustment = max_saturation_adjustment
        self.max_contrast_adjustment = max_contrast_adjustment
        self.max_color_adjustment = max_color_adjustment
        self.max_translation_adjustment = max_translation_adjustment
        self.max_cutout_size = max_cutout_size

        self.clip_values = clip_values

        if isinstance(augmentations, str):
            augmentations = map(str.strip, augmentations.split(','))

        self.augmentations = [
            getattr(self, augmentation)
            for augmentation in augmentations
            if hasattr(self, augmentation)
        ]

    def call(self, *batches: tf.Tensor) -> Tuple[tf.Tensor]:
        images = tf.concat(batches, 0)

        # randomize function order in tf.function-compatible manner
        for i in tf.random.shuffle(tf.range(len(self.augmentations))):
            for j in range(len(self.augmentations)):
                if i == j:
                    images = self.augmentations[j](images)

        # clipping destroys backprop, but prevents fitting to impossible data
        if self.clip_values:
            images = tf.clip_by_value(images, 0., 1.)

        return tf.split(images, len(batches))

    def brightness(self, images: tf.Tensor) -> tf.Tensor:
        # sample adjustment factors
        num_images = tf.shape(images)[0]
        adjustment = tf.random.uniform([num_images, 1, 1, 1], -1., 1.)
        adjustment *= self.max_brightness_adjustment

        # adjust mean pixel brightness
        images = images + adjustment
        return images

    def color(self, images: tf.Tensor) -> tf.Tensor:
        # sample adjustment factors
        num_images = tf.shape(images)[0]
        adjustment = tf.random.uniform([num_images, 1, 1, 3], -1., 1.)
        adjustment *= self.max_color_adjustment

        # adjust mean of each color
        images = images + adjustment
        return images

    def contrast(self, images: tf.Tensor) -> tf.Tensor:
        # sample adjustment factors
        num_images = tf.shape(images)[0]
        adjustment = tf.random.uniform([num_images, 1, 1, 1], -1., 1.)
        adjustment = 1 + self.max_contrast_adjustment * adjustment

        # adjust variance of pixel brightness
        brightness = tf.math.reduce_mean(images, axis=[-3, -2, -1], keepdims=True)
        images = (images - brightness) * adjustment + brightness
        return images

    def saturation(self, images: tf.Tensor) -> tf.Tensor:
        # sample adjustment factors
        num_images = tf.shape(images)[0]
        adjustment = tf.random.uniform([num_images, 1, 1, 1], -1., 1.)
        adjustment = 1 + self.max_saturation_adjustment * adjustment

        # adjust variance of each color
        avg_colors = tf.math.reduce_mean(images, axis=(1, 2), keepdims=True)
        images = (images - avg_colors) * adjustment + avg_colors
        return images

    def cutout(self, images: tf.Tensor) -> tf.Tensor:
        num_images = tf.shape(images)[0]
        image_size = tf.shape(images)[1:3]

        # sample random cutout locations
        centers = tf.random.uniform([num_images, 2]) * tf.cast(image_size, tf.float32)
        centers = tf.cast(centers, tf.int32)

        # sample random cutout size
        size = tf.random.uniform([2]) * tf.cast(image_size, tf.float32)
        size *= self.max_cutout_size

        # convert to grid indices
        indices = tf.ragged.range(tf.cast(-size / 2, tf.int32), tf.cast(size / 2 + 0.5, tf.int32))
        y_indices, x_indices = indices[0], indices[1]
        y_grid, x_grid = tf.meshgrid(y_indices, x_indices, indexing='ij')
        grid = tf.stack([y_grid, x_grid], axis=-1)

        # tile over image index
        indices = tf.tile(grid[None], [num_images, 1, 1, 1])

        # offset each image set by centers
        indices = indices + centers[:, None, None]
        indices %= image_size

        # prepend image index
        image_index = tf.reshape(tf.range(num_images), (-1, 1, 1, 1))
        bcast_shape = tf.concat([tf.shape(indices)[:-1], [1]], axis=0)
        image_index = tf.broadcast_to(image_index, bcast_shape)
        indices = tf.concat([image_index, indices], axis=-1)

        # reshape to be index list
        indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))

        # mask
        zeros = tf.zeros((tf.shape(indices)[0], tf.shape(images)[-1]))
        images = tf.tensor_scatter_nd_update(images, indices, zeros)

        return images

    def translation(self, images: tf.Tensor) -> tf.Tensor:
        num_images = tf.shape(images)[0]
        image_size = tf.shape(images)[1:3]

        # sample y and x translation
        adjustment = tf.random.uniform([num_images, 2], -1., 1.)
        adjustment *= self.max_translation_adjustment * tf.cast(image_size, tf.float32)
        adjustment = tf.cast(adjustment + 0.5 * tf.sign(adjustment), tf.int32)
        adjustment_y, adjustment_x = tf.split(adjustment, 2, axis=1)

        # calculate row and column indices of translated images
        rows, cols = tf.range(image_size[0])[None], tf.range(image_size[1])[None]
        rows, cols = tf.tile(rows, [num_images, 1]), tf.tile(cols, [num_images, 1])
        rows, cols = rows - adjustment_y, cols - adjustment_x

        # shift indices up by one and clip to separate valid and invalid values
        rows = tf.clip_by_value(rows + 1, 0, image_size[0] + 1)[..., None]
        cols = tf.clip_by_value(cols + 1, 0, image_size[1] + 1)[..., None]

        # add padding which will be selected by clipped values
        aug_images = tf.pad(images, [[0, 0], [1, 1], [1, 1], [0, 0]])

        # shift rows
        aug_images = tf.gather_nd(aug_images, rows, batch_dims=1)

        # shift columns
        aug_images = tf.transpose(aug_images, [0, 2, 1, 3])
        aug_images = tf.gather_nd(aug_images, cols, batch_dims=1)
        aug_images = tf.transpose(aug_images, [0, 2, 1, 3])

        # use ensure_shape to allow tf.function to trust the output tensor shape
        images = tf.ensure_shape(aug_images, images.shape)
        return images
