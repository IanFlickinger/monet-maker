import IPython
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

CELL_WIDTH = 16.0


def to_image_grid(images):
    """Transforms tensor of stacked image volumes into a single volume of all images
    concatenated together.

    This function is not meant to be called directly, as it does not support any variety
    of tensor shapes. It is called by image_grid, which handles more data validation.

    Args:
        images (tf.Tensor): tensor of stacked image volumes with rank 5 and indexed by
            (row, column, height, width, channel).

    Returns:
        tf.Tensor: tensor of shape (rows*height, cols*width, channel) with all images
            concatenated into a single volume representing a grid of the images
    """
    # validate input
    if len(images.shape) != 5:
        raise ValueError(f'Invalid tensor rank. Expected rank 5, got {len(images.shape)}')

    # transform tensor into shape (row*height, col*width, channel)
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    nshape = (images.shape[0] * images.shape[1], images.shape[2] * images.shape[3], -1)
    images = tf.reshape(images, nshape)

    # return concatenated images
    return images


def image_grid(images):
    """Transforms Tensor of one or more image volumes into a single volume of all images
    concatenated together.

    This function takes an input Tensor of one of the following shapes:
        (height, width) - 1 image
        (height, width, channel) - 1 image
        (col, height, width, channel) - col images
        (row, col, height, width, channel) - row*col images
    The input Tensor is then transformed into a single image volume with all input images
    concatenated together as a grid. The output image tensor of rank 3 is indexed by
        (height, width, channel)

    Args:
        images (tf.Tensor): tensor of one or more stacked image volumes. This tensor must
            be indexed by one of the following shapes:
             - (height, width): 1 image
             - (height, width, channel): 1 image
             - (col, height, width, channel): 1 row of col images
             - (row, col, height, width, channel): grid of row*col images

    Returns:
        tf.Tensor: tensor of shape (rows*height, cols*width, channel) with all images
            concatenated into a single volume representing a grid of the images
    """
    # ensure 3d image tensor
    irank = images.shape
    if irank == 5:
        # tensor shape: ()
        # concatenate images
        images = to_image_grid(images)
    elif irank == 4:
        # assume horizontal and add empty rows axis before concatenating
        images = images[None]
        images = to_image_grid(images)
    elif irank == 2:
        # add channels axis
        images = images[..., None]
    elif irank != 3:
        raise ValueError(f'Images tensor has improper rank {irank}')

    return images


def to_pil(img):
    """Transforms input image tensor into a PIL Image object.

    Args:
        img (Union[tf.Tensor, np.ndarray]): A single image volume of float
            values in the range [0, 1].

    Returns:
        PIL.Image: The same image as a PIL.Image object
    """
    img = tf.cast(img * tf.constant([255.]), tf.uint8)
    img = img.numpy()
    img = Image.fromarray(img)
    return img


def image_gallery(
        images,
        img_titles=None,
        row_titles=None,
        col_titles=None,
        channels=True,
        cell_width=16.0
):
    images = np.array(images)
    nrows, ncols = images.shape[:2]
    images = images.reshape(-1, *images.shape[2:])

    image_size = cell_width / ncols
    _, axes = plt.subplots(figsize=(image_size * ncols, image_size * nrows),
                           nrows=nrows, ncols=ncols)
    axes = np.array(axes).ravel()

    # initialize title arrays to Nones
    titles = np.full_like(axes, None)
    xtitles = np.full_like(axes, None)
    ytitles = np.full_like(axes, None)

    # evaluate titles
    if col_titles is not None:
        titles[:len(col_titles)] = col_titles
        if img_titles is not None:
            xtitles[:] = img_titles
    elif img_titles is not None:
        titles[:] = img_titles
    if row_titles is not None:
        ytitles.reshape(nrows, ncols)[:, 0] = row_titles

    # draw images
    for img, ax, t, xt, yt in zip(images, axes, titles, xtitles, ytitles):
        draw_image(img, ax)
        ax.set_title(t)
        ax.set_xlabel(xt)
        ax.set_ylabel(yt)


def draw_image(img, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines:
        ax.spines[s].set_visible(False)

    ax.imshow(img)
    return ax


def generator_evolution(
        images,
        filepath=None,
        classes=None,
        epochs=None,
        add_alpha=True,
        separate_classes=True,
        vertical=False,
        display=True,
):
    """Visualizes the training evolution of an image generator.

    TODO: Further description necessary

    Args:
        images (tf.Tensor): rank 6 tensor of image volumes indexed by
            (class, image, epoch, height, width, channel)
        filepath (Optional[str]): filepath to save visual to.
        classes (Optional[Iterable[str]]): list of classes corresponding
            to the first index of the images tensor for annotation
        epochs (Union[None, Iterable[int], Iterable[str]]): list of epochs
            corresponding to the third index of the images tensor for annotation
        separate_classes (bool): If True, a gap will be left between the classes
        vertical (bool): If True, timeline will extend downward along the visual
            while classes and images are spread along horizontal axis.

    Returns:
        tf.Tensor: a single image volume containing the generator evolution
            visual.
    """
    # extract image characteristics
    image_height = images.shape[3]
    # transform each class into an image grid
    grids = [to_image_grid(img_cls) for img_cls in images]
    # add transparency channel if requested
    if add_alpha:
        grids = [tf.concat([grid, tf.fill((*grid.shape[:-1], 1), 1.)], axis=-1) for grid in grids]
    # connect all images with gaps between classes
    final_image = grids[0]
    for grid in grids[1:]:
        final_image = tf.concat([
            final_image,
            tf.fill(
                (image_height, *grid.shape[1:]), 1.
            ),
            grid
        ], axis=0)

    if filepath:
        save_image(final_image, filepath)
    if display:
        display_image(final_image)
    return final_image


def display_image(img):
    IPython.display.display(to_pil(img))


def save_image(img, filepath):
    to_pil(img).save(filepath)
