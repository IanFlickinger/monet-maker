import os
from typing import Iterable

import tensorflow as tf

import KaggleDatasets

from . import distribute
from . import data


# TODO: update docstrings

def load(
        data_path: str,
        data_dirs: Iterable[str],
        batch_size: int = 0,
        buffer_size: int = tf.data.AUTOTUNE,
        prefetch: int = tf.data.AUTOTUNE,
        seed=None
):
    """Load competition dataset.

    Args:
        data_path (str): The path to the database. If the strategy is using a TPU
            cluster for distribution, the database should be loaded from the gcs
            path. If the strategy is using local computation units (GPU or CPU),
            the database should be loaded from the local path.
        data_dirs (Iterable[str]): The list of directories within the database to
            load. Should all be directories containing tfrec files.
        batch_size (int): The number of instances that should be sampled at a
            time. Specify batch_size < 1 to disable batching in the dataset.
        buffer_size (int): The number of instances in a dataset to buffer when
            shuffling. See tf.data.Dataset.shuffle for more information.
        prefetch (int): The number of instances to cache ahead of being requested.
            See tf.data.Dataset.prefetch for more information.
        seed (int): The seed supplied to tf.data.Dataset.shuffle.

    Returns:
        tf.data.Dataset: A zipped dataset which includes tfrec data read from the
            supplied directories. Each iteration will yield a tuple of tf.Tensor
            objects. The first tuple item will be the batch yielded from the
            first supplied directory, the second item yielded from the second
            directory, and so on.
    """
    # append directories to database path
    data_dirs = (os.path.join(data_path, data_dir) for data_dir in data_dirs)

    # initialize empty list of loaded datasets
    datasets = []
    for data_dir in data_dirs:
        # create and configure dataset object
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(f'{data_dir}/*.tfrec'))
        ds = ds.map(data.tfrec_to_img)
        ds = ds.prefetch(prefetch).shuffle(buffer_size, seed=seed).repeat()
        if batch_size > 0:
            ds = ds.batch(batch_size)
        # append to running list
        datasets.append(ds)

    # zip all loaded datasets
    dataset = tf.data.Dataset.zip(tuple(datasets))

    return dataset


def load_sample(
        data_path: str,
        data_dirs: Iterable[str],
        sample_size: int = 1,
        buffer_size: int = tf.data.AUTOTUNE,
        seed: int = None
):
    """Load a sample from the competition dataset.

    Args:
        data_path (str): The path to the database. If the strategy is using a TPU
            cluster for distribution, the database should be loaded from the gcs
            path. If the strategy is using local computation units (GPU or CPU),
            the database should be loaded from the local path.
        data_dirs (Iterable[str]): The list of directories within the database to
            load. Should all be directories containing tfrec files.
        sample_size (int): The number of instances to sample from each directory.
        buffer_size (int): The number of instances in a dataset to buffer when
            shuffling. See tf.data.Dataset.shuffle for more information.
        seed (int): The seed supplied to tf.data.Dataset.shuffle.

    Returns:
        Tuple[tf.Tensor]: The samples from each dataset. The first tuple item
            will be the batch yielded from the first supplied directory, the
            second item yielded from the second directory, and so on.
    """
    # append directories to database path
    data_dirs = (os.path.join(data_path, data_dir) for data_dir in data_dirs)

    # initialize empty list of loaded subsets
    samples = []
    for data_dir in data_dirs:
        # create and configure dataset object
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(f'{data_dir}/*.tfrec'))
        ds = ds.map(tfrec_to_img)
        ds = ds.shuffle(buffer_size, seed=seed)
        ds = ds.batch(sample_size)
        # sample and append to running list
        s = next(iter(ds))
        samples.append(s)

    return tuple(samples)


def tfrec_to_img(tfrec, rgb_max: int = 255):
    """Translate a tf record containing a jpeg into the image tensor.

    Args:
        tfrec (TFRecord): The tensorflow record object to be parsed.

    Returns:
        tf.Tensor[tf.float32]: The image contained in the supplied TFRecord as a
            tensor of values in the range [0, 1] and with rank 3 shape indexed by
            (Height, Width, Channels).
    """
    encoded_image = tf.io.parse_single_example(tfrec, {
        'image': tf.io.FixedLenFeature([], tf.string)
    })['image']
    decoded_image = tf.io.decode_jpeg(encoded_image)
    return tf.cast(decoded_image, tf.float32) / rgb_max


def get_path(database_name: str, local_path: str, strategy=None):
    """Retrieves the database path

    Args:
        strategy (tf.distribute.Strategy): The distribution strategy for the
            calling compute environment. Only needs to be suppplied when
            function is not called within strategy.scope

    Returns:
        str: The path to the competition database.
    """
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    if distribute.is_tpu(strategy):
        return KaggleDatasets().get_gcs_path(database_name)
    return local_path
