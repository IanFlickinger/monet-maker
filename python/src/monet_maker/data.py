"""Interface for using Kaggle datasets

This module contains the functionality for loading images contained in tfrec
files of a Kaggle dataset into tf.Tensor objects using strategy-aware logic.

Attributes:
    DEFAULT_RGB_MAX (int = 255): The maximum pixel value in an ordinary
        image with values [0-255].
    LOCAL_KAGGLE_DATASET_PATH (str = '../input/): The location of a kaggle
        dataset relative to the working directory of a kaggle kernel.
"""

import os
from typing import Iterable

import tensorflow as tf

import kaggle_datasets

from . import distribute

DEFAULT_RGB_MAX = 255
LOCAL_KAGGLE_DATASET_PATH = os.path.join('..', 'input')


class TfrecImageLoader:
    """Loads tfrec files of images in a Kaggle dataset.

    Loads tensorflow tfrec files containing images in a manner consistent with
    the current distribute strategy. Because of this, the loader must be
    instantiated within the strategy scope if the strategy is not supplied as
    an initializer argument.

    Args:
        dataset_name (str): The name of the Kaggle dataset
        data_dirs (Iterable[str]): The list of directories within the database to
            load. Should all be directories containing tfrec files. If not provided,
            the loader will only search the root directory of the dataset.
        strategy (Optional[tf.distribute.Strategy]): The distribute strategy in which
            the loader will operate. Must be provided if the loader is not
            instantiated within the strategy scope.

    Attributes:
        dataset_name (str): The name of the Kaggle dataset being referenced.
        data_path (str): The strategy-aware path to the Kaggle dataset (will be
            gcs path if using TPU strategy)
        data_dirs (Iterable[str]): The list of directories within the database being
            loaded.
    """
    dataset_name: str
    data_path: str
    data_dirs: Iterable[str]

    def __init__(self, dataset_name: str, data_dirs: Iterable[str] = None, strategy: tf.distribute.Strategy = None):
        self.dataset_name = dataset_name
        self.data_dirs = ['.'] if not data_dirs else data_dirs
        if not strategy:
            strategy = tf.distribute.get_strategy()
        local_path = os.path.join(LOCAL_KAGGLE_DATASET_PATH, dataset_name)
        self.data_path = get_path(dataset_name, local_path, strategy)

    def load(
            self,
            batch_size: int = 0,
            buffer_size: int = tf.data.AUTOTUNE,
            prefetch: int = tf.data.AUTOTUNE,
            seed=None
    ):
        """Load competition dataset.

        Args:
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
        data_dirs = (os.path.join(self.data_path, data_dir) for data_dir in self.data_dirs)

        # initialize empty list of loaded datasets
        datasets = []
        for data_dir in data_dirs:
            # create and configure dataset object
            ds = tf.data.TFRecordDataset(tf.io.gfile.glob(f'{data_dir}/*.tfrec'))
            ds = ds.map(tfrec_to_img)
            ds = ds.prefetch(prefetch).shuffle(buffer_size, seed=seed).repeat()
            if batch_size > 0:
                ds = ds.batch(batch_size)
            # append to running list
            datasets.append(ds)

        # zip all loaded datasets
        dataset = tf.data.Dataset.zip(tuple(datasets))

        return dataset

    def load_sample(
            self,
            sample_size: int = 1,
            buffer_size: int = tf.data.AUTOTUNE,
            seed: int = None
    ):
        """Load a sample from the competition dataset.

        Args:
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
        data_dirs = (os.path.join(self.data_path, data_dir) for data_dir in self.data_dirs)

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


def get_path(dataset_name: str, local_path: str, strategy=None):
    """Retrieves the Kaggle database path

    Args:
        dataset_name (str): the name of the kaggle dataset. Used to find the
            gcs path in the event a TPU strategy is in use.
        local_path (str): the path to the dataset on the local machine. Used
            when current strategy is not TPU-based.
        strategy (tf.distribute.Strategy): The distribution strategy for the
            calling compute environment. Must be supplied when
            function is not called within strategy.scope

    Returns:
        str: The path to the competition database.
    """
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    if distribute.is_tpu(strategy):
        return kaggle_datasets.KaggleDatasets().get_gcs_path(dataset_name)
    return local_path


def tfrec_to_img(tfrec, rgb_max: int = DEFAULT_RGB_MAX):
    """Translate a tf record containing a jpeg into the image tensor.

    Args:
        tfrec (TFRecord): The tensorflow record object to be parsed.
        rgb_max (int): The maximum pixel value in each channel. Defaults to
            DEFAULT_RGB_MAX.

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
