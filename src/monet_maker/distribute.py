from typing import Callable, Optional

import tensorflow as tf


def build_strategy():
    """Creates a tf.distribute.Strategy object.

    Assesses the calling compute environment, and creates a Strategy object appropriate
    given the available accelerators. Preference will be granted to TPU accelerators,
    followed by GPU accelerators, before falling back on CPU computation.

    Returns:
        A tf.distribute.Strategy object for the calling compute environment.
        This will be of one of the following types
         - TPUStrategy (TPU Accelerator)
         - MirroredStrategy (GPU Accelerator)
         - _DefaultDistributionStrategy (No Accelerator)
    """
    # prefer TPU if available
    try:
        # resolver will throw ValueError over the lack of a tpu address if tpu not found
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        # connect to tpu system
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except ValueError:
        # resolver will throw ValueError over the lack of a tpu address if tpu not found
        pass

    # connect to GPU if TPU unavailable
    if tf.config.list_physical_devices('GPU'):
        return tf.distribute.MirroredStrategy()

    # fall back on CPU
    # TODO: evaluate MirroredStrategy for cpu
    return tf.distribute.get_strategy()


def is_tpu(strategy):
    """Evaluates whether the strategy uses a TPU cluster.

    Useful for storage interaction, as the TPU cluster will be in a cloud
    environment and will not default to local memory.

    Args:
        strategy (tf.distribute.Strategy): the strategy to be evaluated

    Returns:
        bool: True if the supplied strategy operates on TPU accelerator(s)"""
    return isinstance(strategy, tf.distribute.TPUStrategy)


# TODO: evaluate purpose of this function from a package standpoint
def loss(
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        strategy: Optional[tf.distribute.Strategy] = None
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Wraps a loss function with a strategy-aware reduction.

    Args:
        loss_fn (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): The loss
            function which takes (y_true, y_pred) as arguments and returns
            the loss. Should be compatible with tensorflow's tf.function api
            and does not implement any reduction over the batch. For example,
            a tf.keras.losses.Loss object must be built with the reduction
            argument set to tf.keras.losses.Reduction.NONE.
        strategy (tf.distribute.Strategy): The distribution strategy for the
            calling compute environment. Only needs to be suppplied when
            function is not called within strategy.scope

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the loss function wrapped
            with a strategy-aware reduction. Just like the input loss function,
            will take (y_true, y_pred) as arguments and return the loss as a
            Tensor scalar.
    """
    # capture current strategy
    if strategy is None:
        strategy = tf.distribute.get_strategy()

    # calculate reduction parameters
    global_batch_size = strategy.num_replicas_in_sync * DataConfig.BATCH_SIZE

    # build reduction wrapper
    @tf.function
    def reduced_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # flatten instances
        flat_shape = (-1, tf.math.reduce_prod(y_true.shape[1:]))
        y_true = tf.reshape(y_true, flat_shape)
        y_pred = tf.reshape(y_pred, flat_shape)

        # calculate reduced loss
        loss_by_instance = loss_fn(y_true, y_pred)
        return tf.reduce_sum(loss_by_instance) / global_batch_size

    # noinspection PyTypeChecker
    # tf.function wrapper returns a tf.types.experimental.GenericFunction
    return reduced_loss
