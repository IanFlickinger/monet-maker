from typing import Callable, Iterable, Union, List, Tuple, Collection, TypeVar

import tensorflow as tf

from . import diffaug

LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
T = TypeVar('T')


def _ensure_collection(obj: Union[T, Collection[T]], min_length: int = 1) -> Collection[T]:
    """Converts any non-collection into a collection with a minimum length.

    If obj is not a collection, will return a collection object which has length equal to
    min_length, and whose elements are all obj (not copies).

    If obj is a collection with length lesser than the min_length, will tile the elements of
    obj until the length is sufficient.

    If obj is a collection with length greater than or equal to the min_length, will simply
    return obj.

    Args:
        obj (Union[T, Collection[T]]): the object to be converted
        min_length (int): the minimum length of the returned collection. Defaults to 1.

    Returns:
        Collection[T]: The build collection wrapper
    """
    if not hasattr(obj, '__len__'):
        obj = [obj, ] * min_length
    elif len(obj) < min_length:
        obj = [*obj, ] * int((len(obj) / min_length) + 0.5)
    return obj


class OptimizerList(list):
    @property
    def lr(self):
        lrs = [o.lr.numpy() for o in self]
        return tf.constant(lrs)

    @lr.setter
    def lr(self, learning_rate: Union[float, Iterable[float]]):
        if type(learning_rate) is float:
            learning_rate = tf.fill(len(self), learning_rate)
        for opt, lr in zip(self, learning_rate):
            opt.lr.assign(lr)


class MultiHeadLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            loss_functions: Collection[LossFunction],
            weights: Iterable[float] = None,
            reduction: str = tf.keras.losses.Reduction.SUM,
            name: str = None,
    ):
        """

        Args:
            loss_functions (List[LossFunction]): list of loss functions, each with signature
                loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor.
            weights (tf.Tensor): weights for the given loss functions. Must be of same length
                as loss_functions
            reduction (str): see tf.keras.losses.Loss for more information.
            name (str): see tf.keras.losses.Loss for more information.
        """
        super().__init__(reduction, name)
        self.loss_functions = loss_functions
        self.weights = weights or tf.ones(len(loss_functions))
        if len(self.weights) != len(self.loss_functions):
            raise ValueError(f'There must be a one-to-one equivalence between weights and loss functions. '
                             f'MultiHeadLoss received {len(self.loss_functions)} loss functions and '
                             f'{len(self.weights)} weights.')

    def call(self, y_true: tf.Tensor, y_pred: Tuple[tf.Tensor]):
        """

        Args:
            y_true (tf.Tensor): ground truth values with shape = [batch_size, ...]
            y_pred (Tuple[tf.Tensor]): tuple of predicted values from each head with shapes = [batch_size, ...]

        Returns:
            Tuple[tf.Tensor]: scalar tensors containing loss values for each head.
        """
        losses = []
        for prediction, loss_function, weight in zip(y_pred, self.loss_functions, self.weights):
            losses.append(loss_function(y_true, prediction) * weight)
        return tuple(losses)


class CycleGan(tf.keras.Model):
    def __init__(
            self,
            m_gen,
            p_gen,
            m_dis,
            p_dis,
            aug=None
    ):
        super().__init__()
        self.m_gen = m_gen
        self.p_gen = p_gen
        self.m_dis = m_dis
        self.p_dis = p_dis
        if aug is None:
            aug = diffaug.Augmentor('')
        self.aug = aug

        self.optimizer = None
        self.m_id_loss = self.p_id_loss = None
        self.m_cycle_loss = self.p_cycle_loss = None
        self.m_dis_loss = self.p_dis_loss = None
        self.loss_weights = tf.ones(3)

    # noinspection PyMethodOverriding
    def compile(
            self,
            optimizer: Union[tf.keras.optimizers.Optimizer, OptimizerList],
            id_loss: Union[LossFunction, Collection[LossFunction]],
            cycle_loss: Union[LossFunction, Collection[LossFunction]],
            dis_loss: Union[LossFunction, Collection[LossFunction]],
            loss_weights=None,
    ):
        super().compile()

        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer = OptimizerList([optimizer, ] * 4)
        self.optimizer = optimizer

        self.m_id_loss, self.p_id_loss = _ensure_collection(id_loss, min_length=2)
        self.m_cycle_loss, self.p_cycle_loss = _ensure_collection(cycle_loss, min_length=2)
        self.m_dis_loss, self.p_dis_loss = _ensure_collection(dis_loss, min_length=2)

        if loss_weights is not None:
            loss_weights = tf.constant(loss_weights)
            if loss_weights.shape != (3,):
                raise ValueError(f'Loss weights must have shape=(3,), but compile received shape={loss_weights.shape}')
            self.loss_weights = loss_weights

    def train_step(self, data):
        # read tupled batch data = (monet_batch, photo_batch)
        m_real, p_real = data
        batch_size = tf.shape(m_real)[0]

        # Progress batch data through CycleGAN process
        with tf.GradientTape(persistent=True) as g_tape:
            # identity outputs
            m_id = self.m_gen(m_real, training=True)
            p_id = self.p_gen(p_real, training=True)

            # identity loss
            m_id_loss = self.m_id_loss(m_real, m_id)
            p_id_loss = self.p_id_loss(p_real, p_id)

            # transfer outputs
            m_fake = self.m_gen(p_real, training=True)
            p_fake = self.p_gen(m_real, training=True)

            # cycle outputs
            m_cycle = self.m_gen(p_fake, training=True)
            p_cycle = self.p_gen(m_fake, training=True)

            # cycle loss
            m_cycle_loss = self.m_cycle_loss(m_real, m_cycle)
            p_cycle_loss = self.p_cycle_loss(p_real, p_cycle)
            cycle_loss = m_cycle_loss + p_cycle_loss

            # differentiable augmentations
            m_real, m_fake = self.aug(m_real, m_fake)
            p_real, p_fake = self.aug(p_real, p_fake)

            # discriminator outputs
            m_dis_real = self.m_dis(m_real, training=True)
            m_dis_fake = self.m_dis(m_fake, training=True)
            p_dis_real = self.p_dis(p_real, training=True)
            p_dis_fake = self.p_dis(p_fake, training=True)

            # discriminator loss
            m_dis = tf.concat([m_dis_real, m_dis_fake], 0)
            p_dis = tf.concat([p_dis_real, p_dis_fake], 0)

            labels_real = tf.ones(batch_size)
            labels_fake = tf.zeros(batch_size)
            labels = tf.concat([labels_real, labels_fake], 0)

            m_dis_loss = self.m_dis_loss(labels, m_dis)
            p_dis_loss = self.p_dis_loss(labels, p_dis)

            # generator loss
            m_gen_loss = self.m_dis_loss(labels_real, m_dis_fake)
            p_gen_loss = self.p_dis_loss(labels_real, p_dis_fake)

            m_gen_loss = tf.reduce_sum(  # TODO: determine if this gets tricky in distribute scope
                tf.stack([m_gen_loss, m_id_loss, cycle_loss]) * self.loss_weights
            )
            p_gen_loss = tf.reduce_sum(
                tf.stack([p_gen_loss, p_id_loss, cycle_loss]) * self.loss_weights
            )

        # collect model losses and variables
        models = [self.m_gen, self.p_gen, self.m_dis, self.p_dis]
        losses = [m_gen_loss, p_gen_loss, m_dis_loss, p_dis_loss]
        variables = [model.trainable_variables for model in models]

        # apply backpropagation
        for model_loss, model_vars, opt in zip(losses, variables, self.optimizer):
            grads = g_tape.gradient(model_loss, model_vars)
            opt.apply_gradients(zip(grads, model_vars))

        # return losses and metrics
        return {
            'monet_id_loss': m_id_loss,
            'photo_id_loss': p_id_loss,
            'monet_cycle_loss': m_cycle_loss,
            'photo_cycle_loss': p_cycle_loss,
            'monet_discriminator_loss': m_dis_loss,
            'photo_discriminator_loss': p_dis_loss
        }

    # noinspection PyMethodOverriding
    def call(self, x, output_class='monet'):
        if output_class == 'monet':
            return self.m_gen(x)
        if output_class == 'photo':
            return self.p_gen(x)
