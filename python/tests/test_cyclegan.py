import pytest

import tensorflow as tf

from monet_maker import cyclegan


class IdentityGenerator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return inputs


class ZeroGenerator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return tf.zeros_like(inputs)


class OneGenerator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return tf.ones_like(inputs)


class ZeroDiscriminator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros(batch_size)


class OneDiscriminator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        return tf.ones(batch_size)


@pytest.fixture()
def dual_discriminator():
    return lambda *args: (tf.constant([0.]), tf.constant([1.]))


@pytest.fixture()
def loss_log():
    return {}


@pytest.fixture()
def register_loss(loss_log):
    def add_loss(name, array):
        loss_log[name] = array

    return add_loss


@pytest.fixture()
def one_loss():
    return lambda *_, **__: tf.ones(1)


@pytest.fixture()
def zero_loss():
    return lambda *_, **__: tf.zeros(1)


@pytest.fixture()
def logged_loss(register_loss):
    def log_loss(loss, name):
        log = []

        def calc_loss(*args, **kwargs):
            value = loss(*args, **kwargs)
            log.append(value)
            return value

        register_loss(name, log)
        return calc_loss

    return log_loss


@pytest.fixture()
def optimizers():
    return cyclegan.OptimizerList([tf.keras.optimizers.SGD(), ] * 4)


@pytest.mark.parametrize('optimizer,num_optimizers,initial_learning_rate,new_learning_rate', [
    (
            tf.keras.optimizers.RMSprop,
            1, 1e-3, 1e-4
    ),
    (
            tf.keras.optimizers.Adam,
            2, 1e-4, 3e-5,
    ),
    (
            tf.keras.optimizers.Adagrad,
            4, 1e-4, tf.constant([1e-2, 1e-3, 1e-4, 1e-5]),
    ),
], scope='class')
class TestOptimizerList:
    @pytest.fixture(scope='class')
    def optimizer_list(self, optimizer, num_optimizers, initial_learning_rate, new_learning_rate):
        optimizers = [optimizer(initial_learning_rate) for _ in range(num_optimizers)]
        return cyclegan.OptimizerList(optimizers)

    def test_lr_get(self, optimizer_list, optimizer, num_optimizers, initial_learning_rate, new_learning_rate):
        assert tf.math.reduce_all(optimizer_list.lr == initial_learning_rate)

    def test_lr_set(self, optimizer_list, optimizer, num_optimizers, initial_learning_rate, new_learning_rate):
        optimizer_list.lr = new_learning_rate
        assert tf.math.reduce_all(optimizer_list.lr == new_learning_rate)

    def test_iter(self, optimizer_list, optimizer, num_optimizers, initial_learning_rate, new_learning_rate):
        i = -1
        for i, opt in enumerate(optimizer_list):
            assert isinstance(opt, tf.keras.optimizers.Optimizer)
        assert i + 1 == num_optimizers

    def test_len(self, optimizer_list, optimizer, num_optimizers, initial_learning_rate, new_learning_rate):
        assert len(optimizer_list) == num_optimizers


def test_cycle_gan_default_augmentor(monkeypatch):
    gan = cyclegan.CycleGan(
        m_gen=IdentityGenerator(),
        p_gen=IdentityGenerator(),
        m_dis=ZeroDiscriminator(),
        p_dis=ZeroDiscriminator(),
    )

    input_image = tf.random.uniform((4, 100, 100, 3))

    # set tf.random.uniform to return ones to ensure augmentor doesn't just sample
    # a zero adjustment which appears to mean no augmentation occurs
    monkeypatch.setattr(tf.random, 'uniform', lambda *args: tf.ones(args[0]))
    assert tf.reduce_all(gan.aug(input_image) == input_image)


@pytest.mark.parametrize('weights,expected_loss', [
    (None, 1.0),
    ([1.0, 0.0], 1.0),
    ([0.7, 0.3], 0.7)
])
def test_multi_head_loss(one_loss, zero_loss, weights, expected_loss):
    dual_loss = cyclegan.MultiHeadLoss([
        one_loss,
        zero_loss,
    ], weights=[0.7, 0.3])
    # dummy data - won't affect deterministic losses
    y_pred = (tf.zeros(10), tf.zeros(10))
    y_true = tf.ones(10)
    expected_loss = 0.7
    provided_loss = dual_loss(y_true, y_pred)

    assert provided_loss == expected_loss


class TestCycleGan:
    @pytest.fixture()
    def gan(self):
        gan = cyclegan.CycleGan(
            m_gen=IdentityGenerator(),
            p_gen=IdentityGenerator(),
            m_dis=ZeroDiscriminator(),
            p_dis=ZeroDiscriminator(),
        )

        return gan

    @pytest.fixture()
    def compiled_gan(self, gan, zero_loss, optimizers, logged_loss):
        # noinspection PyTypeChecker
        gan.compile(
            optimizer=optimizers,
            id_loss=logged_loss(zero_loss, 'id_loss'),
            cycle_loss=logged_loss(zero_loss, 'cycle_loss'),
            dis_loss=logged_loss(zero_loss, 'dis_loss'),
        )

    def test_train_step_losses_called(self, gan, optimizers, logged_loss, loss_log):
        # noinspection PyTypeChecker
        # compile gan with unique losses
        gan.compile(
            optimizer=optimizers,
            id_loss=[
                logged_loss(lambda *_, **__: tf.constant([1.]), 'm_id_loss'),
                logged_loss(lambda *_, **__: tf.constant([2.]), 'p_id_loss'),
            ],
            cycle_loss=[
                logged_loss(lambda *_, **__: tf.constant([3.]), 'm_cycle_loss'),
                logged_loss(lambda *_, **__: tf.constant([4.]), 'p_cycle_loss'),
            ],
            dis_loss=[
                logged_loss(lambda *_, **__: tf.constant([5.]), 'm_dis_loss'),
                logged_loss(lambda *_, **__: tf.constant([6.]), 'p_dis_loss'),
            ],
        )

        # train one step on dummy data
        shape = 5, 10, 10, 3
        data = (tf.ones(shape), tf.zeros(shape))
        gan.train_step(data)

        # id and cycle loss should be called once per class
        assert tf.reduce_sum(loss_log['m_id_loss']) == 1.
        assert tf.reduce_sum(loss_log['p_id_loss']) == 2.
        assert tf.reduce_sum(loss_log['m_cycle_loss']) == 3.
        assert tf.reduce_sum(loss_log['p_cycle_loss']) == 4.
        # dis loss is called twice: once for discriminator targets and once for generator targets
        assert tf.reduce_sum(loss_log['m_dis_loss']) == 5. * 2
        assert tf.reduce_sum(loss_log['p_dis_loss']) == 6. * 2


class TestMultiHeadDiscriminatorInCycleGan:
    @pytest.fixture()
    def gan(self, identity_generator, dual_discriminator, zero_discriminator, one_loss, zero_loss, optimizers):
        gan = cyclegan.CycleGan(
            m_gen=identity_generator,
            p_gen=identity_generator,
            m_dis=dual_discriminator,
            p_dis=zero_discriminator,
        )

        gan.compile(
            optimizer=optimizers,
            id_loss=zero_loss,
            cycle_loss=zero_loss,
            dis_loss=[
                cyclegan.MultiHeadLoss([
                    one_loss,
                    one_loss,
                ]),
                zero_loss,
            ]
        )

        return gan
