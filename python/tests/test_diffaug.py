import pytest
import tensorflow as tf
from monet_maker import diffaug


# TODO
#   - More complex test examples
#       - Batches
#       - Variance among and within color channels
#       - Post-augmentation clipping
#       - Cutout wrapping
#       - Zero-sized cutout
#   - Read test examples in from other file or module
#   - Define abstract of the diffaug test process for adding new operations


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # test no adjustment
            tf.ones((1, 10, 10, 3)),
            tf.constant([0.]),
            tf.ones((1, 10, 10, 3)),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
    (  # test 0.5 brightness increase
            tf.fill((1, 10, 10, 3), 0.2),
            tf.constant([0.5]),
            tf.fill((1, 10, 10, 3), 0.7),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
    (  # test 0.5 brightness decrease
            tf.fill((1, 10, 10, 3), 0.7),
            tf.constant([-0.5]),
            tf.fill((1, 10, 10, 3), 0.2),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
])
def test_brightness(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to always return predetermined value
    monkeypatch.setattr(tf.random, 'uniform', lambda *_: adjustment)
    # instantiate augmentor with brightness adjustment factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('brightness', max_brightness_adjustment=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # test no adjustment
            tf.fill((1, 10, 10, 3), 0.5),
            tf.constant([0., 0., 0.]),
            tf.fill((1, 10, 10, 3), 0.5),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
    (  # test 0.5 blue increase
            tf.fill((1, 10, 10, 3), 0.2),
            tf.constant([0., 0., 0.5]),
            tf.concat([tf.fill((1, 10, 10, 2), 0.2), tf.fill((1, 10, 10, 1), 0.7)], axis=-1),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
    (  # test 0.5 green decrease
            tf.fill((1, 10, 10, 3), 0.7),
            tf.constant([0., -0.5, 0.]),
            tf.stack([tf.fill((1, 10, 10), 0.7), tf.fill((1, 10, 10), 0.2), tf.fill((1, 10, 10), 0.7)], axis=-1),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.range(300), 2),
                    (300, 2)
                ),
                tf.ones(300),
                (300, 300)
            ), (1, 10, 10, 3, ) * 2)
    ),
])
def test_color(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to always return predetermined value
    monkeypatch.setattr(tf.random, 'uniform', lambda *_: adjustment)
    # instantiate augmentor with color adjustment factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('color', max_color_adjustment=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # test no adjustment
            tf.ones((1, 2, 2, 3)),
            tf.constant([0.]),
            tf.ones((1, 2, 2, 3)),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.zeros((12, 12)),  # begin with (-a/N)
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),  # identity index
                tf.ones(12),  # add 1+a
            ), (1, 2, 2, 3, ) * 2)
    ),
    (  # test 0.5 contrast increase on flat grey image (no output change)
            tf.fill((1, 2, 2, 3), 0.5),
            tf.constant([0.5]),
            tf.fill((1, 2, 2, 3), 0.5),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.fill((12, 12), -0.5/12.),  # begin with (-a/N)
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),  # identity index
                tf.fill(12, 1.5),  # add 1+a
            ), (1, 2, 2, 3, ) * 2)
    ),
    (  # test 0.5 contrast decrease on flat grey image (no output change)
            tf.fill((1, 2, 2, 3), 0.5),
            tf.constant([-0.5]),
            tf.fill((1, 2, 2, 3), 0.5),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.fill((12, 12), 0.5/12.),  # begin with (-a/N)
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),  # identity index
                tf.fill(12, 0.5),  # add 1+a
            ), (1, 2, 2, 3, ) * 2)
    ),
])
def test_contrast(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to always return predetermined value
    monkeypatch.setattr(tf.random, 'uniform', lambda *_: adjustment)
    # instantiate augmentor with contrast adjustment factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('contrast', max_contrast_adjustment=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # test no adjustment
            tf.fill((1, 2, 2, 3), 0.5),
            tf.constant([0., 0., 0.]),
            tf.fill((1, 2, 2, 3), 0.5),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.zeros((12, 12)),  # begin with (-a/N) in channel, zeros out
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),  # identity index
                tf.ones(12),  # add 1+a
            ), (1, 2, 2, 3, ) * 2)
    ),
    (  # test 0.5 blue saturation increase on flat grey image (no output change)
            tf.fill((1, 2, 2, 3), 0.2),
            tf.constant([0., 0., 0.5]),
            tf.fill((1, 2, 2, 3), 0.2),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.zeros((12, 12)),
                indices=tf.concat([
                    tf.stack([
                        tf.repeat(tf.range(2, 12, 3), 4),
                        tf.tile(tf.range(2, 12, 3), [4]),
                    ], axis=1),  # channel 2 index
                    tf.reshape(tf.repeat(tf.range(12), 2), (12, 2))  # identity index
                ], axis=0),
                updates=tf.concat([
                    tf.fill(16, -0.5/4.),  # add -a/N in channel
                    tf.ones(12)+tf.tile([0., 0., 0.5], [4]),  # add 1+a in identity
                ], axis=0)
            ), (1, 2, 2, 3, ) * 2)
    ),
    (  # test 0.5 green saturation decrease on flat grey image (no output change)
            tf.fill((1, 2, 2, 3), 0.7),
            tf.constant([0., -0.5, 0.]),
            tf.fill((1, 2, 2, 3), 0.7),
            tf.reshape(tf.tensor_scatter_nd_add(
                tf.zeros((12, 12)),
                indices=tf.concat([
                    tf.stack([
                        tf.repeat(tf.range(1, 12, 3), 4),
                        tf.tile(tf.range(1, 12, 3), [4]),
                    ], axis=1),  # channel 1 index
                    tf.reshape(tf.repeat(tf.range(12), 2), (12, 2))  # identity index
                    # channel 1 identity index
                ], axis=0),
                updates=tf.concat([
                    tf.fill(16, 0.5/4.),  # add -a/N in channel
                    tf.ones(12)+tf.tile([0., -0.5, 0.], [4]),  # add 1+a in identity
                ], axis=0)
            ), (1, 2, 2, 3, ) * 2)
    ),
])
def test_saturation(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to always return predetermined value
    monkeypatch.setattr(tf.random, 'uniform', lambda *_: adjustment)
    # instantiate augmentor with saturation adjustment factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('saturation', max_saturation_adjustment=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # no translation
            tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
            tf.zeros((1, 2)),
            tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
            tf.reshape(tf.scatter_nd(
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),
                tf.ones(12),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    ),
    (  # 1x1 translation with constant padding (zeros)
            tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
            tf.fill((1, 2), 0.5),
            tf.concat([
                tf.zeros((1, 1, 3, 3)),
                tf.concat([
                    tf.zeros((1, 2, 1, 3)),
                    tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
                ], axis=2),
            ], axis=1)[:, :2, :2, :],
            tf.reshape(tf.scatter_nd(
                tf.constant([[9, 0], [10, 1], [11, 2]]),
                tf.ones(3),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    ),
    (  # 0x-1 translation with constant padding (zeros)
            tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
            tf.constant([[0., -0.5]]),
            tf.concat([
                tf.reshape(tf.linspace(0., 1., 12), (1, 2, 2, 3)),
                tf.zeros((1, 2, 1, 3)),
            ], axis=2)[:, :, -2:, :],
            tf.reshape(tf.scatter_nd(
                tf.constant([[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]),
                tf.ones(6),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    ),
])
def test_translation(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to always return predetermined value
    monkeypatch.setattr(tf.random, 'uniform', lambda *_: adjustment)
    # instantiate augmentor with translation adjustment factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('translation', max_translation_adjustment=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)


@pytest.mark.parametrize('images,adjustment,expected_images,expected_jacobian', [
    (  # no cutout
            tf.ones((1, 2, 2, 3)),
            (tf.zeros((1, 2)), tf.zeros(2)),
            tf.ones((1, 2, 2, 3)),
            tf.reshape(tf.scatter_nd(
                tf.reshape(tf.repeat(tf.range(12), 2), (12, 2)),
                tf.ones(12),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    ),
    (  # center 1,0; size 1,1
            tf.ones((1, 2, 2, 3)),
            (tf.constant([[0.5, 0.0]]), tf.constant([0.5, 0.5])),
            tf.Variable(tf.ones((1, 2, 2, 3)))[0, 1, 0].assign(0),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.concat([tf.range(6), tf.range(9, 12)], axis=0), 2),
                    (9, 2)
                ),
                tf.ones(9),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    ),
    (  # center 0,1; size 2,1
            tf.ones((1, 2, 2, 3)),
            (tf.constant([[0.0, 0.5]]), tf.constant([1.0, 0.5])),
            tf.concat([tf.ones((1, 2, 1, 3)), tf.zeros((1, 2, 1, 3))], axis=2),
            tf.reshape(tf.scatter_nd(
                tf.reshape(
                    tf.repeat(tf.concat([tf.range(3), tf.range(6, 9)], axis=0), 2),
                    (6, 2)
                ),
                tf.ones(6),
                (12, 12),
            ), (1, 2, 2, 3,) * 2),
    )
])
def test_cutout(monkeypatch, images, adjustment, expected_images, expected_jacobian):
    # patch random call to return center, then repatch to return size
    center, size = adjustment

    def uniform_patch(*_):
        monkeypatch.setattr(tf.random, 'uniform', lambda *_: size)
        return center

    monkeypatch.setattr(tf.random, 'uniform', uniform_patch)
    # instantiate augmentor with cutout size factor of 1 to simplify the patched randomness
    augmentor = diffaug.Augmentor('cutout', max_cutout_size=1.)

    # capture gradient of augmentation
    with tf.GradientTape() as tape:
        tape.watch(images)
        new_images, = augmentor(images)

    # calculate augmentation jacobian
    jacobian = tape.jacobian(new_images, images)

    # use allclose to account for possible floating point errors
    assert tf.experimental.numpy.allclose(new_images, expected_images)
    assert tf.experimental.numpy.allclose(jacobian, expected_jacobian)
