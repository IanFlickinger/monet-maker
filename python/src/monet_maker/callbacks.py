from typing import Union, Iterable

import tensorflow as tf

from . import vis


class VisualizeCycleGanEvolution(tf.keras.callbacks.Callback):
    DEFAULT_FILEPATH = './cycle-gan-evolution.png'

    def __init__(
            self,
            test_images: Union[tf.Tensor, Iterable[tf.Tensor]],
            classes: Union[None, str, Iterable[str]] = None,
            frequency: Union[int, Iterable[int]] = 1,
            filepath: str = DEFAULT_FILEPATH,
            separate_classes: bool = True,
            show_initial: bool = True,
    ):
        """
        Args:
            test_images (Union[tf.Tensor, Iterable[tf.Tensor]]): tensor containing the batch of images to test on. If
                multiple classes are being visualized, test_images should be an iterable containing a batch for each
                class. Index of batches should match the index of the class in classes for which each batch is to be
                transformed into.
            classes (Union[None, str, Iterable[str]]): The name(s) of the classes explored by the CycleGAN model. Will
                each be used as an argument to the __call__ method of the CycleGAN. If None, length of classes is
                assumed to be 1, and the model will be called with no other arguments.
            frequency (Union[int, Iterable[int]]). If single int, test will be run at the end of every epoch such that
                'epoch % frequency == 0' evaluates to True. If Iterable, test will be run whenever 'epoch in frequency'
                evaluates to True. Epoch in this consideration will begin at one - not zero.
            filepath (str): The location at which to save the resulting image.
            separate_classes (bool): If true, each class will be saved as a separate image with the class prepended to
                the file name.
            show_initial (bool): If true, will include initial predictions of the gan model (before any training
                occurs).
        """
        super().__init__()

        # ensure classes is Iterable[str]
        if classes is None or type(classes) is str:
            classes = [classes, ]

        # images tensor should be of shape (epoch, class, image, height, width[, channels])
        if len(classes) == 1:
            # test_images is of shape (image, height, width[, channels])
            self.images = test_images[None, None]
        else:
            # test_images is iterable with tensors
            self.images = tf.stack(test_images)[None]

        # generate save paths
        if separate_classes and len(classes) > 1:
            name_index = max(0, filepath.rfind('/') + 1)
            self.filepaths = [filepath[:name_index] + class_name + '-' + filepath[name_index:]
                              for class_name in classes]
        else:
            self.filepaths = [filepath, ]

        # assign remaining args to attributes
        self.classes = classes
        self.frequency = frequency
        self.show_initial = show_initial

    def on_train_begin(self, logs=None):
        # collect initial transformations
        if self.show_initial:
            self.images = self._collect_images(self.images)

    def on_epoch_end(self, epoch, logs=None):
        # check if frequency dictates this epoch to be detailed
        epoch += 1
        if (
                type(self.frequency) is int and epoch % self.frequency == 0 or
                hasattr(self.frequency, '__iter__') and epoch in self.frequency
        ):
            self.images = self._collect_images(self.images)

    #     @tf.function # TODO: either finish tensorflowizing this, or reformat to need no arguments or returns
    def _collect_images(self, images):
        # initialize new tensor with shape (0, height, width[, channels])
        new_images = tf.zeros([0, *images.shape[3:]], dtype=images.dtype)

        # iterate over classes and images
        for c, cla in enumerate(self.classes):
            # extract original images
            oimgs = images[0, c]
            # transform image batch (with class name as argument if available)
            nimgs = self.model(oimgs, cla) if cla else self.model(oimgs)
            # concatenate along image axis
            new_images = tf.concat((new_images, nimgs), axis=0)

        # add epoch and class axes to tensor
        new_images = tf.reshape(new_images, (len(self.classes), -1, *new_images.shape[1:]))[None]
        # concatenate existing epoch data with new
        return tf.concat((images, new_images), axis=0)

    def on_train_end(self, logs=None):
        rank = len(self.images.shape)
        # ensure channels axis exists
        if rank == 5:
            self.images = self.images[..., None]

        # reshape images from (epoch, class, image, height, width, channels)
        #                  to (class, image, epoch, height, width, channels)
        images = tf.transpose(self.images, [1, 2, 0, 3, 4, 5])

        if len(self.filepaths) > 1:
            [vis.generator_evolution(img[None], fp) for img, fp in zip(images, self.filepaths)]
        else:
            vis.generator_evolution(images, self.filepaths[0])


class AlternateTraining(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        gen_batch = (batch % 2 == 0)
        self.model.m_gen.trainable = self.model.p_gen.trainable = gen_batch
        self.model.m_dis.trainable = self.model.p_dis.trainable = not gen_batch
