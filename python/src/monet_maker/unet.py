def Unet(
        input_shape=DataConfig.IMAGE_SHAPE,
        unet_depth=UnetConfig.UNET_DEPTH,
        base_depth=UnetConfig.BASE_DEPTH,
        conv_depth=UnetConfig.CONV_DEPTH,
        dropout_rate=UnetConfig.DROPOUT_RATE,
        activation=UnetConfig.ACTIVATION,
        base=UnetConfig.BASE_TYPE,
):
    # Input Layer
    inputs = tf.keras.Input(input_shape)

    # initialize empty list to track unet skip connections
    skips = list()

    # Down Stack
    dn_stack = inputs
    for level in range(unet_depth):
        # skip
        skips.append(dn_stack)
        # downsample and increase filters
        filters_in = dn_stack.shape[-1]
        filters_out = unet_filters_at_level(level + 1)
        dn_stack = unet_downsample(filters_in, filters_out)(dn_stack)

    # Base Stack
    base_stack = dn_stack
    base_shape = dn_stack.shape[1:]
    if base == 'residual':
        base_stack = unet_residual_base(base_shape, base_depth)(base_stack)
    # TODO: inception base option
    elif base == 'inception':
        base_stack = unet_inception_base(base_shape, base_depth)(base_stack)
    # TODO: xception base option
    elif base == 'xception':
        base_stack = unet_xception_base(base_shape, base_depth)(base_stack)

    # Up Stack
    up_stack = base_stack
    for level, skip in reversed(list(enumerate(skips))):
        # upsample and decrease filters
        filters_in = up_stack.shape[-1]
        filters_out = skip.shape[-1]
        up_stack = unet_upsample(filters_in, filters_out)(up_stack)
        # concatenate (or add) skip connection along channel axis
        up_stack = tf.keras.layers.Add()([up_stack, skip])

    # Output
    outputs = up_stack
    # convolve reconstructed pixels with originals
    outputs = unet_upsample(
        filters_in=3,
        filters_out=3,
        dropout_rate=0,
        sample_rate=1,
        activation='tanh'
    )(outputs)
    # rescale pixel values from [-1, 1] of tanh output to [0, 1]
    outputs = tf.keras.layers.Rescaling(scale=0.5, offset=0.5)(outputs)

    # Assemble Model
    unet = tf.keras.Model(inputs=inputs, outputs=outputs)
    return unet


# TODO: put in primary config for easy experimentation
def unet_filters_at_level(level):
    return min(32 * (2 ** level), 512)


def unet_downsample(
        filters_in,
        filters_out,
        depth=3,
        kernel_size=3,
        sample_rate=2,
        dropout_rate=UnetConfig.DROPOUT_RATE,
        normalize=True,
        activation='leaky_relu',
        kernel_initializer='glorot_uniform',
):
    block = tf.keras.Sequential()

    # dropout
    if dropout_rate:
        block.add(tf.keras.layers.SpatialDropout2D(dropout_rate))

    # conv layers
    for _ in range(depth - 1):
        block.add(tf.keras.layers.Conv2D(
            filters_in,
            kernel_size,
            padding='same',
            kernel_initializer=kernel_initializer
        ))

    # downsample layer
    block.add(tf.keras.layers.Conv2D(
        filters_out,
        kernel_size,
        strides=sample_rate,
        padding='same',
        kernel_initializer=kernel_initializer,
    ))

    # normalize
    if normalize:
        block.add(tfa.layers.InstanceNormalization())

    # activate
    if activation:
        block.add(tf.keras.layers.Activation(activation))

    return block


def unet_upsample(
        filters_in,
        filters_out,
        depth=3,
        kernel_size=3,
        sample_rate=2,
        dropout_rate=UnetConfig.DROPOUT_RATE,
        normalize=True,
        activation='leaky_relu',
        kernel_initializer='glorot_uniform',
):
    block = tf.keras.Sequential()

    # dropout
    if dropout_rate:
        block.add(tf.keras.layers.SpatialDropout2D(dropout_rate))

    # conv layers
    for _ in range(depth - 1):
        block.add(tf.keras.layers.Conv2D(
            filters_in,
            kernel_size,
            padding='same',
            kernel_initializer=kernel_initializer,
        ))

    # upsample layer
    block.add(tf.keras.layers.Conv2DTranspose(
        filters_out,
        kernel_size,
        strides=sample_rate,
        padding='same',
        kernel_initializer=kernel_initializer,
    ))

    # normalize
    if normalize:
        block.add(tfa.layers.InstanceNormalization())

    # activate
    if activation:
        block.add(tf.keras.layers.Activation(activation))

    return block


def unet_residual_base(
        base_shape,
        stack_depth,
        kernel_size=3,
        dropout_rate=UnetConfig.DROPOUT_RATE,
        activation=UnetConfig.ACTIVATION,
        compression_factor=0.5,
        preactivation=True,
        bottleneck=True,
        normalize=True,
):
    stack = stack_input = tf.keras.Input(base_shape)

    normalized = (lambda stack: tfa.layers.InstanceNormalization()(stack))
    activated = (lambda stack: tf.keras.layers.Activation(activation)(stack))
    actnorm = (lambda stack: activated(normalized(stack)))

    base_filters = base_shape[-1]
    neck_filters = int(base_filters * compression_factor)
    for _ in range(stack_depth):
        block = block_input = stack

        if dropout_rate:
            block = tf.keras.layers.SpatialDropout2D(dropout_rate)(block)

        if preactivation:
            block = activated(block)

        if bottleneck:
            # compression conv
            block = tf.keras.layers.Conv2D(neck_filters, 1)(block)
            block = actnorm(block)
            # bottleneck conv
            block = tf.keras.layers.Conv2D(neck_filters, kernel_size, padding='same')(block)
            block = actnorm(block)
            # expansion conv
            block = tf.keras.layers.Conv2D(base_filters, 1)(block)
            block = normalized(block)
        else:
            block = tf.keras.layers.Conv2D(base_filters, kernel_size, padding='same')(block)
            block = actnorm(block)
            block = tf.keras.layers.Conv2D(base_filters, kernel_size, padding='same')(block)
            block = normalized(block)

        stack = tf.keras.layers.Add()([block, block_input])

        if not preactivation:
            stack = activated(stack)

    return tf.keras.Model(inputs=stack_input, outputs=stack)