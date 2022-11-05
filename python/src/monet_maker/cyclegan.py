class CycleGAN(tf.keras.Model):
    def __init__(self, m_gen, p_gen, m_dis, p_dis, aug):
        super().__init__()
        self.m_gen = m_gen
        self.p_gen = p_gen
        self.m_dis = m_dis
        self.p_dis = p_dis
        self.aug = aug

    def compile(self, optimizer, id_loss, cycle_loss, dis_loss, gen_loss_weights):
        super().compile()
        self.optimizer = optimizer
        self.id_loss = id_loss
        self.cycle_loss = cycle_loss
        self.dis_loss = dis_loss
        self.gen_loss_weights = gen_loss_weights

    def train_step(self, data):
        # read tupled batch data = (monet_batch, photo_batch)
        m_real, p_real = data

        # Progress batch data through CycleGAN process
        with tf.GradientTape(persistent=True) as g_tape:
            # identity outputs
            m_id = self.m_gen(m_real, training=True)
            p_id = self.p_gen(p_real, training=True)

            # identity loss
            m_id_loss = self.id_loss(m_real, m_id)
            p_id_loss = self.id_loss(p_real, p_id)

            # transfer outputs
            m_fake = self.m_gen(p_real, training=True)
            p_fake = self.p_gen(m_real, training=True)

            # cycle outputs
            m_cycle = self.m_gen(p_fake, training=True)
            p_cycle = self.p_gen(m_fake, training=True)

            # cycle loss
            m_cycle_loss = self.cycle_loss(m_real, m_cycle)
            p_cycle_loss = self.cycle_loss(p_real, p_cycle)
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

            labels_real = tf.ones_like(m_dis_real)
            labels_fake = tf.zeros_like(m_dis_fake)
            labels = tf.concat([labels_real, labels_fake], 0)

            m_dis_loss = self.dis_loss(labels, m_dis)
            p_dis_loss = self.dis_loss(labels, p_dis)

            # generator loss
            m_gen_loss = self.dis_loss(labels_real, m_dis_fake)
            p_gen_loss = self.dis_loss(labels_real, p_dis_fake)

            m_gen_loss = tf.tensordot(
                tf.stack([m_gen_loss, m_id_loss, cycle_loss]),
                self.gen_loss_weights, 1
            )
            p_gen_loss = tf.tensordot(
                tf.stack([p_gen_loss, p_id_loss, cycle_loss]),
                self.gen_loss_weights, 1
            )

        # collect model losses and variables
        models = [self.m_gen, self.p_gen, self.m_dis, self.p_dis]
        losses = [m_gen_loss, p_gen_loss, m_dis_loss, p_dis_loss]
        variables = [model.trainable_variables for model in models]

        # apply backpropagation
        for model_loss, model_vars in zip(losses, variables):
            grads = g_tape.gradient(model_loss, model_vars)
            self.optimizer.apply_gradients(zip(grads, model_vars))

        # return losses and metrics
        return {
            'monet_id_loss': m_id_loss,
            'photo_id_loss': p_id_loss,
            'monet_cycle_loss': m_cycle_loss,
            'photo_cycle_loss': p_cycle_loss,
            'monet_discriminator_loss': m_dis_loss,
            'photo_discriminator_loss': p_dis_loss
        }

    def call(self, x, output_class='monet'):
        if output_class == 'monet':
            return self.m_gen(x)
        if output_class == 'photo':
            return self.p_gen(x)