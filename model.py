import tensorflow as tf
from tensorflow.keras import layers, Model

class MLP(Model):
    def __init__(self, code_dim, n_layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.hidden = [layers.Dense(units=code_dim, use_bias=False) for _ in range(n_layers)]

    def call(self, inputs):
        for layer in self.hidden:
            inputs = layer(inputs)
        return inputs

class Encoder(Model):
    def __init__(self, img_rows, d, activation_function=layers.ReLU()):
        super(Encoder, self).__init__()

        s1 = s2 = s3 = 4
        s4 = 2
        kernel = 5
        act_fn = activation_function if isinstance(activation_function, layers.Layer) else layers.Lambda(lambda x: activation_function(x))

        encoder_layers = [
            layers.InputLayer(shape=(img_rows, img_rows, 1)),
            layers.Conv2D(64, kernel_size=kernel, strides=s1, padding="same"), act_fn,
            layers.Conv2D(64, kernel_size=kernel, strides=s2, padding="same"), act_fn,
            layers.Conv2D(128, kernel_size=kernel, strides=s3, padding="same"), act_fn,
            layers.Conv2D(256, kernel_size=kernel, strides=s4, padding="same"), act_fn,
            layers.Flatten(),
            layers.Dense(d),
            act_fn,
        ]

        self.enc = tf.keras.Sequential(encoder_layers)

    def call(self, inputs):
        return self.enc(inputs)

class Decoder(Model):
    def __init__(self, img_rows, d, activation_function=layers.ReLU()):
        super(Decoder, self).__init__()

        s1 = s2 = s3 = 4
        s4 = 2
        kernel = 5
        act_fn = activation_function if isinstance(activation_function, layers.Layer) else layers.Lambda(lambda x: activation_function(x))

        last_layer = img_rows // s1 // s2 // s3 // s4
        pre_flatten_shape = (last_layer, last_layer, 256)

        decoder_layers = [
            layers.InputLayer(shape=(d,)),
            layers.Dense(units=pre_flatten_shape[0] * pre_flatten_shape[1] * pre_flatten_shape[2]), act_fn,
            layers.Reshape(target_shape=pre_flatten_shape),
            layers.Conv2DTranspose(256, kernel_size=kernel, strides=s4, padding="same"), act_fn,
            layers.Conv2DTranspose(128, kernel_size=kernel, strides=s3, padding="same"), act_fn,
            layers.Conv2DTranspose(64, kernel_size=kernel, strides=s2, padding="same"), act_fn,
            layers.Conv2DTranspose(1, kernel_size=kernel, strides=s1, padding="same"),
            layers.Activation("tanh")
        ]

        self.dec = tf.keras.Sequential(decoder_layers)

    def call(self, inputs):
        return self.dec(inputs)

class AE(Model):
    def __init__(self, args):
        super(AE, self).__init__()
        d = args['d']
        self.l = args['l']
        shape = args['imgs_shape']
        self.L1_weight = args['L1_lambda']

        if self.l > 0:
            self.mlp = MLP(d, self.l)

        self.enc = Encoder(shape, d)
        self.dec = Decoder(shape, d)

    def encode(self, x):
        z = self.enc(x)
        return self.mlp(z) if self.l > 0 else z

    def decode(self, z):
        return self.dec(z)

    def call(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)

        mse_loss = tf.reduce_mean(tf.square(x_reconstructed - x))
        avg_z_loss = tf.reduce_mean(tf.abs(z))
        w_z_loss = self.L1_weight * avg_z_loss

        total_loss = mse_loss + w_z_loss
        return mse_loss, avg_z_loss, total_loss

