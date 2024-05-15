import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.activations import tanh

class MLP(Model):
    def __init__(self, code_dim, n_layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.hidden = []
        for k in range(n_layers):
            linear_layer = layers.Dense(units=code_dim, use_bias=False)
            self.hidden.append(linear_layer)

    def call(self, inputs):
        for l in self.hidden:
            inputs = l(inputs)
        return inputs

class Encoder(Model):
    def __init__(self, img_rows, d, activation_function=layers.ReLU()):
        super(Encoder, self).__init__()

        s1 = s2 = s3 = 4
        s4 = 2
        kernel = 5

        if not isinstance(activation_function, layers.Layer):
            act_fn = layers.Lambda(lambda x: activation_function(x))
        else:
            act_fn = activation_function

        encoder_layers =  [
            layers.InputLayer(input_shape=(img_rows, img_rows, 1)),

            layers.Conv2D(2**6, kernel_size=kernel, strides=s1, padding="same"),
            act_fn,

            layers.Conv2D(2**6, kernel_size=kernel, strides=s2, padding="same"),
            act_fn,

            layers.Conv2D(2**7, kernel_size=kernel, strides=s3, padding="same"),
            act_fn,

            layers.Conv2D(2**8, kernel_size=kernel, strides=s4, padding="same"),
            act_fn,

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
        img_rows = img_rows
        d = d

        s1 = s2 = s3 = 4
        s4 = 2
        kernel = 5

        if not isinstance(activation_function, layers.Layer):
            act_fn = layers.Lambda(lambda x: activation_function(x))
        else:
            act_fn = activation_function

        last_layer = img_rows // s1  // s2 // s3 // s4
        pre_flatten_shape = (last_layer, last_layer, 2**8)

        decoder_layers = [
            layers.InputLayer(input_shape=(d,)),

            # Reshape to match previous Conv2D layer
            layers.Dense(units=pre_flatten_shape[0] * pre_flatten_shape[1] * pre_flatten_shape[2]),
            act_fn,
            layers.Reshape(target_shape=pre_flatten_shape),

            layers.Conv2DTranspose(2**8, kernel_size=kernel, strides=s4, padding="same"),
            act_fn,

            layers.Conv2DTranspose(2**7, kernel_size=kernel, strides=s3, padding="same"),
            act_fn,

            layers.Conv2DTranspose(2**6, kernel_size=kernel, strides=s2, padding="same"),
            act_fn,

            layers.Conv2DTranspose(1, kernel_size=kernel, strides=s1, padding="same"),
            layers.Activation("tanh")
        ]

        self.dec = tf.keras.Sequential(decoder_layers)

    def call(self, inputs):
        return self.dec(inputs)


class AE(tf.keras.Model):
    def __init__(self, args):
        super(AE, self).__init__()
        print(args)
        n = args['n']
        self.l = args['l']
        shape = args['imgs_shape']
        self.L1_weight = args['L1_lambda']

        if self.l > 0:
            l_weight = 0
            self.mlp = MLP(n, self.l)

        self.enc = Encoder(shape, n)
        self.dec = Decoder(shape, n)

    def encode(self, x):
        z = self.enc(x)
        if self.l > 0:
            z_bar = self.mlp(z)
        else:
            z_bar = z
        return z_bar

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
