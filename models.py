import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class PositionEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PositionEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def positional_encoding(num_patches, projection_dim):
    depth = projection_dim / 2

    positions = np.arange(num_patches)[:, np.newaxis]     # (num_patches, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth      # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (num_patches, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class AngularPositionEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(AngularPositionEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)

    def call(self, patch):
        pos_encoding = positional_encoding(self.num_patches, self.projection.units)
        encoded = self.projection(patch) + pos_encoding[tf.newaxis, ...]
        return encoded


def choose_model(encoder_type,
                 decoder_type, 
                 latent_dim,
                 time_step,
                 feature_num,
                 d_model,
                 num_heads,
                 num_transformer_blocks,
                 activation='tanh',
                 position_encoder=None,
                 encoder_encoding=False,
                 decoder_encoding=False):

    """
    Construct an encoder-decoder model architecture for sequence-to-sequence tasks.

    Parameters:
    encoder_type (str): Type of encoder to use ('LSTM' or 'Transformer').
    decoder_type (str): Type of decoder to use ('LSTM' or 'Transformer').
    latent_dim (int): Dimensionality of the latent space.
    time_step (int): Number of time steps in input sequences.
    feature_num (int): Number of features at each time step.
    d_model (int): Dimensionality of the Transformer model.
    num_heads (int): Number of Transformer heads.
    num_transformer_blocks (int): Number of Transformer blocks to stack.
    activation (str): Activation function for the model layers (default is 'tanh').
    position_encoder (str, optional): Type of positional encoding ('angular', 'embedding', or None).
    encoder_encoding (bool, optional): Apply positional encoding to encoder input sequences (for Transformer encoder).
    decoder_encoding (bool, optional): Apply positional encoding to decoder input sequences (for Transformer decoder).
    
    Returns:
    encoder (tf.keras.Model): Constructed encoder model.
    decoder (tf.keras.Model): Constructed decoder model.
    """

    if encoder_type == 'LSTM':
        encoder_inputs = keras.Input(shape=(time_step, feature_num))
        x              = tf.keras.layers.LSTM(128, activation=activation, dropout=0.1, return_sequences=True)(encoder_inputs)
        x              = tf.keras.layers.LSTM(64, activation=activation, dropout=0.1,return_sequences=False)(x)
        # x              = layers.Dense(256, activation=activation)(x)
        z_mean         = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var      = layers.Dense(latent_dim, name="z_log_var")(x)
        z              = Sampling()([z_mean, z_log_var])
        encoder        = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
    elif encoder_type == 'Transformer':
        encoder_inputs = keras.Input(shape=(time_step, feature_num))
        if position_encoder == 'angular' and encoder_encoding:
            x              = AngularPositionEncoder(time_step, d_model)(encoder_inputs)
        elif position_encoder == 'embedding' and encoder_encoding:
            x              = PositionEncoder(time_step, d_model)(encoder_inputs)
        else:
            x              = tf.keras.layers.Dense(units=d_model)(encoder_inputs)

        for _ in range(num_transformer_blocks):
            x   = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
            x_h = layers.LayerNormalization(epsilon=1e-6)(x)
            x_h = layers.Dropout(0.1)(x_h)
#             x   = layers.Add()([x, x_h])
            x   = x_h
            x   = layers.LayerNormalization(epsilon=1e-6)(x)
            x_h = layers.Dense(d_model * 2, activation=activation)(x)
            x_h = layers.Dense(d_model,     activation=activation)(x_h)
#             x   = layers.Add()([x, x_h])
            x   = x_h

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        z_mean         = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var      = layers.Dense(latent_dim, name="z_log_var")(x)
        z              = Sampling()([z_mean, z_log_var])
        encoder        = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        
    if decoder_type == 'LSTM':
        
        latent_inputs   = keras.Input(shape=(latent_dim,))
        x               = layers.Dense(time_step * feature_num, activation=activation)(latent_inputs)
        x               = layers.Reshape((time_step, feature_num))(x)
        x               = tf.keras.layers.LSTM(32, activation=activation, return_sequences=True)(x)
        x               = tf.keras.layers.LSTM(64, activation=activation, return_sequences=True)(x)
        decoder_outputs = tf.keras.layers.LSTM(feature_num, activation=activation, return_sequences=True)(x)
#         decoder_outputs = tf.keras.layers.Lambda(lambda x: x * threshold)(decoder_outputs)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    elif decoder_type == 'Transformer':
        
        latent_inputs   = keras.Input(shape=(latent_dim,))
        x               = layers.Dense(time_step * feature_num, activation=activation)(latent_inputs)
        x               = layers.Reshape((time_step, feature_num))(x)        
        if position_encoder == 'angular' and decoder_encoding:
            x              = AngularPositionEncoder(time_step, d_model)(x)
        elif position_encoder == 'embedding' and decoder_encoding:
            x              = PositionEncoder(time_step, d_model)(x)
        else:
            x              = tf.keras.layers.Dense(units=d_model)(x)

        for _ in range(num_transformer_blocks):
            x   = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
            x_h = layers.LayerNormalization(epsilon=1e-6)(x)
            x_h = layers.Dropout(0.1)(x_h)
#             x   = layers.Add()([x, x_h])
            x   = x_h
            x   = layers.LayerNormalization(epsilon=1e-6)(x)
            x_h = layers.Dense(d_model * 2, activation=activation)(x)
            x_h = layers.Dense(d_model,     activation=activation)(x_h)
            x   = x_h
#             x   = layers.Add()([x, x_h])

        decoder_output = layers.Dense(feature_num)(x)


        decoder = keras.Model(latent_inputs, decoder_output, name="decoder")
        decoder.summary()
        
    return encoder, decoder




class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, loss_type,  kl_weights, **kwargs):
        super().__init__(**kwargs)
        self.encoder    = encoder
        self.decoder    = decoder
        self.loss_type  = loss_type
        self.kl_weights = kl_weights # (kl_weight, kl_weight_start, kl_decay_rate)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        if self.loss_type == 'mae':
            loss = keras.losses.mae
        elif self.loss_type == 'mse':
            loss = keras.losses.mse

        kl_weight       = self.kl_weights[0]
        kl_weight_start = self.kl_weights[1]
        kl_decay_rate   = self.kl_weights[2]

            
        step  = tf.cast(self.optimizer.iterations, tf.float32)
        klw   = kl_weight - (kl_weight - kl_weight_start) * kl_decay_rate ** step
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    loss(data, reconstruction), axis=1
                )
            )
            kl_loss = -klw * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)))
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    
    def test_step(self, data):
        if self.loss_type == 'mae':
            loss = keras.losses.mae
        elif self.loss_type == 'mse':
            loss = keras.losses.mse
            
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    loss(data, reconstruction), axis=1
                )
            )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }
                