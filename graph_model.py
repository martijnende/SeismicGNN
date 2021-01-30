import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Reshape, RepeatVector
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Lambda, Multiply, Permute
from tensorflow.keras.layers import Activation, SpatialDropout2D, UpSampling2D
from tensorflow.keras.layers import GaussianNoise, GaussianDropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import numpy as np
import pandas as pd
import os
import pickle


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_dir, catalogue, stations, lookup, N_sub, N_t, batch_size):
        """
        stations: (N_stations x 2)
        catalogue: (N_events x 5) {ID, lat, lon, depth, mag}
        """
        
        self.data_dir = data_dir
        self.cat = catalogue
        self.stations = stations
        self.lookup = lookup
        self.N_events = catalogue.shape[0]
        self.N_t = N_t
        self.N_sub = N_sub
        self.event_inds = np.arange(catalogue.shape[0]).astype(int)
        self.batch_size = batch_size

        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(self.N_events / float(self.batch_size))

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        pass

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        waveforms = self.waveforms[selection]
        coords = self.station_coords[selection]
        weights = self.weights[selection]
        labels = self.labels[selection]
        loss_weights = self.loss_weights[selection]
#         return (waveforms, coords, weights), labels, loss_weights
#         return (waveforms, coords, weights), (labels, waveforms)
        return (waveforms, coords, weights), labels, [None]

    def __data_generation(self):
        """ Generate a total batch """
        N_events, N_sub, N_t = self.N_events, self.N_sub, self.N_t
        catalogue = self.cat
        stations = self.stations
        event_inds = self.event_inds
        lookup = self.lookup
        
        np.random.shuffle(event_inds)
        
        waveforms = np.zeros((N_events, N_sub, N_t, 3))
        station_coords = np.zeros((N_events, N_sub, 1, 3))
        weights = np.ones((N_events, N_sub))
        labels = np.zeros((N_events, 4))
        loss_weights = np.zeros(N_events)
        
        for i in range(N_events):
            event = catalogue[event_inds[i]]
            event_id = int(event[0])
            station_codes = lookup[event_id]
            N_codes = len(station_codes)
            station_inds = np.arange(N_codes)
            
            event_data = np.load(os.path.join(self.data_dir, "%d.npy" % event_id))
            
            if N_codes >= N_sub: 
                selection = np.random.choice(station_inds, size=N_sub, replace=False)
                station_codes_select = station_codes[selection]
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values[selection]
                waveforms[i] = event_data[selection]
                station_coords[i, :, 0, :2] = coords
            else:
                waveforms[i, :N_codes] = event_data
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values
                station_coords[i, :N_codes, 0, :2] = coords
                weights[i, N_codes:] = 0
                
            labels[i] = event[2:]
            loss_weights[i] = event[1]
        
        self.waveforms = waveforms
        self.station_coords = station_coords
        self.weights = weights
        self.labels = labels
        self.loss_weights = loss_weights
        pass

    
class CallBacks:

    @staticmethod
    def tensorboard(logdir):
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            profile_batch=0,
            update_freq="epoch",
            histogram_freq=0,
        )
        return tensorboard_callback

    @staticmethod
    def checkpoint(savefile, best=True):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            savefile,
            verbose=0,
            save_weights_only=False,
            save_best_only=best,
            monitor="val_loss",
            mode="auto",
            update_freq="epoch",
        )
        return checkpoint_callback


class GraphNet:
    
    def __init__(self):
        self.kernel = (1, 5)
        self.f0 = 2
        self.use_bn = False
        self.use_dropout = True
        self.dropout_at_test = True
        self.dropout_rate = 0.1
        self.LR = 2e-4
        self.initializer = keras.initializers.Orthogonal()
        self.activation = "tanh"
        self.N_sub = 50
        self.N_t = 1024
        self.data_shape_waveforms = (self.N_sub, self.N_t, 3)
        self.data_shape_coords = (self.N_sub, 1, 3)
        self.data_shape_weights = (self.N_sub,)

        pass

    def set_params(self, params):
        """
        Update model parameters
        """
        self.__dict__.update(params)
        self.data_shape_waveforms = (self.N_sub, self.N_t, 3)
        self.data_shape_coords = (self.N_sub, 1, 3)
        self.data_shape_weights = (self.N_sub,)
        pass
    
    def conv_layer(self, x, filters, kernel_size, use_bn=False, use_dropout=False, activ=None, dropout_at_test=False):
        """
        Convolution layer > batch normalisation > activation > dropout
        """
        use_bias = True
        if use_bn:
            use_bias = False

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same",
            activation=None, kernel_initializer=self.initializer,
            use_bias=use_bias
        )(x)

        if use_bn:
            x = BatchNormalization()(x)

        if activ is not None:
            x = Activation(activ)(x)

        if use_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x, training=dropout_at_test)

        return x
    
    def construct(self):
        """
        Construct Graph Neural Network
        """
        
        f = self.f0
        kernel = self.kernel
        use_bn = self.use_bn
        use_dropout = self.use_dropout
        dropout_at_test = self.dropout_at_test
        activation = self.activation
        
        data_shape = self.data_shape_waveforms
        data_shape2 = self.data_shape_coords
        data_shape3 = self.data_shape_weights        
        
        input_data = Input(data_shape)
        input_coords = Input(data_shape2)
        input_weights = Input(data_shape3)

        x = input_data
        
        """
        Component 1: CNN that processes station waveforms
        """
        
        # (t, f): (Nb, Ns, 2048, 3) -> (Nb, Ns, 512, 8) -> (Nb, Ns, 128, 16) -> (Nb, Ns, 32, 32)
        # Construct 3 blocks, each with 3 convolutional layers
        for i in range(3):
            # Double number of filters each block
            f = f * 2
            for j in range(3):
                x = self.conv_layer(
                    x, filters=f, kernel_size=kernel, use_bn=use_bn, use_dropout=use_dropout, 
                    activ=activation, dropout_at_test=dropout_at_test
                )
            # Downsample time axis
            x = MaxPool2D(pool_size=(1, 4))(x)            
        
        # Last block is special (no final dropout or downsampling)
        # (Nb, Ns, 32, 32) -> (Nb, Ns, 32, 64)
        f = f * 2
        # Create 2 layers
        for i in range(2):
            x = self.conv_layer(
                x, filters=f, kernel_size=kernel, use_bn=use_bn, use_dropout=use_dropout, 
                activ=activation, dropout_at_test=dropout_at_test
            )
        # Final layer. Do not dropout!
        x = self.conv_layer(
            x, filters=f, kernel_size=kernel, use_bn=False, use_dropout=False, 
            activ="tanh", dropout_at_test=dropout_at_test
        )
        

        # Reduce: (Nb, Ns, 32, 64) -> (Nb, Ns, 1, 64)
        # x = Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(x)
        x = Lambda(lambda x: tf.reduce_max(x, axis=2, keepdims=True))(x)
        x = GaussianDropout(self.dropout_rate)(x)
        # Concatenate: (Nb, Ns, 1, 64) -> (Nb, Ns, 1, 64+2)
        x = concatenate([x, input_coords])
        
        """
        Component 2: MLP that processes extracted features and location
        """

        # Location processing (Nb, Ns, 1, 64+2) -> (Nb, Ns, 1, 64)
        x = self.conv_layer(
            x, filters=128, kernel_size=(1, 1), use_bn=False, use_dropout=use_dropout, 
            activ=activation, dropout_at_test=dropout_at_test
        )
        # Final layer. Do not dropout!
        x = self.conv_layer(
            x, filters=128, kernel_size=(1, 1), use_bn=False, use_dropout=False, 
            activ=activation, dropout_at_test=dropout_at_test
        )
        
        # Reshape weights: (Nb, Ns) -> (Nb, Ns, 64) -> (Nb, Ns, 1, 64)
        aggr_weights_reshape = RepeatVector(x.shape[-1])(input_weights)
        aggr_weights_reshape = Permute([2, 1])(aggr_weights_reshape)
        aggr_weights_reshape = tf.keras.backend.expand_dims(aggr_weights_reshape, axis=-2)

        # Reduce: (Nb, Ns, 1, 64) -> (Nb, 1, 1, 64) -> (Nb, 64)
        x = x * aggr_weights_reshape
        x = Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True))(x)
        x = Flatten()(x)
        x = GaussianDropout(self.dropout_rate)(x)
        
        """
        Component 3: MLP that combines aggregated node attributes 
        to predict source characteristics
        """        

        # Source characterisation (Nb, 64) -> (Nb, 4)
        x = Dense(128, activation=activation, kernel_initializer=self.initializer)(x)
        x = GaussianDropout(self.dropout_rate)(x, training=dropout_at_test)
        x = Dense(4, activation="tanh", kernel_initializer=self.initializer)(x)
        x = Lambda(lambda x: x * 1, name="prediction")(x)
        
        # Build and compile model
        model = Model([input_data, input_coords, input_weights], x)
        model.build(input_shape=(data_shape, data_shape2, data_shape3))
        
        self.model = model
        return model
