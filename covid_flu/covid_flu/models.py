"""
Module containing different tools and utilities for tensorflow models.
"""
import numpy as np
import tensorflow as tf

from covid_flu import config, utils

K = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
Model = tf.keras.Model
Sequential = tf.keras.Sequential
Input = tf.keras.Input


class Attention(layers.Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self,x):
        et = K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at,axis=-1)
        output = x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()


class Seq2Seq:
    def __init__(self,
                 history_length=25,
                 target_length=5,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 hidden_size=32,
                 pre_output_dense_size=16,
                 dropout=0,
                 state_embed_size=None):

        self.history_length = history_length
        self.target_length = target_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.pre_output_dense_size = pre_output_dense_size
        self.dropout = dropout
        self.state_embed_size = state_embed_size

        if self.state_embed_size:
            self.training_network, self.encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, \
                self.bridge, state_embed_layer, self.state_embed_model = self.build_training_network()
            self.decoder_model = self.build_inference_network(decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense)
        else:
            self.training_network, self.encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, \
                self.bridge = self.build_training_network()
            self.decoder_model = self.build_inference_network(decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense)

    def build_training_network(self, transfer_model=None, transfer_mode='freeze'):
        # Building the model
        # Taken from https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        # and https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb
        encoder_inputs = Input(shape=(self.history_length, 1))

        x = encoder_inputs
        state_h, state_c = None, None
        for i in range(self.num_encoder_layers):
            x, state_h, state_c = layers.LSTM(self.hidden_size, activation='tanh', dropout=self.dropout,
                                              name=f'encoder_lstm{i}', return_sequences=True, return_state=True)(x)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        bridge = layers.Dense(2 * self.hidden_size)
        if self.state_embed_size:
            # Just hard-coding the number of states
            state_inputs = Input(shape=(1,))
            state_embed_layer = layers.Embedding(53, self.state_embed_size)
            state_embeds = state_embed_layer(state_inputs)
            state_embeds = layers.Flatten()(state_embeds)
            state_embed_model = Model(state_inputs, state_embeds)

            x = layers.Concatenate(axis=-1)([state_h, state_c, state_embeds])
            x = bridge(x)

            decoder_init = layers.Lambda(lambda t: tf.split(t, 2, axis=-1))(x)
        else:
            x = layers.Concatenate(axis=-1)([state_h, state_c])
            x = bridge(x)
            decoder_init = layers.Lambda(lambda t: tf.split(t, 2, axis=-1))(x)


        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, 1))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        # NOTE: this only supports a single decoder layer at the moment
        decoder_lstm = layers.LSTM(self.hidden_size, return_sequences=True, return_state=True, activation='tanh',
                                   name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=decoder_init)

        decoder_pre_output = layers.Dense(self.pre_output_dense_size, activation='relu', name='decoder_pre_output')
        decoder_dense = layers.Dense(1, activation='linear', name='decoder_dense')

        decoder_pre_outputs = decoder_pre_output(decoder_outputs)
        decoder_outputs = decoder_dense(decoder_pre_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        if self.state_embed_size:
            inputs = [encoder_inputs, decoder_inputs, state_inputs]
        else:
            inputs = [encoder_inputs, decoder_inputs]
        model = Model(inputs, decoder_outputs)

        if transfer_model is not None:
            model.set_weights(transfer_model.get_weights())
            if transfer_mode == 'freeze':
                for layer in model.layers[:-2]:
                    layer.trainable = False
        model.compile(optimizer='adam', loss='mse')

        encoder_model = Model(encoder_inputs, encoder_states)

        if self.state_embed_size:
            return model, encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, bridge, \
                   state_embed_layer, state_embed_model
        else:
            return model, encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, bridge

    def build_inference_network(self, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense):
        decoder_state_input_h = Input(shape=(self.hidden_size,))
        decoder_state_input_c = Input(shape=(self.hidden_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]

        decoder_pre_outputs = decoder_pre_output(decoder_outputs)
        decoder_outputs = decoder_dense(decoder_pre_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        return decoder_model

    def decode_sequence(self, input_seq, pred_steps=None):
        if not pred_steps:
            pred_steps = self.target_length
        if self.state_embed_size:
            input_seq, state_idx = input_seq
        if input_seq.ndim == 2:
            input_seq = np.expand_dims(input_seq, 0)
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        if self.state_embed_size:
            states_value.append(self.state_embed_model(state_idx))
        x = tf.concat(states_value, axis=-1)
        x = self.bridge(x)
        x = tf.split(x, 2, axis=-1)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 1))

        # Populate the first target sequence with end of encoding series
        target_seq[0, 0, 0] = input_seq[0, -1, 0]

        # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
        # (to simplify, here we assume a batch of size 1).

        decoded_seq = np.zeros((1, pred_steps, 1))

        for i in range(pred_steps):
            output, h, c = self.decoder_model.predict([target_seq] + x)

            decoded_seq[0, i, 0] = output[0, 0, 0]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 1))
            target_seq[0, 0, 0] = output[0, 0, 0]

            # Update states
            states_value = [h, c]

        return decoded_seq

    def predict(self, X, pred_steps=None):
        if not pred_steps:
            pred_steps = self.target_length
        if X.ndim == 2:
            X = np.expand_dims(X, 0)

        states_value = self.encoder_model.predict(X)
        target_seq = np.zeros((X.shape[0], 1, 1))
        target_seq[:, 0, 0] = X[:, -1, 0]
        decoded = np.zeros((X.shape[0], pred_steps, 1))

        for i in range(pred_steps):
            output, h, c = self.decoder_model.predict([target_seq] + states_value)
            decoded[:, i, 0] = output[:, 0, 0]
            target_seq = np.zeros((X.shape[0], 1, 1))
            target_seq[:, 0, 0] = output[:, 0, 0]

            states_value = [h, c]

        return decoded

    def summary(self):
        self.training_network.summary()

    def fit(self, *args, **kwargs):
        return self.training_network.fit(*args, **kwargs)

    def transfer(self, freeze=True):
        transfer_mode = 'freeze' if freeze else 'unfreeze'
        if self.state_embed_size:
            training_network, encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, \
                bridge, state_embed_layer, state_embed_model = self.build_training_network(transfer_model=self.training_network,
                                                                                           transfer_mode=transfer_mode)
            decoder_model = self.build_inference_network(decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense)
        else:
            training_network, encoder_model, decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense, \
                bridge = self.build_training_network(transfer_model=self.training_network,
                                                     transfer_mode=transfer_mode)
            decoder_model = self.build_inference_network(decoder_inputs, decoder_lstm, decoder_pre_output, decoder_dense)


        model = Seq2Seq(history_length=self.history_length,
                        target_length=self.target_length,
                        num_encoder_layers=self.num_encoder_layers,
                        num_decoder_layers=self.num_decoder_layers,
                        hidden_size=self.hidden_size,
                        pre_output_dense_size=self.pre_output_dense_size,
                        dropout=self.dropout)
        model.training_network = training_network
        model.decoder_model = decoder_model
        if self.state_embed_size:
            state_inputs = Input(shape=(1,))
            state_embed_layer_ = layers.Embedding(53, self.state_embed_size,
                                                  weights=state_embed_layer.get_weights(),
                                                  trainable=False)
            state_embeds = state_embed_layer_(state_inputs)

            state_embeds = layers.Flatten()(state_embeds)

            state_embed_model = Model(state_inputs, state_embeds)
            model.state_embed_model = state_embed_model
            # model.state_embed_model = tf.keras.models.clone_model(state_embed_model)

        return model


def build_seq2seq_with_attention(history_length=25,
                                 target_length=5,
                                 num_encoder_layers=2,
                                 num_decoder_layers=2,
                                 hidden_size=32,
                                 pre_output_dense_size=16,
                                 dropout=0):
    inputs=layers.Input(shape=(history_length,1))
    for i in range(num_encoder_layers):
        if i==0:
            x=layers.LSTM(hidden_size, activation='relu', return_sequences=True, dropout=dropout)(inputs)
        else:
            x=layers.LSTM(hidden_size, activation='relu', return_sequences=True, dropout=dropout)(x)
    att_out=Attention()(x)
    decoded = layers.RepeatVector(target_length)(att_out)
    for i in range(num_decoder_layers):
        decoded = layers.LSTM(hidden_size, activation='relu', return_sequences=True, dropout=dropout)(decoded)
    if pre_output_dense_size!=None:
        decoded = layers.Dense(pre_output_dense_size, activation='relu')(decoded)
        decoded = layers.Dropout(dropout)(decoded)
    decoded = layers.Dense(1)(decoded)

    model = tf.keras.Model(inputs,decoded)
    model.compile(loss='mse', optimizer='adam')
    return model


def transfer_seq2seq(model):
    model_tl = tf.keras.models.clone_model(model)
    for layer in model_tl.layers:
        if 'dense' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    model_tl.compile(loss='mse', optimizer='adam')
    return model_tl
