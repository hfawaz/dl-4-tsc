# Encoder
import keras
import keras_contrib
import numpy as np
import pandas as pd
import time

from utils.utils import save_logs


class Classifier_ENCODER:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if (verbose == True):
            self.model.summary()
        self.verbose = verbose

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_layer)
        conv1 = keras_contrib.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
        conv2 = keras_contrib.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
        conv3 = keras_contrib.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='loss', save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 12
        nb_epochs = 100

        mini_batch_size = batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()