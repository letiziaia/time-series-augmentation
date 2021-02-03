# Ismail Fawaz, H., Forestier, G., Weber, J. et al.
# Deep learning for time series classification: a review.
# Data Min Knowl Disc 33, 917–963 (2019).
# https://doi.org/10.1007/s10618-019-00619-1
import time

import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from DataLoader import DataLoader


class ClassifierCNN:

    def __init__(self, input_shape, nb_classes, epochs=2000, mini_batch_size=16):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.model = self.__build_model()

        return

    def __build_model(self):
        # for italypowerondemand data
        padding = 'same' if self.input_shape[0] < 60 else 'valid'
        input_layer = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding=padding, activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding=padding, activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=self.nb_classes, activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        print(model.summary())

        return model

    def fit_predict(self, x_train, y_train, x_test, y_test):
        start_time = time.time()

        print("Start fitting...")

        hist = self.model.fit(x_train, y_train,
                              batch_size=self.mini_batch_size, epochs=self.epochs,
                              # x_test and y_test are used for monitoring only, NOT for training
                              verbose=True, validation_data=(x_test, y_test))
        print(hist)

        print("Total time:", time.time() - start_time)

        y_pred = self.model.predict(x_test)

        # convert y_pred back to original format
        y_pred = np.argmax(y_pred, axis=1)

        # convert back to original format for classification report
        y_test = np.argmax(y_test, axis=1)

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(balanced_accuracy_score(y_test, y_pred))

        keras.backend.clear_session()

        return y_pred


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'ecg200'
    dt = DataLoader(path="C:/Users/letiz/Desktop/Aalto/Bachelor\'s Thesis and Seminar - JOIN.bsc/data",
                    data_name=data_name,
                    bootstrap_test=True)
    dt.describe()
    X_train, y_train, X_test, y_test = dt.get_X_y(one_hot_encoding=True)
    nb_classes = y_train.shape[1]

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]

    print("-----------------------------------")
    print("ORIGINAL DATA SET")
    print("-----------------------------------")
    ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
    ccnn.fit_predict(X_train, y_train, X_test, y_test)

    dt = DataLoader(path="", data_name=data_name, cgan=True)
    dt.describe()
    Xs_train, ys_train, _, _ = dt.get_X_y(one_hot_encoding=True)
    nb_classes = ys_train.shape[1]

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    # print("-----------------------------------")
    # print("CGAN DATA SET")
    # print("-----------------------------------")
    # ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
    # ccnn.fit_predict(Xs_train, ys_train, X_test, y_test)
    #
    # print("-----------------------------------")
    # print("COMBINED DATA SET")
    # print("-----------------------------------")
    # ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
    # ccnn.fit_predict(np.concatenate((X_train, Xs_train),axis=0),
    #                  np.concatenate((y_train,ys_train), axis=0),
    #                  X_test, y_test)