import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from DataLoader import DataLoader
from cnn_classifier import ClassifierCNN


def main(data_name, method=None, combined=True):
    """
    Evaluate the CNN classifier on the augmented data set.
    :param data_name: str, data set
    :param method: str or None, the augmentation method
    :param combined: bool, if True, combine the original train set with the synthetic data
    :return:
    """

    print("-----------------------------------")
    print("ORIGINAL DATA SET")
    print("-----------------------------------")
    dt = DataLoader(path="C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data",
                    data_name=data_name,
                    bootstrap_test=True)
    dt.describe()
    X_train, y_train, X_test, y_test = dt.get_X_y(one_hot_encoding=False)

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()

    nb_classes = y_train_enc.shape[1]

    if method is None:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(X_train, y_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("CGAN DATA SET")
    print("-----------------------------------")
    dt = DataLoader(path="", data_name=data_name, cgan=True)
    dt.describe()
    Xs_train, ys_train, _, _ = dt.get_X_y(one_hot_encoding=False)

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(
        np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1), ys_train.reshape(-1, 1)), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    ys_train_enc = enc.transform(ys_train.reshape(-1, 1)).toarray()

    nb_classes = ys_train_enc.shape[1]

    if method == "cgan" and not combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(Xs_train, ys_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("ORIG+CGAN DATA SET")
    print("-----------------------------------")

    if method == "cgan" and combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(np.concatenate((X_train, Xs_train), axis=0),
                         np.concatenate((y_train_enc, ys_train_enc), axis=0),
                         X_test, y_test_enc)

    print("-----------------------------------")
    print("JITTERED DATA SET")
    print("-----------------------------------")
    file_name = "gen_data/aug_jitt_" + data_name
    df = pd.read_csv(glob.glob(file_name + "*.tsv")[0], sep='\t', header=0, index_col=0)
    Xs_train, ys_train = df[df.columns[:-1]], df["l"].to_numpy().reshape(-1, 1)

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(
        np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1), ys_train.reshape(-1, 1)), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    ys_train_enc = enc.transform(ys_train.reshape(-1, 1)).toarray()

    nb_classes = ys_train_enc.shape[1]

    if method == "jit" and not combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(Xs_train, ys_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("ORIG+JITT DATA SET")
    print("-----------------------------------")
    if method == "jit" and combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(np.concatenate((X_train, Xs_train), axis=0),
                         np.concatenate((y_train_enc, ys_train_enc), axis=0),
                         X_test, y_test_enc)

    print("-----------------------------------")
    print("FLIPPED DATA SET")
    print("-----------------------------------")
    file_name = "gen_data/aug_flip_" + data_name
    df = pd.read_csv(glob.glob(file_name + "*.tsv")[0], sep='\t', header=0, index_col=0)
    Xs_train, ys_train = df[df.columns[:-1]], df["l"].to_numpy().reshape(-1, 1)

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(
        np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1), ys_train.reshape(-1, 1)), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    ys_train_enc = enc.transform(ys_train.reshape(-1, 1)).toarray()

    nb_classes = ys_train_enc.shape[1]

    if method == "flip" and not combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(Xs_train, ys_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("ORIG+FLIP DATA SET")
    print("-----------------------------------")
    if method == "flip" and combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(np.concatenate((X_train, Xs_train), axis=0),
                         np.concatenate((y_train_enc, ys_train_enc), axis=0),
                         X_test, y_test_enc)

    print("-----------------------------------")
    print("PERMUTED DATA SET")
    print("-----------------------------------")
    file_name = "gen_data/aug_perm_" + data_name
    df = pd.read_csv(glob.glob(file_name + "*.tsv")[0], sep='\t', header=0, index_col=0)
    Xs_train, ys_train, = df[df.columns[:-1]], df["l"].to_numpy().reshape(-1, 1)

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(
        np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1), ys_train.reshape(-1, 1)), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    ys_train_enc = enc.transform(ys_train.reshape(-1, 1)).toarray()

    nb_classes = ys_train_enc.shape[1]

    if method == "perm" and not combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(Xs_train, ys_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("ORIG+PERM DATA SET")
    print("-----------------------------------")
    if method == "perm" and combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(np.concatenate((X_train, Xs_train), axis=0),
                         np.concatenate((y_train_enc, ys_train_enc), axis=0),
                         X_test, y_test_enc)

    print("-----------------------------------")
    print("AVG_SMOTE DATA SET")
    print("-----------------------------------")
    file_name = "gen_data/aug_smote_" + data_name
    df = pd.read_csv(glob.glob(file_name + "*.tsv")[0], sep='\t', header=0, index_col=0)
    Xs_train, ys_train = df[df.columns[:-1]], df["l"].to_numpy().reshape(-1, 1)

    if len(Xs_train.shape) == 2:  # if univariate
        # add a dimension
        Xs_train = Xs_train.to_numpy().reshape((Xs_train.shape[0], Xs_train.shape[1], 1))

    input_shape = Xs_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(
        np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1), ys_train.reshape(-1, 1)), axis=0).reshape(-1, 1))
    y_train_enc = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    ys_train_enc = enc.transform(ys_train.reshape(-1, 1)).toarray()

    nb_classes = ys_train_enc.shape[1]

    if method == "smote" and not combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(Xs_train, ys_train_enc, X_test, y_test_enc)

    print("-----------------------------------")
    print("ORIG+AVG_SMOTE DATA SET")
    print("-----------------------------------")
    if method == "smote" and combined:
        ccnn = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes)
        ccnn.fit_predict(np.concatenate((X_train, Xs_train), axis=0),
                         np.concatenate((y_train_enc, ys_train_enc), axis=0),
                         X_test, y_test_enc)


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'gunpoint'
    # method = [None, 'jit', 'flip', 'perm', 'smote', 'cgan']
    method = 'smote'
    main(data_name=data_name, method=method, combined=True)
