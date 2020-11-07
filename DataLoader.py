import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    """Load one of the 3 datasets (Freezer, InsectEPG, MixedShapes) from UCR time series archive.
    Each one of those 3 datasets has 2 different train sets, one of which is smaller and the other one is regular,
    and one unique test set for evaluation."""

    def __init__(self, path, data_name='freezer'):
        """
        Load the regular train set, small train set, and test set in the object.
        :param path: str, the path to the data folder
        :param data_name: str, a shorter name identifying the data set to load
        """
        data_name = data_name.lower()
        if 'insect' in data_name:
            # Load InsectEPG
            print("Loading the InsectEPG data set...")
            regular_train_source = path + "/InsectEPGRegularTrain"
            small_train_source = path + "/InsectEPGSmallTrain"

        elif 'shapes' in data_name:
            # Load MixedShapes
            print("Loading the MixedShapes data set...")
            regular_train_source = path + "/MixedShapesRegularTrain"
            small_train_source = path + "/MixedShapesSmallTrain"

        else:
            # Load freezer
            print("Loading the Freezer data set...")
            regular_train_source = path + "/FreezerRegularTrain"
            small_train_source = path + "/FreezerSmallTrain"
        self.short_train_df = pd.read_csv(glob.glob(small_train_source + "/*TRAIN.tsv")[0],
                                          sep='\t', header=None)
        self.regular_train_df = pd.read_csv(glob.glob(regular_train_source + "/*TRAIN.tsv")[0],
                                            sep='\t', header=None)
        self.test_df = pd.read_csv(glob.glob(regular_train_source + "/*TEST.tsv")[0],
                                   sep='\t', header=None)

        return

    def get_X_y(self, one_hot_encoding=True):
        """
        Return X_train, y_train, X_test, y_test (the train is always the regular train).
        X_train and X_test are pandas DataFrame;
        y_train and y_test contain the labels, and are either 1d (pandas Series) or multidimensional
        one-hot encoded arrays (if one_hot_encoding=True)
        :param one_hot_encoding: bool, if true, the label column is one-hot encoded
        :return: X_train, y_train, X_test, y_test
        """
        tr = self.regular_train_df
        te = self.test_df
        X_train = tr[tr.columns[1:]]
        y_train = tr[tr.columns[0]]
        X_test = te[te.columns[1:]]
        y_test = te[te.columns[0]]
        if one_hot_encoding:
            enc = OneHotEncoder(categories='auto')
            enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
            y_train = enc.transform(y_train.to_numpy().reshape(-1, 1)).toarray()
            y_test = enc.transform(y_test.to_numpy().reshape(-1, 1)).toarray()
        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    path = 'C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    data = DataLoader(path=path, data_name='insect')
    print(data.regular_train_df)
    # print(data.short_train_df)
    print(data.test_df)
