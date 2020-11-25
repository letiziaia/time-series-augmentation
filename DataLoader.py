import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class DataLoader:
    """Load one of the selected datasets (Beef, Freezer, InsectEPG, MixedShapes, etc) from UCR time series archive.
    It can also load the synthetic datasets, if they have been saved to tsv file already."""

    def __init__(self, path, data_name='freezer', cgan=False):
        """
        Load the regular train set, small train set, and test set in the object.
        :param path: str, the path to the data folder
        :param data_name: str, a shorter name identifying the data set to load
        :param cgan: bool, if True, load the synthetic data from the tsv file
        """
        self.cgan = cgan
        data_name = data_name.lower()
        if self.cgan:
            path = "gen_data/"
            if 'insect' in data_name:
                # Load InsectEPG
                print("Loading the CGAN generated InsectEPG data set...")
                file_name = path + "cgan_insect"

            elif 'shapes' in data_name:
                # Load MixedShapes
                print("Loading the CGAN generated MixedShapes data set...")
                file_name = path + "cgan_shapes"

            elif 'freezer' in data_name:
                # Load freezer
                print("Loading the CGAN generated Freezer data set...")
                file_name = path + "cgan_freezer"

            elif 'beef' in data_name:
                # Load beef
                print("Loading the CGAN generated Beef data set...")
                file_name = path + "cgan_beef"

            elif 'coffee' in data_name:
                # Load coffee
                print("Loading the CGAN generated Coffee data set...")
                file_name = path + "cgan_coffee"

            elif 'ecg200' in data_name:
                # Load ECG200
                print("Loading the CGAN generated ECG200 data set...")
                file_name = path + "cgan_ecg200"

            elif 'gunpoint' in data_name:
                # Load gunpoint
                print("Loading the CGAN generated GunPoint data set...")
                file_name = path + "cgan_gunpoint"

            else:
                pass  # unsafe

            self.short_train_df = None
            self.regular_train_df = pd.read_csv(glob.glob(file_name + "*.tsv")[0], sep='\t', header=0, index_col=0)
            self.test_df = None
        else:
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

            elif 'freezer' in data_name:
                # Load Freezer
                print("Loading the Freezer data set...")
                regular_train_source = path + "/FreezerRegularTrain"
                small_train_source = path + "/FreezerSmallTrain"

            elif 'beef' in data_name:
                # Load Beef
                print("Loading the Beef data set...")
                regular_train_source = path + "/Beef"
                small_train_source = path + "/Beef"

            elif 'coffee' in data_name:
                # Load Coffee
                print("Loading the Coffee data set...")
                regular_train_source = path + "/Coffee"
                small_train_source = path + "/Coffee"

            elif 'ecg200' in data_name:
                # Load ECG200
                print("Loading the ECG200 data set...")
                regular_train_source = path + "/ECG200"
                small_train_source = path + "/ECG200"

            elif 'gunpoint' in data_name:
                # Load Gunpoint
                print("Loading the GunPoint data set...")
                regular_train_source = path + "/GunPoint"
                small_train_source = path + "/GunPoint"

            else:
                pass  # unsafe

            self.short_train_df = pd.read_csv(glob.glob(small_train_source + "/*TRAIN.tsv")[0],
                                              sep='\t', header=None)
            self.regular_train_df = pd.read_csv(glob.glob(regular_train_source + "/*TRAIN.tsv")[0],
                                                sep='\t', header=None)
            self.test_df = pd.read_csv(glob.glob(regular_train_source + "/*TEST.tsv")[0],
                                       sep='\t', header=None)

        return

    def describe(self):
        X_train, y_train, X_test, y_test = self.get_X_y(one_hot_encoding=False)
        train_size = X_train.shape[0]
        ts_len = X_train.shape[1]
        test_size = "None" if X_test is None else X_test.shape[0]
        classes, counts = np.unique(y_train, return_counts=True)
        print("\n")
        print(f"The data contains {len(classes)} classes of time series of length {ts_len}.")
        print(f"The training set contains {train_size} samples and the test set has {test_size} samples.")
        for i in range(len(classes)):
            print(f"Class {classes[i]} contains {counts[i]} samples.")
        print("\n")
        return

    def get_X_y(self, one_hot_encoding=True):
        """
        Return X_train, y_train, X_test, y_test (the train is always the regular train).
        X_train and X_test are pandas DataFrame;
        y_train and y_test contain the labels, and are either 1d np.array with values 0 to num_classes-1
        or multidimensional one-hot encoded arrays (if one_hot_encoding=True)
        :param one_hot_encoding: bool, if true, the label column is one-hot encoded
        :return: X_train, y_train, X_test, y_test
        """
        tr = self.regular_train_df
        te = self.test_df
        if self.cgan:
            X_train = tr[tr.columns[:-1]]
            y_train = tr["l"].to_numpy()
            X_test = None
            y_test = None
            if one_hot_encoding:
                enc = OneHotEncoder(categories='auto')
                enc.fit(y_train.reshape(-1, 1))
                y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        else:
            X_train = tr[tr.columns[1:]]
            y_train = tr[tr.columns[0]]
            X_test = te[te.columns[1:]]
            y_test = te[te.columns[0]]
            if one_hot_encoding:
                enc = OneHotEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                y_train = enc.transform(y_train.to_numpy().reshape(-1, 1)).toarray()
                y_test = enc.transform(y_test.to_numpy().reshape(-1, 1)).toarray()
            else:
                enc = OrdinalEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                y_train = np.ravel(enc.transform(y_train.to_numpy().reshape(-1, 1)))
                y_test = np.ravel(enc.transform(y_test.to_numpy().reshape(-1, 1)))
        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'insect'
    path = 'C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    data = DataLoader(path=path, data_name=data_name, cgan=True)
    data.describe()
    print(data.regular_train_df)
    # print(data.short_train_df)
    print(data.test_df)
