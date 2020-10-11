import glob

import pandas as pd


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


if __name__ == '__main__':
    path = 'C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    data = DataLoader(path=path, data_name='insect')
    print(data.regular_train_df)
    print(data.short_train_df)
    print(data.test_df)
