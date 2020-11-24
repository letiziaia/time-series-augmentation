import time

import numpy as np
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from DataLoader import DataLoader


class ClassifierDTW:
    def __init__(self, neighbors=1):
        self.n = neighbors
        return

    def fit_predict(self, X_train, y_train, X_test, y_test):
        start_time = time.time()
        print("Start fitting...")

        X = X_train.to_numpy(dtype=np.double)
        y_pred = []
        # for each test time series
        for i, ts in X_test.iterrows():
            ds = []
            # for each train time series
            for tts in X:
                # compute how far the test time series is from each train time series
                ds.append(dtw.distance_fast(ts.to_numpy(dtype=np.double), tts))
            # ds is an array of distances
            ds = np.array(ds)
            # sort in ascending order, get the indices of the closest n neighbors
            nearest_neighbors = ds.argsort()[:self.n]
            # collect the labels from the neighbors
            candidates = []
            for c in nearest_neighbors:
                candidates.append(y_train[c])
            # majority vote
            candidates = np.array(candidates)
            unique, unique_counts = np.unique(candidates, return_counts=True)
            label = [x for _, x in sorted(zip(unique_counts, unique))][-1]
            y_pred.append(label)
            print("y_pred: ", label)
            print("y_test: ", y_test[i])
            print("------")

        print("Total time:", time.time() - start_time)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(balanced_accuracy_score(y_test, y_pred))
        return y_pred


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'insect'
    dt = DataLoader(path="C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data", data_name=data_name)
    dt.describe()
    X_train, y_train, X_test, y_test = dt.get_X_y(one_hot_encoding=False)
    nb_classes = len(np.unique(y_train))

    clf = ClassifierDTW(neighbors=1)
    clf.fit_predict(X_train, y_train, X_test, y_test)
