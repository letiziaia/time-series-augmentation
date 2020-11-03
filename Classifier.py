import numpy as np
import pandas as pd
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from Augmenter import Augmenter
from DataLoader import DataLoader


class DTW_based:
    def __init__(self, neighbors=1):
        self.n = neighbors
        dt = DataLoader(path="C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data")
        tr = dt.regular_train_df
        te = dt.test_df
        self.X_train = tr[tr.columns[1:]]
        self.y_train = tr[tr.columns[0]]
        self.X_test = te[te.columns[1:]]
        self.y_test = te[te.columns[0]]

    def __augment(self, fraction):
        print(self.X_train.shape)
        n = int(self.X_train.shape[0] * fraction) + 1
        print(f"generating {n} time series")
        generated = []
        glab = []
        for i in range(n):
            a = Augmenter(data=self.X_train, labels=self.y_train)
            xx, yx, _ = a.jittering()
            generated.append(xx)
            glab.append(yx)
        print(self.X_train.tail())
        print(self.y_train.tail())
        self.X_train = self.X_train.append(pd.DataFrame(generated))
        self.y_train = self.y_train.append(pd.Series(glab))
        print(self.X_train.tail())
        print(self.y_train.tail())
        print(self.X_train.shape)
        return

    def dtw_1_nn(self, augment_by=0, method="jittering"):
        if augment_by > 0.0:
            self.__augment(fraction=augment_by)
        last_idx = self.X_train.shape[0]
        X = self.X_train.to_numpy(dtype=np.double)
        y_pred = []
        for i, ts in self.X_test.iterrows():
            ds = []
            for tts in X:
                ds.append(dtw.distance_fast(ts.to_numpy(dtype=np.double), tts))
            ds = np.array(ds)
            idx = np.argmin(ds)
            y_pred.append(self.y_train.iloc[idx])
            print(self.y_train.iloc[idx])
            print(self.y_test.iloc[i])
            print("------")
        print(classification_report(self.y_test.values, y_pred))
        print(confusion_matrix(self.y_test.values, y_pred))
        print(balanced_accuracy_score(self.y_test.values, y_pred))
        return


if __name__ == '__main__':
    clf = DTW_based(neighbors=0.05)
    clf.dtw_1_nn(augment_by=0.1)
