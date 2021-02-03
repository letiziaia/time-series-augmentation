import time

import numpy as np
import pandas as pd

from Augmenter import Augmenter
from DataLoader import DataLoader


def main(data_name, method):
    path = 'C:/Users/letiz/Desktop/Aalto/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    dt = DataLoader(path=path, data_name=data_name)
    X_train, y_train, _, _ = dt.get_X_y(one_hot_encoding=False)

    # Classes counts
    classes, counts = np.unique(y_train, return_counts=True)
    print("Classes: ", classes)
    print("Counts: ", counts)
    minority = [x for _, x in sorted(zip(counts, classes))][0]
    print("Is it balanced? ", len(np.unique(counts)) == 1)

    new_ts = []
    new_l = []
    if len(np.unique(counts)) == 1:
        for c in classes:
            for i in range(np.unique(counts)[0]):
                print("sample ", i)
                idx = np.where(y_train == c)[0]
                aug = Augmenter(data=X_train.to_numpy()[idx], labels=y_train[idx])
                if method == 'jitt':
                    xx, l, _ = aug.jittering(mu=0, sigma=0.001)
                elif method == 'flip':
                    xx, l, _ = aug.flipping()
                elif method == 'perm':
                    xx, l, _ = aug.permutation(n_segments=7)
                elif method == 'smote':
                    xx, l, _ = aug.smote_oversampling()
                else:
                    print("Method not valid")
                    return
                new_ts.append(xx)
                new_l.append(l)
    else:
        print("Minority class: ", minority)
        majority = [(x, y) for y, x in sorted(zip(counts, classes))][-1]
        print("Samples in majority: ", majority[1])
        print(counts)
        print(classes)
        for c in classes:
            samples_needed = majority[1] - counts[np.where(classes == c)][0]
            print(f"Class {c} needs {samples_needed} more samples.")
            for i in range(majority[1] + samples_needed):
                print("sample ", i)
                idx = np.where(y_train == c)[0]
                aug = Augmenter(data=X_train.to_numpy()[idx], labels=y_train[idx])
                if method == 'jitt':
                    xx, l, _ = aug.jittering(mu=0, sigma=0.001)
                elif method == 'flip':
                    xx, l, _ = aug.flipping()
                elif method == 'perm':
                    xx, l, _ = aug.permutation(n_segments=7)
                elif method == 'smote':
                    xx, l, _ = aug.smote_oversampling()
                else:
                    print("Method not valid")
                    return
                new_ts.append(xx)
                new_l.append(l)

    new_ts = np.array(new_ts)
    df = pd.DataFrame(new_ts, columns=range(new_ts.shape[1]))
    df["l"] = new_l
    timestamp = time.strftime("%d.%m.%y_%H%M%S")
    df.to_csv("gen_data/aug_" + method + "_" + data_name + "_" + timestamp + ".tsv", sep="\t")
    print("saved")


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'ecg200'
    # method = ['jitt', 'flip', 'perm', 'smote']
    method = 'jitt'
    main(data_name=data_name, method=method)
