import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from Augmenter import Augmenter
from DataLoader import DataLoader
from cnn_classifier import ClassifierCNN


def main():
    # unbalanced data =  ['insect', 'ecg200', 'gunpoint']
    data_name = 'gunpoint'
    path = 'C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    data = DataLoader(path=path, data_name=data_name, cgan=False)
    X_train, y_train, _, _ = data.get_X_y(one_hot_encoding=False)

    # minority class
    classes, counts = np.unique(y_train, return_counts=True)
    print("Classes: ", classes)
    print("Counts: ", counts)
    minority = [(x, y) for y, x in sorted(zip(counts, classes))][0]
    print("Minority class: ", minority[0])
    print("Minority samples: ", minority[1])
    majority = [(x, y) for y, x in sorted(zip(counts, classes))][-1]
    print("Majority class: ", majority[0])
    print("Majority samples: ", majority[1])

    fake = []
    fake_y = []
    if len(np.unique(counts)) == 1:
        print("This dataset is balanced")
        print("Set the number of fake samples per class you want to generate: ")
        n = int(input())
        if n > 0:
            for c in range(len(classes)):
                label = classes[c]
                print(f"Class {label} will get {n} more samples.")
                take_idx = np.where(y_train == label)[0]
                aug = Augmenter(data=X_train.to_numpy()[take_idx], labels=y_train[take_idx])
                for i in range(n):
                    # print("Jittering")
                    # x, y, idx = aug.jittering(mu=0, sigma=0.001)
                    # print("Flipping")
                    # x, y, idx = aug.flipping()
                    # print("Permutation")
                    # x, y, idx = aug.permutation(n_segments=7)
                    print(f"{i + 1} artificial samples from class {label} done. The seed was {idx}")
                    fake.append(x)
                    fake_y.append(y)

    for c in range(len(classes)):
        samples_needed = majority[1] - counts[c]
        label = classes[c]
        print(f"Class {label} needs {samples_needed} more samples.")

        if samples_needed > 0:
            # isolate the samples from the class
            take_idx = np.where(y_train == label)[0]
            aug = Augmenter(data=X_train.to_numpy()[take_idx], labels=y_train[take_idx])
            for i in range(samples_needed):
                # print("Jittering")
                # x, y, idx = aug.jittering(mu=0, sigma=0.001)
                # print("Flipping")
                # x, y, idx = aug.flipping()
                # print("Permutation")
                # x, y, idx = aug.permutation(n_segments=7)
                print(f"{i + 1} artificial samples from class {label} done. The seed was {idx}")
                fake.append(x)
                fake_y.append(y)

    fake_X = pd.DataFrame(fake)
    fake_y = np.array(fake_y)

    # AUGMENTED
    print("--------------------------------------------------------------")
    print("--- AUGMENTED DATA SET ---------------------------------------")
    print("--------------------------------------------------------------")
    X_train, y_train, X_test, y_test = data.get_X_y(one_hot_encoding=False)
    nb_classes = len(np.unique(y_train))

    X_train = np.concatenate((X_train, fake_X))
    y_train = np.concatenate((y_train, fake_y))

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    clf = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes, epochs=2000)
    clf.fit_predict(X_train, y_train, X_test, y_test)

    # ORIGINAL
    print("--------------------------------------------------------------")
    print("--- ORIGINAL DATA SET ---------------------------------------")
    print("--------------------------------------------------------------")
    X_train, y_train, X_test, y_test = data.get_X_y(one_hot_encoding=False)
    nb_classes = len(np.unique(y_train))

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, fake_y, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    clf = ClassifierCNN(input_shape=input_shape, nb_classes=nb_classes, epochs=2000)
    clf.fit_predict(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
