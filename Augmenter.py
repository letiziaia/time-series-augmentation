import numpy as np
import seaborn as sns


class Augmenter:
    def __init__(self, data, labels):
        """
        Initialize the Augmenter object with the training data and their labels
        :param data: pandas dataframe, each row is one time series
        :param labels: pandas series or dataframe
        """
        self.data = data
        self.labels = labels

    def __random_selection(self):
        """
        Helper function, randomly selects one seed time series
        :return: seed time series, label, index
        """
        idx = np.random.randint(0, self.data.shape[0])
        print(idx)
        x = self.data[idx]
        y = self.labels[idx]
        # sns.lineplot(x=range(len(x)), y=x, label=f"ts {idx}, class {y}")
        # plt.show()
        return x, y, idx

    def jittering(self, mu=0.0, sigma=0.01, additive=True):
        """
        Produce a new time series by adding normally distributed random noise
        :param mu: float, the mean of the random noise distribution
        :param sigma: float, the standard deviation of the random noise distribution
        :param additive: bool, the noise is added as T+epsilon when true;
                        if false, noise is multiplicative: T*(1+epsilon)
        :return: a new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        if additive:
            return x + np.random.normal(loc=mu, scale=sigma, size=len(x)), y, idx
        else:
            return x * (1 + np.random.normal(loc=mu, scale=sigma, size=len(x))), y, idx

    def flipping(self):
        """
        Produce a new time series by inverting the sign
        :return: the new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        return -x, y, idx

    def smote_oversampling(self):
        """
        Produce a new time series as element-wise average of two randomly chosen time series
        :return: the new time series, its label, the indices of the seed time series
        """
        x1, y1, idx1 = self.__random_selection()
        x2, y2, idx2 = self.__random_selection()
        return (x1 + x2) / 2, y1, (idx1, idx2)

    def permutation(self, n_segments=2):
        """

        :param n_segments: int, the seed time series is splitted into n_segments parts, which are then shuffled
        before recombining them
        :return: the new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        assert 0 < n_segments < len(x)
        # Randomly pick n_segments-1 points where to slice
        idxs = np.random.randint(0, self.data.shape[0], size=n_segments - 1)
        # print(idxs)
        slices = []
        start_idx = 0
        for i in sorted(idxs):
            s = x[start_idx:i]
            start_idx = i
            slices.append(s)
        slices.append(x[start_idx:])
        # print(len(slices))
        np.random.shuffle(slices)
        # print("Finally", slices)
        return np.ravel(np.concatenate(slices)), y, idx

    def window_slicing(self, d):
        """

        :param d: int, length of the slice. After selecting a random point in the seed time series, a slice of size d
        is produced ... [to fix]
        :return: the new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        assert d < len(x)
        # Randomly pick 1 point where to slice
        i = np.random.randint(0, self.data.shape[0])
        if i + d <= len(x):
            sliced_x = x[i:i + d]
        else:
            sliced_x = x[i - d:i]
        return sliced_x, y, idx

    def window_warping(self, factor=1):
        return


if __name__ == '__main__':
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'gunpoint'
    sns.set_theme(style="darkgrid")

    # aug = Augmenter(data=X_train.to_numpy(), labels=y_train)
    # xx, _, i = aug.jittering(mu=0.02, sigma=0.02)
    # print(xx)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    # plt.suptitle("Data set: " + data_name + ", data point " + str(i))
    # sns.lineplot(x=range(len(xx)), y=X_train.to_numpy()[i], label='Original', ax=axes[0])
    # sns.lineplot(x=range(len(xx)), y=xx, label="Jittered", color="red", ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    # xx, _, i = aug.flipping()
    # print(xx)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    # plt.suptitle("Data set: " + data_name + ", data point " + str(i))
    # sns.lineplot(x=range(len(xx)), y=X_train.to_numpy()[i], label='Original', ax=axes[0])
    # sns.lineplot(x=range(len(xx)), y=xx, label="Flipped", color="red", ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    # xx, _, i = aug.permutation(n_segments=7)
    # print(xx)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    # plt.suptitle("Data set: " + data_name + ", data point " + str(i))
    # sns.lineplot(x=range(len(xx)), y=X_train.to_numpy()[i], label='Original', ax=axes[0])
    # sns.lineplot(x=range(len(xx)), y=xx, label="Permuted", color="red", ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    # xx, _, i = aug.window_slicing(d=400)
    # print(xx)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    # plt.suptitle("Data set: " + data_name + ", data point " + str(i))
    # sns.lineplot(x=range(len(X_train.to_numpy()[i])), y=X_train.to_numpy()[i], label='Original', ax=axes[0])
    # sns.lineplot(x=range(len(xx)), y=xx, label="Sliced", color="red", ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    # REMEMBER TO PASS ONLY ONE CLASS AT A TIME FOR THIS ONE
    # xx, _, (i,j) = aug.smote_oversampling()
    # print(xx)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    # plt.suptitle("Data set: " + data_name + ", data points " + str((i, j)))
    # sns.lineplot(x=range(len(X_train.to_numpy()[i])), y=X_train.to_numpy()[i], label='Original 1', ax=axes[0])
    # sns.lineplot(x=range(len(X_train.to_numpy()[j])), y=X_train.to_numpy()[j], label='Original 2', ax=axes[0])
    # sns.lineplot(x=range(len(xx)), y=xx, label="AVG_TS_SMOTE", color="red", ax=axes[1])
    # plt.tight_layout()
    # plt.show()
