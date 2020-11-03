import matplotlib.pyplot as plt
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
        x = self.data.iloc[idx].values
        y = self.labels.iloc[idx]
        sns.lineplot(x=range(len(x)), y=x, label=f"ts {idx}, class {y}")
        plt.show()
        return x, y, idx

    def jittering(self, mu=0.0, sigma=0.01):
        """
        Produce a new time series by adding normally distributed random noise
        :param mu: float, the mean of the random noise distribution
        :param sigma: float, the standard deviation of the random noise distribution
        :return: a new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        return x * (1 + np.random.normal(loc=mu, scale=sigma, size=len(x))), y, idx

    def flipping(self):
        """
        Produce a new time series by inverting the sign
        :return: the new time series, its label, the index of the seed time series
        """
        x, y, idx = self.__random_selection()
        return -x, y, idx

    def smote(self):
        return

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
    from DataLoader import DataLoader

    path = 'C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data'
    dt = DataLoader(path=path, data_name='insect')
    df = dt.regular_train_df
    print(df)
    a = Augmenter(data=df[df.columns[1:]], labels=df[df.columns[0]])

    # xx, _, _ = a.jittering(mu=0, sigma=0.001)
    # print(xx)
    # sns.lineplot(x=range(len(xx)), y=xx, label="jittered")
    # plt.show()
    #
    # xx, _, _ = a.flipping()
    # print(xx)
    # sns.lineplot(x=range(len(xx)), y=xx, label="flipped")
    # plt.show()

    xx, _, _ = a.permutation(n_segments=7)
    print(xx)
    sns.lineplot(x=range(len(xx)), y=xx, label="permuted")
    plt.show()

    xx, _, _ = a.window_slicing(d=400)
    print(xx)
    sns.lineplot(x=range(len(xx)), y=xx, label="sliced")
    plt.show()
