import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization, Embedding, Conv1D, Bidirectional, LSTM, Conv1DTranspose
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler

from DataLoader import DataLoader


class CGAN():
    def __init__(self, input_shape, nb_classes, latent_dim=50, epochs=10000, mini_batch_size=15,
                 n_synthetic_per_class=10):
        # Input shape
        self.input_rows = input_shape[0]
        self.input_cols = input_shape[1]
        self.input_shape = (self.input_rows, self.input_cols)
        self.num_classes = nb_classes
        self.latent_dim = latent_dim
        self.mini_batch_size = mini_batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.n_synthetic = n_synthetic_per_class

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding time series
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        time_series = self.generator([noise, label])

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated time series as input
        # and determines validity and the label of that time series
        valid = self.discriminator([time_series, label])

        # The combined model (stacked generator and discriminator)
        # Trains generator to fool discriminator
        # self.combined = self.combined()
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)
                              # optimizer=RMSprop(lr=0.0001, decay=3e-8)),
                              # metrics=['accuracy'])
        self.discriminator.trainable = True

    def combined(self):
        model = Sequential()
        model.add(self.generator)
        self.discriminator.trainable = False
        model.add(self.discriminator)
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.0001, decay=3e-8),
                      metrics=['accuracy'])
        self.discriminator.trainable = True

    def build_generator(self):

        model = Sequential()

        model.add(Dense(int(self.latent_dim * 0.852), input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(Reshape((1, -1)))
        # model.add(Conv1D(filters=12, kernel_size=3, padding="same"))
        model.add(Conv1DTranspose(filters=2, kernel_size=3, padding="same"))
        model.add(Dense(1024))
        model.add(Reshape((4, 256)))
        model.add(Bidirectional(LSTM(60)))
        # model.add(Flatten())
        model.add(Dense(2048))
        model.add(LeakyReLU(alpha=0.1))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalization(momentum=0.2, renorm=True, renorm_clipping={'rmax': 0.25, 'dmax': 0.2}))
        model.add(Dense(np.prod(self.input_shape)))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.input_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.input_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.input_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train_generate(self, X_train, y_train, checkpoint_interval=1000):

        print("classes: ", self.nb_classes)
        X_train_representative = []
        representative_labels = []
        for i in range(self.nb_classes):
            # get the first sample in class i from the original set
            class_i = np.where(y_train == i)[0][0]
            print("Plot original sample n. ", class_i)
            data = np.ravel(X_train[class_i, :, :])
            X_train_representative.append(data)
            representative_labels.append(i)
            sns.lineplot(x=range(X_train.shape[1]), y=data, label=f"orig {i}")
            plt.show()

        # Adversarial ground truths
        valid = np.ones((self.mini_batch_size, 1))
        fake = np.zeros((self.mini_batch_size, 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of original time series
            idx = np.random.randint(0, X_train.shape[0], self.mini_batch_size)
            ts, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.mini_batch_size, self.latent_dim))

            # Generate a batch of new samples
            gen_ts = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([ts, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_ts, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.nb_classes, self.mini_batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # Every checkpoint_interval epochs, view the generated samples
            if epoch % checkpoint_interval == 0:
                self.sample_images(X_train_representative, representative_labels)

        self.sample_images(X_train_representative, representative_labels)
        ts, l = self._generate()
        return ts, l

    def sample_images(self, X_repr, y_repr):
        noise = np.random.normal(0, 1, (self.nb_classes, self.latent_dim))
        sampled_labels = np.arange(0, self.nb_classes).reshape(-1, 1)

        gen_ts = self.generator.predict([noise, sampled_labels])

        for i in range(gen_ts.shape[0]):
            label = sampled_labels[i]
            idx = np.where(y_repr == label[0])[0][0]
            X = X_repr[idx]
            sns.lineplot(x=range(X.shape[0]), y=X, label=f"orig {label}")
            sns.lineplot(x=range(gen_ts.shape[1]), y=np.ravel(gen_ts[i, :, :]), label=f"gen {label}")
            plt.show()

    def _generate(self):
        new_samples = []
        new_labels = []
        # Each generation steps creates one sample per class
        for i in range(self.n_synthetic):
            noise = np.random.normal(0, 1, (self.nb_classes, self.latent_dim))
            sampled_labels = np.arange(0, self.nb_classes).reshape(-1, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])
            for i in range(gen_imgs.shape[0]):
                new_ts = np.ravel(gen_imgs[i, :, :])
                new_samples.append(new_ts)
                label = sampled_labels[i]
                new_labels.append(label)
        return np.array(new_samples), np.array(new_labels)


if __name__ == '__main__':
    path = "C:/Users/letiz/Desktop/Aalto/Bachelor\'s Thesis and Seminar - JOIN.bsc/data"
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'gunpoint'
    dt = DataLoader(path=path, data_name=data_name)
    dt.describe()
    samples_per_class = int(input("How many samples per class do you want to generate? "))
    X_train, y_train, _, _ = dt.get_X_y(one_hot_encoding=False)

    print("X_train shape: ", X_train.shape)
    nb_classes = len(np.unique(y_train))

    # Scale input
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = X_train.shape[1:]
    print("X_train shape (dimension added): ", X_train.shape)

    y_train = y_train.reshape(-1, 1)
    print("y_train shape: ", y_train.shape)


    ddf = pd.DataFrame()
    for i in range(samples_per_class):
        # Retrain everytime: safer in case of mode collapse
        cgan = CGAN(input_shape=input_shape, nb_classes=nb_classes,
                    mini_batch_size=input_shape[0],
                    latent_dim=input_shape[1],
                    epochs=500, n_synthetic_per_class=1)
        ts, l = cgan.train_generate(X_train, y_train, checkpoint_interval=500)

        print("\n")
        print("Shape of generated data: ", ts.shape)
        print("Shape of labels: ", l.shape)
        print("\n")

        df = pd.DataFrame(ts, columns=range(ts.shape[1]))
        ddf = ddf.append(df)

    new_df = []
    for i, row in ddf.iterrows():
        # row = row.rolling(window=7, center=True).median()
        row = row.rolling(window=7, center=True).mean()
        row.bfill(inplace=True)
        row.ffill(inplace=True)
        new_df.append(row)
        if i == 0:
            sns.lineplot(x=range(X_train.shape[1]), y=np.ravel(X_train[-1]), label=f"orig")
            sns.lineplot(x=range(len(row)), y=row, label=f"gen")
            plt.show()

    df = pd.DataFrame(new_df, columns=range(ts.shape[1]))
    df["l"] = l
    print(df.head())

    timestamp = time.strftime("%d.%m.%y_%H%M%S")
    df.to_csv("gen_data/cgan_" + data_name + "_" + timestamp + ".tsv", sep="\t")

    # ts = scaler.inverse_transform(ts)
    # for i in range(min(ts.shape[0], 3)):
    #     label = l[i]
    #     sns.lineplot(x=range(ts.shape[1]), y=np.ravel(ts[i, :]), label=f"gen {label}")
    #     plt.show()
