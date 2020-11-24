import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization, Embedding
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from DataLoader import DataLoader


class CGAN():
    def __init__(self, input_shape, nb_classes, latent_dim=600, epochs=10000, mini_batch_size=15,
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
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
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
        for i in range(self.nb_classes):
            # get the first sample in class i from the original set
            class_i = np.where(y_train == i)[0][0]
            print("Plot original sample n. ", class_i)
            data = np.ravel(X_train[class_i, :, :])
            sns.lineplot(x=range(X_train.shape[1]), y=data, label=f"orig {i}")
            plt.show()

        # Adversarial ground truths
        valid = np.ones((self.mini_batch_size, 1))
        fake = np.zeros((self.mini_batch_size, 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.mini_batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.mini_batch_size, self.latent_dim))

            # Generate a batch of new samples
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
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

            # Every checkpoin_interval epochs, view the generated samples
            if epoch % checkpoint_interval == 0:
                self.sample_images()
        self.sample_images()
        ts, l = self._generate()
        return ts, l

    def sample_images(self):
        noise = np.random.normal(0, 1, (self.nb_classes, self.latent_dim))
        sampled_labels = np.arange(0, self.nb_classes).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        for i in range(gen_imgs.shape[0]):
            label = sampled_labels[i]
            sns.lineplot(x=range(gen_imgs.shape[1]), y=np.ravel(gen_imgs[i, :, :]), label=f"gen {label}")
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
    # data =  ['insect', 'shapes', 'freezer', 'beef', 'coffee', 'ecg200', 'gunpoint']
    data_name = 'insect'
    dt = DataLoader(path="C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data", data_name=data_name)
    dt.describe()
    samples_per_class = int(input("How many samples per class do you want to generate? "))
    X_train, y_train, _, _ = dt.get_X_y(one_hot_encoding=False)

    print("X_train shape: ", X_train.shape)
    nb_classes = len(np.unique(y_train))

    # Scale input
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # X_train = X_train.T

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = X_train.shape[1:]
    print("X_train shape (dimension added): ", X_train.shape)

    y_train = y_train.reshape(-1, 1)
    print("y_train shape: ", y_train.shape)

    cgan = CGAN(input_shape=input_shape, nb_classes=nb_classes,
                mini_batch_size=input_shape[0],
                latent_dim=input_shape[1], epochs=10000, n_synthetic_per_class=samples_per_class)
    ts, l = cgan.train_generate(X_train, y_train, checkpoint_interval=20000)

    print("\n")
    print("Shape of generated data: ", ts.shape)
    print("Shape of labels: ", l.shape)
    print("\n")

    df = pd.DataFrame(ts, columns=range(ts.shape[1]))
    df["l"] = l
    print(df.head())

    timestamp = time.strftime("%d.%m.%y_%H%M%S")
    df.to_csv("gen_data/cgan_" + data_name + "_" + timestamp + ".tsv", sep="\t")

    # ts = scaler.inverse_transform(ts)
    # for i in range(min(ts.shape[0], 3)):
    #     label = l[i]
    #     sns.lineplot(x=range(ts.shape[1]), y=np.ravel(ts[i, :]), label=f"gen {label}")
    #     plt.show()
