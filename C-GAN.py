import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.layers import BatchNormalization, Embedding
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from DataLoader import DataLoader


class CGAN():
    def __init__(self, input_shape, nb_classes, latent_dim=100, epochs=10000, mini_batch_size=15):
        # Input shape
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_shape = (self.img_rows, self.img_cols)
        self.num_classes = nb_classes
        self.latent_dim = latent_dim
        self.mini_batch_size = mini_batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs

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
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, X_train, y_train, sample_interval=100):

        print("classes: ", self.nb_classes)
        for i in range(self.nb_classes):
            class_i = np.where(y_train == i)[0][0]
            print(class_i)
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

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], self.mini_batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.mini_batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 2, self.mini_batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images()
        self.sample_images()

    def sample_images(self):
        noise = np.random.normal(0, 1, (self.nb_classes, 100))
        sampled_labels = np.arange(0, self.nb_classes).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        print(gen_imgs.shape)
        print(sampled_labels)
        for i in range(gen_imgs.shape[0]):
            label = sampled_labels[i]
            sns.lineplot(x=range(gen_imgs.shape[1]), y=np.ravel(gen_imgs[i, :, :]), label=f"gen {label}")
            plt.show()


if __name__ == '__main__':
    dt = DataLoader(path="C:/Users/letiz/Desktop/Bachelor\'s Thesis and Seminar - JOIN.bsc/data")
    X_train, y_train, _, _ = dt.get_X_y(one_hot_encoding=False)
    nb_classes = len(np.unique(y_train))

    # Scale input
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # X_train = X_train.T

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = X_train.shape[1:]
    print(X_train.shape)

    y_train = np.array([0 if x == 1 else 1 for x in y_train])
    y_train = y_train.reshape(-1, 1)
    print(y_train.shape)

    cgan = CGAN(input_shape=input_shape, nb_classes=nb_classes)
    cgan.train(X_train, y_train, sample_interval=1000)
