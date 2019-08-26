from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from data_loader import DataLoader
import matplotlib.pyplot as plt
import glob
import os
from scipy.misc import imsave
import numpy as np
from PIL import Image


class DCGAN:
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.kernel_size = 3
        self.upsample_layers = 5


        optimizer = Adam(0.0002, 0.5)

        # 建立鑑別器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 建立一個產生器
        self.generator = self.build_generator()
        
        # 將雜訊輸入到產生器並產生圖片
        noise =  Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # 我們將只訓練組合模型的產生器
        self.discriminator.trainable = False

        # 將產生的圖片輸入判斷並確定為有效性的
        valid = self.discriminator(img)

        # 組合模型 (堆疊產生器和鑑別器)
        # 訓練產生器跟鑑別器
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)


    def build_generator(self):
        noise_shape = (100,)
        model = Sequential()
        model.add(
            Dense(64 * (self.img_shape[0] // (2 ** self.upsample_layers))  *  (self.img_shape[1] // (2 ** self.upsample_layers)),
                  activation="relu", input_shape=noise_shape))
        model.add(Reshape(((self.img_shape[0] // (2 ** self.upsample_layers)),
                           (self.img_shape[1] // (2 ** self.upsample_layers)),
                           64)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 6x8 -> 12x16
        model.add(Conv2D(1024, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 12x16 -> 24x32
        model.add(Conv2D(512, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 24x32 -> 48x64
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 48x64 -> 96x128
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 96x128 -> 192x256
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))  # 12x16 -> 6x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_imgs(self, image_path):
        X_train = []
        for i in glob.glob(image_path):
            img = Image.open(i)
            img = np.asarray(img)
            X_train.append(img)
        return np.asarray(X_train)
    
    def load_dataset(self, dataset_path, batch_size, image_shape):
        dataset_generator = ImageDataGenerator()
        dataset_generator = dataset_generator.flow_from_directory(
            directory = dataset_path, 
            classes=['resize'],
            target_size=(image_shape[0], image_shape[1]),
            batch_size=batch_size,
            class_mode=None)

        return dataset_generator

    def train(self, epochs, batch_size=1, save_interval=50):
        IMG_SHAPE = (128, 128, 3)
        dataset_path = "/home/is90057/Documents/train_data/imgs/asiayo/"
        dataset_generator = self.load_dataset(dataset_path, batch_size, IMG_SHAPE)
        
        # 19370 is the total number of images on the bird dataset
        number_of_batches = int(11934 / batch_size)
        

        # 對抗的真理
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for batch_number in range(number_of_batches):
                X_train = dataset_generator.next()
                X_train /= 127.5
                X_train -= 1
                # ---------------------
                #  訓練 鑑別器
                # ---------------------

                # 隨機選擇一半的圖片
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
            

                # 採樣噪音並生成一批新的圖像
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # 訓練鑑別器
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  訓練 產生器
                # ---------------------

                # 訓練產生器
                g_loss = self.combined.train_on_batch(noise, valid)

                
                # 繪製進度
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # 保存生存的圖像樣本
            # if epoch % 11934 == 0:
            self.save_model()
            self.save_imgs(epoch)


        
    def gene_imgs(self, count):
        # Generate images from the currently loaded model
        noise = np.random.normal(0, 1, (count, 100))
        return self.generator.predict(noise)

    def save_imgs(self, epoch):
        r, c = 5, 5

        # Generates r*c images from the model, saves them individually and as a gallery

        imgs = self.gene_imgs(r*c)
        imgs = 0.5 * imgs + 0.5

        # for i, img_array in enumerate(imgs):
        #     path = "images"
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     imsave(path + f"/{epoch}_{i}.png", img_array)

        nindex, height, width, intensity = imgs.shape
        nrows = nindex // c
        assert nindex == nrows * c
        # want result.shape = (height*nrows, width*ncols, intensity)
        gallery = (imgs.reshape(nrows, c, height, width, intensity)
                  .swapaxes(1, 2)
                  .reshape(height * nrows, width * c, intensity))

        path = "images"
        if not os.path.exists(path):
            os.makedirs(path)
        imsave(path + f"/{epoch}.jpg", gallery)

    def save_model(self):
        
        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")



if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
