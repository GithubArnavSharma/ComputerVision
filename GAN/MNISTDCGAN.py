import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.datasets.mnist import load_data
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, LeakyReLU
from keras.optimizers import Adam
from keras import backend


#Load the data, convert it to -1 - 1, and batch it with 256 batch size
(data_arr, _), (_, _) = load_data()
data_arr = np.array(data_arr).reshape(data_arr.shape[0], 28, 28, 1).astype('float32')
data_arr = (data_arr - 127.5) / 127.5
print(len(data_arr))
BATCH_SIZE = 256
data = tf.data.Dataset.from_tensor_slices(data_arr).shuffle(len(data_arr)).batch(BATCH_SIZE)

#Discriminator model, which takes an image and outputs how real that image is
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

#Generation model, which takes a 100 dimensional vector and transforms it to a 28*28*1 image
def generator_model():
    model = Sequential()
    model.add(Dense(7*7*256, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7,7,256)))
    model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model

discriminator = discriminator_model()
generator = generator_model()

#Train the discriminator and generator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(fake_output):
    return cross_entropy(np.ones_like(fake_output), fake_output)

def disc_loss(real_output, fake_output):
    real_loss = cross_entropy(np.ones_like(real_output), real_output)
    fake_loss = cross_entropy(np.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

disc_opt = Adam(1e-4)
gen_opt = Adam(1e-4)

def train_step(images):
    noise = np.random.rand(BATCH_SIZE, 100)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated = generator(noise, training=True)

        pred_fake = discriminator(generated, training=True)
        pred_real = discriminator(images, training=True)
        d_loss = disc_loss(pred_real, pred_fake)

        g_loss = gen_loss(pred_fake)

    dis_gradients = dis_tape.gradient(d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

    gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))


def train(epochs):
    for i in range(1, epochs+1):
        print(f"Epoch: {i}")
        for image_batch in data:
            train_step(image_batch)

def generate_images(side_num):
    fig, axes = plt.subplots(nrows=side_num, ncols=side_num)
    for i in range(side_num):
        for j in range(side_num):
            image_pred = generator(np.random.rand(1,100), training=False)[0,:,:,0]
            axes[i][j].imshow(image_pred, cmap='gray')
    fig.tight_layout()
    plt.show()


train(50)
generate_images(4)
