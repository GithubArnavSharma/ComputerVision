import matplotlib.pyplot as plt
import numpy as np
import time
import random
import tensorflow as tf
from keras.datasets.mnist import load_data
from keras import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = load_data()
X = list(X_train) + list(X_test)
y = list(y_train) + list(y_test)
X = np.array(X).reshape(len(X), 28, 28, 1).astype('float32')
X = X / 255.0
BATCH_SIZE = 128
data = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(BATCH_SIZE)
LEN_BATCH = len(data)

def discriminator_model():
  model = Sequential()
  model.add(Conv2D(64, kernel_size=5, padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2D(128, kernel_size=5, padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2D(128, kernel_size=5, padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2D(256, kernel_size=5, padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))

  return model

def generator_model():
  model = Sequential()
  model.add(Dense(7*7*128, input_shape=(100,)))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Reshape((7,7,128)))

  model.add(Conv2DTranspose(256, (4,4), strides=(1,1), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(128, (4,4), strides=(1,1), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(128, (4,4), strides=(1,1), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='sigmoid'))

  assert model.output_shape == (None, 28, 28, 1)
  return model

discriminator = discriminator_model()
generator = generator_model()

#The loss here will be Binary Crossentropy
cross_entropy = tf.keras.losses.BinaryCrossentropy()

#The generator will try to maximize what the discriminator predicts for it, and therefore will be a comparison between 1's and the discriminators guess
def gen_loss(fake_output):
    return cross_entropy(np.ones_like(fake_output), fake_output)

#The discriminator will try its best to get accurate in its distinction between real and fake images, so therefore the loss will be the magnitude of what it got wrong
def disc_loss(real_output, fake_output):
    real_loss = cross_entropy(np.ones_like(real_output), real_output)
    fake_loss = cross_entropy(np.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss)/2

#The optimizers used will be Adam, with the learning rate as 0.0002, as a large learning rate on a complex model would be inefficient
disc_opt = Adam(lr=0.0002, beta_1=0.5)
gen_opt = Adam(lr=0.0002, beta_1=0.5)

#Function that can train on a single batch of images
#This GAN will learn both the generator and discriminator at the same time, so there's enough loss in the discriminator for the generator to improve,
#but not enough loss in the discriminator for the generator to make mistakes and get rewarded by the discriminator for it
def train_step(images, counters):
    #Create a 100 dimensional noise with the batch size to make the generator generaye
    noise = np.random.rand(BATCH_SIZE, 100)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        #Generate the images with the generator
        generated = generator(noise, training=True)

        #Get the discriminators input on the fake and real images, and calculate its loss 
        pred_fake = discriminator(generated, training=True)
        pred_real = discriminator(images, training=True)
        d_loss = disc_loss(pred_real, pred_fake)


        #Calculate the generator loss with the discriminator's output for its fake images
        g_loss = gen_loss(pred_fake)

    #Calculate the gradients for the discriminator's trainable weights and biases, and apply those gradients to the discriminator optimizer
    dis_gradients = dis_tape.gradient(d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

    #Calculate the gradients for the generator's trainable weights and biases, and apply those gradients to the generator optimizer
    gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))

#Function that, when given a certain amount of epochs, can use the train_step function and batches from the images to train the model
def train(epochs):
    for i in range(1, epochs+1):
        start = time.time()
        counter_batch = 0
        for image_batch in data:
            train_step(image_batch, (counter_batch, i))
            counter_batch += 1
        print("Time for Epoch {}: {} secs".format(i, time.time()-start))

#Function that, when given input y, will generate y x y images of the generators output
def generate_images(side_num):
    fig, axes = plt.subplots(nrows=side_num, ncols=side_num)
    for i in range(side_num):
        for j in range(side_num):
            image_pred = generator(np.random.rand(1,100), training=False)[0,:,:,0]
            axes[i][j].imshow(image_pred, cmap='gray')
    fig.tight_layout()
    plt.show()

#The model was trained on 30 epochs, and 4x4 = 16 images were outputted
train(30)
generate_images(4)
