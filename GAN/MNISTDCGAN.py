#Import neccessary modules
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets.mnist import load_data
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, LeakyReLU
from keras.optimizers import Adam


#Load the data
(data_arr, _), (_, _) = load_data()
#Change each of the images to be 28 x 28 x 1 for the discriminator
data_arr = np.array(data_arr).reshape(data_arr.shape[0], 28, 28, 1).astype('float32')
#Normalize the array to [-1,1] for tanh activation in the generator
data_arr = (data_arr - 127.5) / 127.5
#Shuffle the data and batch it into sizes of 256
BATCH_SIZE = 256
data = tf.data.Dataset.from_tensor_slices(data_arr).shuffle(len(data_arr)).batch(BATCH_SIZE)

#Discriminator model, which takes an image and outputs how real that image is
def discriminator_model():
    model = Sequential()
    #The discriminator model will have 2 Conv2D layers with 64 filters each and a 25% Dropout(to reduce overfitting)
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    #Then, the model will flatten the Conv2D Layers and make the output be from [0,1] with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

#Generation model, which takes a 100 dimensional vector and transforms it to a 28*28*1 image using Conv2D Transpose
#Conv2D Tranpose layers are like Conv2D Layers, but they upscale the input instead of downscaling it
def generator_model():
    model = Sequential()
    #Connect the input neurons to a 7*7*256 neuron layer for reshaping for the Conv2DTranspose
    model.add(Dense(7*7*256, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7,7,256)))
    #The Conv2DTranspose layers will consist of 2 128 filter ones
    model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    #At the end, add the last filter as 1, as this is a black and white picture, not an RGB one(which would need 3)
    model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model

#Use functions to assign the models to a discriminator and generator variable
discriminator = discriminator_model()
generator = generator_model()

#The loss here will be Binary Crossentropy
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#The generator will try to maximize what the discriminator predicts for it, and therefore will be a comparison between 1's and the discriminators guess
def gen_loss(fake_output):
    return cross_entropy(np.ones_like(fake_output), fake_output)

#The discriminator will try its best to get accurate in its distinction between real and fake images, so therefore the loss will be the magnitude of what it got wrong
def disc_loss(real_output, fake_output):
    real_loss = cross_entropy(np.ones_like(real_output), real_output)
    fake_loss = cross_entropy(np.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

#The optimizers used will be Adam, with the learning rate as 1e-4(10^-4), as a large learning rate on a complex model would be inefficient
disc_opt = Adam(1e-4)
gen_opt = Adam(1e-4)

#Function that can train on a single batch of images
#This GAN will learn both the generator and discriminator at the same time, so there's enough loss in the discriminator for the generator to improve,
#but not enough loss in the discriminator for the generator to make mistakes and get rewarded by the discriminator for it
def train_step(images):
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
        print(f"Epoch: {i}")
        for image_batch in data:
            train_step(image_batch)

#Function that, when given input y, will generate y x y images of the generators output
def generate_images(side_num):
    fig, axes = plt.subplots(nrows=side_num, ncols=side_num)
    for i in range(side_num):
        for j in range(side_num):
            image_pred = generator(np.random.rand(1,100), training=False)[0,:,:,0]
            axes[i][j].imshow(image_pred, cmap='gray')
    fig.tight_layout()
    plt.show()

#The model was trained on 50 epochs, and 4x4=16 images were outputted
train(50)
generate_images(4)
