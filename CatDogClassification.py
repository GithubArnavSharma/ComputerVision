#Import neccessary modules
import numpy as np
import random
import os
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dense, Flatten
from keras.preprocessing.image import load_img, img_to_array

#Function that can convert an image to a normalized array with size 150 x 150 x 3
def img_to_arr(the_path, dog_img):
    img = load_img(the_path+'\\'+dog_img, target_size=(150,150))
    img_arr = img_to_array(img)
    img_arr = img_arr / 255.0
    return img_arr

#Function that returns array of images based on a path given, along with which_one labels of the image
def return_images(the_path, which_one, amount_img):
    X = []
    y = []
    counter = 0
    #Make sure the length of the path is greater or equal to amount of images requested
    assert len(os.listdir(the_path)) >= amount_img
    #Convert each image in the os.listdir to an array, and append which_one to the y array
    for dog_img in os.listdir(the_path):
        X.append(img_to_arr(the_path, dog_img))
        y.append(which_one)
        counter += 1
        if counter == amount_img:
            break
    return (X, y)

#Collect the dogs
X_dog, y_dog = return_images('C:\\Users\\arnie2014\\Desktop\\dogs', 1, 4000)
#Collect the cats
X_cat, y_cat = return_images('C:\\Users\\arnie2014\\Desktop\\cats', 0, 4000)
#Combine both arrays together
for arr in X_cat:
    X_dog.append(arr)
for num in y_cat:
    y_dog.append(num)
#Convert the arrays to np arrays for keras to understand
X = np.array(X_dog)
y = np.array(y_dog)

#Create a Sequential model(model that can take sequences)
model = Sequential()
#Use Conv2D Layers to extract 32, 64, and 128 features with a kernel_size of 3 from the input
for filters in [32, 64, 128]:
    model.add(Conv2D(filters, kernel_size=3,
                     kernel_initializer='he_uniform',
                     padding='same'))
    #Add Leaky ReLU to fix the dying RelU problem
    model.add(LeakyReLU())
    #Add Max Pooling to summarize the Conv2D
    model.add(MaxPooling2D((2,2)))
#Flatten the output for a neural network to understand
model.add(Flatten())
#Make a 128 Dense neural network layer with LeakyReLU
model.add(Dense(128))
model.add(LeakyReLU())
#Add a final 1 neuron layer with sigmoid due to the problem being binary
model.add(Dense(1, activation='sigmoid'))
#Compile the model with the adam Gradient Descent optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fit the dataset with a 64 batch size and 5 epochs
model.fit(X, y, batch_size=64, epochs=5)

#Function that can predict an image given in my Desktop folder
def predict_animal(image_path):
    the_path = 'C:\\Users\\arnie2014\\Desktop'
    arr = img_to_arr(the_path, image_path)
    thePred = model.predict(arr.reshape(1,150,150,3))
    if thePred > 0.5:
        thePred = int(thePred*100)
        the_str = "{}% sure that this is a dog!".format(thePred)
    else:
        thePred = 100-int(thePred*100)
        the_str = "{}% sure that this is a cat!".format(thePred)
    return the_str
