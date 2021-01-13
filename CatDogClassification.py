import numpy as np
import random
import os
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, LeakyReLU, Dense, Flatten
from keras.preprocessing.image import load_img, img_to_array

X = []
y = []

def img_to_arr(the_path, dog_img):
    img = load_img(the_path+'\\'+dog_img, target_size=(150,150))
    img_arr = img_to_array(img)
    img_arr = img_arr / 255.0
    return img_arr

def return_images(the_path, which_one):
    X = []
    y = []
    counter = 0
    for dog_img in os.listdir(the_path):
        X.append(img_to_arr(the_path, dog_img))
        y.append(which_one)
        counter += 1
        if counter == 4000:
            break
    return (X, y)

print("Collecting dogies:")
X_dog, y_dog = return_images('C:\\Users\\arnie2014\\Desktop\\dogs', 1)
print("Collecting caties:")
X_cat, y_cat = return_images('C:\\Users\\arnie2014\\Desktop\\cats', 0)
for arr in X_cat:
    X_dog.append(arr)
for num in y_cat:
    y_dog.append(num)
X = np.array(X_dog)
y = np.array(y_dog)

xy = list(zip(X, y))
random.shuffle(xy)
X, y = zip(*xy)
X = np.array(list(X))
y = np.array(list(y))

model = Sequential()
kernel_size = 3
for filters in [32, 64, 128]:
    model.add(Conv2D(filters, kernel_size=kernel_size,
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size=64, epochs=5)

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
