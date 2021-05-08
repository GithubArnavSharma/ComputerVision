#Import neccessary modules
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.constraints import maxnorm #A way of constraining weights from having too high of a value and decreasing accuracy
from keras.preprocessing.image import ImageDataGenerator

#Import the data
train = pd.read_csv('mnistdigit.csv')
test = pd.read_csv('test.csv')

#Get the data, reshape it to 28 x 28 x 1, and normalize it
X = np.array([np.array(arr).reshape(28,28,1) for arr in np.array(train.drop('label', axis=1))])/255.0
y = np.array(train['label'])

#Make the sequential Neural Network
model = Sequential()
#Add Dropout to the input images in order to deal with noise and as a regularization technique 
model.add(Dropout(0.1))
#Add 2 Conv2D layers which take 32 filters, and extract them through a kernel size of 5. When these layers work, the numbers go through ReLU
model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
#Add a Max Pooling 2D Layer to simplify the model and extract the main features
model.add(MaxPooling2D(pool_size=(2,2)))
#Add a Dropout layer to reduce overfitting
model.add(Dropout(0.2))
#Add 2 other Conv2D layers, but with 64 features and a 3 kernel_size due to less to extract
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Flatten for the neural network 
model.add(Flatten())
#Add 2 128 x 128 neural nets along with a 20% Dropout for the classification of feature extraction
model.add(Dense(128, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
model.add(Dense(128, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
#Add a 10 Dense Layer with softmax activation, which normalizes the output to a probability 
model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

#Make an Image Data Generator which shifts the image around for better classification
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.1
)
datagen.fit(X)

#Compile the model with an adam optimizer 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Fit the model in a batch size of 64 and 10 epochs
model.fit(datagen.flow(X, y, batch_size=64), epochs=10)

#Use the model to predict from the test dataset
theIds = np.array([i for i in range(1,28001)])
thePred = np.array(test).reshape(test.shape[0],28,28,1)
thePred = model.predict(thePred)
preds = np.argmax(thePred, axis=1)

#Upload the new findings and IDs to a dataframe. This model got a 99% accuracy
df = pd.DataFrame({
    "ImageId":theIds,
    "Label":preds
})
df.to_csv('image_pred.csv',index=False)
