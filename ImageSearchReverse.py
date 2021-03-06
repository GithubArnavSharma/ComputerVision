#Import neccessary modules
import numpy as np
import matplotlib.pyplot as plt
import joblib
from zipfile import ZipFile
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from keras.applications import InceptionV3
from keras import Model

#Load the InceptionV3 Model, an Image Recognition model developed by Google with a 78.1% accuracy on the ImageNet images
#It uses a mixture of convultion2d layers, max pooling layers, average pooling layers, softmax layers, fully connected layers, concatenation, etc...
#The last layer is the 1000 neuron softmax layer of all the classes, so we will take that out to use the last fully connected layer before that
#The output will be a 2048 dimensional vector which will represent the image and can be used in comparison with other vectors
inception = InceptionV3()
inception = Model(inception.input, inception.layers[-2].output)

def vectorize_image(file_name):
    image = Image.open(file_name)
    image.thumbnail((299,299))
    image = np.expand_dims(np.array(image), axis=0)
    vector = inception(image / 255.0)
    return np.array(vector).reshape(2048,)

image_vectors = []
image_files = []
with ZipFile('C:/Users/arnie2014/Desktop/Images.zip') as zip_fi:
    for entry in zip_fi.infolist():
        with zip_fi.open(entry) as file:
            vector = vectorize_image(file)
            image_vectors.append(vector)
            
            file_name = str(file).split("name='")[1].split("'")[0]
            image_files.append(file_name)
image_vectors = np.array(image_vectors)
image_files = np.ravel(np.array(image_files))

model = KNeighborsClassifier(1).fit(image_vectors, image_files)
joblib.dump(model, 'nearest_image.sav')
model = joblib.load('nearest_image.sav')

def find_image(file_name):
    with ZipFile('C:/Users/arnie2014/Desktop/Images.zip') as zip_fi:
        for entry in zip_fi.infolist():
            if file_name in str(entry):
                with zip_fi.open(entry) as file:
                    return np.array(Image.open(file))

def predict_image(file_name):
    vector = vectorize_image(file_name).reshape(1,2048)
    pred_file = model.predict(vector)[0]
    image_pred = find_image(pred_file)

    fig, axes = plt.subplots(1, 2)
    original_img = np.array(Image.open(file_name))
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image:')
    axes[1].imshow(image_pred)
    axes[1].set_title('Nearest Image: ')
    fig.tight_layout()
    plt.show()
