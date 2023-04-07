"""
# Install the packages below before running the program;
pip install opencv-python
pip install requests
pip install -U scikit-image
pip install imageio
"""


# Importing the dataset from sklearn
from sklearn import datasets
digits = datasets.load_digits()
import cv2
import matplotlib.pyplot as plt


# Loading my own image and processing it 
image = cv2.imread('img203.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
a=(16-gray*16).astype(int) # really weird here, but try to convert to 0..16
resized = cv2.resize(gray, (28,28))
imageBit = cv2.bitwise_not(resized)
plt.imshow(imageBit, cmap="binary")

# Import all the neccesary packages
from PIL import Image, ImageOps
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt
import imageio
import skimage.io
import io
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt


# check the image shape and the type
print(imageBit.shape)
print(type(imageBit))

# Check the contents of 'digits'
dir(digits)

print(type(digits.images))

#print(digits.images.shape)
#print(digits.images[7])

#new_array=[]

#for i in digits.images:
    #j=cv2.resize(i, (28,28))
    #new_array.append(j)
#new_array=np.array(new_array)

# resizing the image and creating a new tuple of resized images and convert them into a np array
new_array=np.array(tuple(cv2.resize(i, (28,28)) for i in digits.images))

print(new_array.shape)

print(new_array[7])

# Displaying a random number (7 in this case) through the array index to see if it is correct
#plt.imshow(digits.images[7], cmap="binary")
plt.imshow(new_array[7], cmap="binary")
plt.show()

print(digits.target.shape)
print(digits.target)

def plot_multi(i):
    nplots=16
    fig=plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(new_array[i+j], cmap="binary")
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()

plot_multi(0)

y = digits.target
x = new_array.reshape((len(digits.images), -1))
z = imageBit.reshape(1,-1)
print(x.shape)
print(z.shape)

x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,), activation='logistic',alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init =.05, max_iter=300, verbose=True)

mlp.fit(x_train, y_train)

predictions = mlp.predict(z)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

