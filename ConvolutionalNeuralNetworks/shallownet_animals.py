from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ConvolutionalNeuralNetworks.datasets.simpledatasetloader import SimpleDatasetLoader
from ConvolutionalNeuralNetworks.conv.shallownet import ShallowNet
from ConvolutionalNeuralNetworks.preprocessing.simplepreprocessor import SimplePreprocessor
from ConvolutionalNeuralNetworks.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

from keras.optimizers import SGD
from imutils import paths
from matplotlib.pyplot import plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="Path to the input dataset")
args=vars(ap.parse_args())

#Grab the list of images that we'll be describing
imagePaths=list(paths.list_images(args["dataset"]))

#Initialize the image preprocessors
sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreprocessor()

#Load the dataset from the disk and then scale the raw pixel intensities to range [0,1]

sdl=SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels)=sdl.load(imagePaths, verbose=500)

data=data.astype("float")/255.0



#Partition the data into training and testing splits using 75% of the data for training and the remaining data for testing
(train_X, test_X, train_Y, test_Y)=train_test_split(data, labels, test_size=0.25, random_state=42)

#Convert the labels from integers to vectors

trainY=LabelBinarizer().fit_transform(train_Y)
testY=LabelBinarizer().fit_transform(test_Y)

print("Compiling the model ... ")
opt=SGD(lr=0.05)

model=ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#Print the network 
print("[INFO] training the network....")
H=model.fit(train_X, trainY, validation_data=(test_X, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] evaluating network....")
predictions=model.predict(test_X, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog","panda"]))

plt.figure()
plt.plot(np.arange(0,100), H.history("loss"), label="train_loss")
plt.plot(np.arange(0,100), H.history("val_loss"), label="val_loss")
plt.plot(np.arange(0,100), H.history("acc"), label="train_acc")
plt.plot(np.arange(0,100), H.history("val_acc"), label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.show()
