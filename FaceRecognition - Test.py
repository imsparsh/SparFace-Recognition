__author__ = 'imsparsh'

print("Importing modules..")
import sys, os
import numpy as np
from PIL import Image
import pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import sklearn.cross_validation as scv
import sklearn.preprocessing as scp


print("Import successful.")

class FaceRecognition():
    def __init__(self):
        print("Facial Recognition Using the PCA")

    def imgToFeature(self, imagePath):
        # convert to grayscale, reduce size, and return numpy array
        image = Image.open(imagePath) # open image in binary
        image = image.resize((64, 48), Image.ANTIALIAS) # reduce size
        r, g, b = image.split() # partition the color scheme
        image = Image.merge("RGB", (g,g,g)).convert("L") # convert to grayscale by merging RGB bands
        return list(image.getdata())# return image data in list

    def featuresToPCA(self, X):
        print("")
        print("Reducing data..")
        X = [X]
        X = PCA(n_components=20).fit_transform(X)
        return X
    
    def test(self, inDir, title, fileClf):
        print("")
        print("Loading classifier..")
        fileClf += (str(title)+'.pickle')

        pickleF = open(fileClf, 'rb')
        clf = pickle.load(pickleF)
        pickleF.close()

        print("Recognition starting..")
        print("")
        print(title)
        correct, incorrect = 0, 0
        for image in os.listdir(inDir):
            imagePath = os.path.join(os.path.abspath(inDir), image)
            imgData = self.imgToFeature(imagePath)
            # imgData = self.featuresToPCA(imgData)
            classF = clf.predict(imgData)
            if classF[0] in image:
                correct += 1
            else:
                incorrect += 1
        print(str(correct)+" out of "+str(correct+incorrect)+" correct.")
        print("Accuracy: "+str(correct/float(correct+incorrect)*100)+"%")



if __name__ == '__main__':
    model = FaceRecognition()
    testDir = "data_test"
    clfDir = "classifier_data"
    if not os.path.exists(clfDir):
        print("")
        print("Errrrrrrrr.... \nClassifier Features not present.\nPlease try to scan the images and collect the features.")
        sys.exit(0)
    fileClf = os.path.join(clfDir, "clf")

    model.test(testDir, 'SVC(Linear)Classifier', fileClf)
