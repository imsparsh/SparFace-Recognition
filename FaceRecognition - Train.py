__author__ = 'imsparsh'

print("Importing modules..")
import sys, os
import numpy as np
from PIL import Image
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model.logistic import LogisticRegression
import sklearn.cross_validation as scv
import sklearn.preprocessing as scp


print("Import successful.")

class FaceRecognition():
    def __init__(self):
        print("Facial Recognition Using the PCA")

    def scanImages(self, inDir, outDir):
        # scan images from folder and dump to pickle
        for root, directories, files in os.walk(inDir):
            fileclass = os.path.join(os.path.abspath(outDir), os.path.split(root)[1])+'.pickle'
            Y_train = [] # label data list
            X_train = [] # image data list
            for filename in files:
                filepath = os.path.join(root, filename)
                print("Scanning file: "+filepath)
                getData = self.imgToFeature(filepath) # scan data
                X_train.append(getData) # append data with label
                Y_train.append(os.path.split(root)[1])
            if len(X_train) > 0:
                pickleF = open(fileclass, 'wb')
                pickle.dump(X_train, pickleF)
                pickle.dump(Y_train, pickleF)
                pickleF.close()
                print("")
                print("Dumping pickle to: "+fileclass)
                print("")

    def imgToFeature(self, imagePath):
        # convert to grayscale, reduce size, and return numpy array
        image = Image.open(imagePath) # open image in binary
        image = image.resize((64, 48), Image.ANTIALIAS) # reduce size
        r, g, b = image.split() # partition the color scheme
        image = Image.merge("RGB", (g,g,g)).convert("L") # convert to grayscale by merging RGB bands
        return list(image.getdata())# return image data in list

    def loadFeatures(self, directory):
        # pickle to train_test data
        X, Y = [], []
        for person in os.listdir(directory):
            filepath = os.path.join(directory, person)
            print("Reading features from: "+filepath)
            pickleF = open(filepath, 'rb')
            X.extend(pickle.load(pickleF))
            Y.extend(pickle.load(pickleF))
            pickleF.close()
        return X, Y

    def featuresToPCA(self, X, Y, dirF):
        print("")
        print("Reducing data to: "+str(len(X))+", "+str(20))
        # print len(X), len(X[0])
        X = PCA(n_components=20).fit_transform(X)
        # print len(X), len(X[0])
        pickleF = open(dirF, 'wb')
        pickle.dump(X, pickleF)
        pickle.dump(Y, pickleF)
        pickleF.close()
        print("Features successfully dumped.")
    
    def loadPCAFeatures(self, dirF):
        X, Y = [], []
        if not os.path.exists(dirF):
            return False, False
        else:
            print("")
            print("Reading PCA Features from "+dirF)
            pickleF = open(dirF, 'rb')
            X.extend(pickle.load(pickleF))
            Y.extend(pickle.load(pickleF))
            pickleF.close()
            return X, Y
    
    def train(self, clf, title, X, Y, fileClf):
        #X = scp.scale(X)
        print("")
        # X_train, X_test, y_train, y_test = scv.train_test_split(X, Y, test_size=0.25, random_state=0)
        print("Train Predict starting..")
        print("")
        print(title)
        # clf.fit(X_train, y_train)
        clf.fit(X, Y)
        print("Fitness completed")

        fileClf += (str(title)+'.pickle')
        pickleF = open(fileClf, 'wb')
        pickle.dump(clf, pickleF)
        pickleF.close()

if __name__ == '__main__':
    model = FaceRecognition()
    outDir = "features"
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDirPCA = "features_minimal"
    if not os.path.exists(outDirPCA):
        os.mkdir(outDirPCA)
    clfDir = "classifier_data"
    if not os.path.exists(clfDir):
        os.mkdir(clfDir)
    filePCA = os.path.join(outDirPCA, "PCAFeatures")+".pickle"
    fileClf = os.path.join(clfDir, "clf")

    model.scanImages('data_train', outDir) # comment if already collected the features

    X, Y = model.loadFeatures(outDir)

    model.featuresToPCA(X, Y, filePCA)

    X, Y = model.loadPCAFeatures(filePCA)
    if False == X:
        print("")
        print("Errrrrrrrr.... \nPCA Features not present.\nPlease try to scan the images and collect the features.")
        sys.exit(0)

    print("Classes Loaded: "+str(len(set(Y))))
    print("Features Loaded: ("+str(len(X)//len(set(Y)))+", "+str(len(X[0]))+") for each class.")
    model.train(SVC(kernel='linear', C=0.3), 'SVC(Linear)Classifier', X, Y, fileClf)
