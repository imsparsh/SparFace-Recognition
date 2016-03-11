__author__ = 'imsparsh'

# initialize Libraries
print("Importing modules..")
import resources_rc
from PyQt4 import QtCore, QtGui

import sys, os
import numpy as np
from PIL import Image
import pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import sklearn.cross_validation as scv
import sklearn.preprocessing as scp
print("Import successful.")

classifierPath = "./classifier_data/clfSVC(Linear)Classifier.pickle"
print("")
print("Loading classifier..")

pickleF = open(classifierPath, 'rb')
clf = pickle.load(pickleF)
pickleF.close()
print("Done!")

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class WindowLayout(QtGui.QWidget):
    def __init__(self):
        '''        initialize Main Window with Widgets        '''
        super(WindowLayout, self).__init__()

        self.classF = ""
        global clf
        self.clf = clf

        # initialize Vertical Box Layout
        self.fp1 = QtGui.QVBoxLayout()
        self.fp2 = QtGui.QVBoxLayout()
        self.fp3 = QtGui.QVBoxLayout()
        self.fp4 = QtGui.QVBoxLayout()
        self.fp5 = QtGui.QVBoxLayout()
        self.fp6 = QtGui.QVBoxLayout()

        # initialize main Box and add Layouts
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(self.fp1)
        vbox.addLayout(self.fp2)
        vbox.addLayout(self.fp3)
        vbox.addLayout(self.fp4)
        vbox.addLayout(self.fp5)
        vbox.addLayout(self.fp6)

        # set layout for Main Window
        self.setLayout(vbox)

        # set internal margin and spacing between widgets to zero
        vbox.setSpacing(0)
        vbox.setMargin(0)

        # initialize first Layout Activity
        self.fp_land()

        self.setObjectName(_fromUtf8("Wizard"))
        self.setGeometry(300,100,960,540) # set height, width, position
        self.move(QtGui.QApplication.desktop().screen().rect().center()- self.rect().center()) # reset position to center
        self.resize(960, 540)
        # fix height and width
        self.setMinimumSize(QtCore.QSize(960, 540))
        self.setMaximumSize(QtCore.QSize(960, 540))
        self.setWindowTitle(_translate("Wizard", "Face Recognition System", None)) # set Window Title
        
        # add icon to WizardPageH | Taskbar Panel
        app_icon = QtGui.QIcon()
        app_icon.addFile('./icon.png', QtCore.QSize(16,16))
        app_icon.addFile('./icon.png', QtCore.QSize(24,24))
        app_icon.addFile('./icon.png', QtCore.QSize(32,32))
        app_icon.addFile('./icon.png', QtCore.QSize(48,48))
        app_icon.addFile('./icon.png', QtCore.QSize(256,256))
        # app_icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/icon/images/listen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(app_icon)

    
    # terminate all process on Escape keyPress
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
    

    def fp_land(self):

        WizardPageH = QtGui.QWizardPage()
        WizardPageH.setGeometry(0,0,320,480)
        WizardPageH.setObjectName(_fromUtf8("WizardPageH"))
        WizardPageH.resize(960, 540)
        WizardPageH.setMinimumSize(QtCore.QSize(960, 540))
        WizardPageH.setMaximumSize(QtCore.QSize(960, 540))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/icon/images/listen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WizardPageH.setWindowIcon(icon)
        WizardPageH.setAutoFillBackground(False)
        WizardPageH.setStyleSheet(_fromUtf8("QWidget  {background: url(:/bg/images/bg.png);}"))

        self.widgetH = QtGui.QWidget(WizardPageH)
        self.widgetH.setObjectName(_fromUtf8("widgetH"))
        self.bgFrame = QtGui.QFrame(self.widgetH)
        self.bgFrame.setGeometry(QtCore.QRect(-11, -1, 991, 551))
        self.bgFrame.setStyleSheet(_fromUtf8("background-image: url(:/mainBg/back.PNG);"))
        self.bgFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.bgFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.bgFrame.setObjectName(_fromUtf8("bgFrame"))
        self.shadowFrame = QtGui.QFrame(self.widgetH)
        self.shadowFrame.setGeometry(QtCore.QRect(-1, -1, 961, 561))
        self.shadowFrame.setStyleSheet(_fromUtf8("background-color: rgba(0, 0, 0, 100);"))
        self.shadowFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.shadowFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.shadowFrame.setObjectName(_fromUtf8("shadowFrame"))
        self.frame = QtGui.QFrame(self.widgetH)
        self.frame.setGeometry(QtCore.QRect(-21, -12, 1001, 561))
        self.frame.setStyleSheet(_fromUtf8("#frame {background: transparent;}"))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.create = QtGui.QPushButton(self.frame)
        self.create.setGeometry(QtCore.QRect(300, 390, 421, 91))
        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.create.setToolTip(_fromUtf8(""))
        self.create.setWhatsThis(_fromUtf8(""))
        self.create.setStyleSheet(_fromUtf8("QPushButton {background:white; font: 22pt \"Calibri\"; border-radius: 50px 0; color: #210000; }"))
        self.create.setAutoDefault(False)
        self.create.setDefault(False)
        self.create.setFlat(False)
        self.create.setObjectName(_fromUtf8("create"))
        self.pushButton = QtGui.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(880, 510, 81, 31))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet(_fromUtf8("QPushButton {background: rgba(0, 0, 0, 50); font: 12pt \"Calibri\"; border-radius: 10px 0; color: #ffffff; }"))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.create_2 = QtGui.QPushButton(self.frame)
        self.create_2.setGeometry(QtCore.QRect(20, 110, 970, 131))
        self.create_2.setToolTip(_fromUtf8(""))
        self.create_2.setWhatsThis(_fromUtf8(""))
        self.create_2.setStyleSheet(_fromUtf8("QPushButton {background:rgba(0, 0, 0, 80); font: 50pt \"Calibri\"; border-radius: 50px 0; color: white; }"))
        self.create_2.setAutoDefault(False)
        self.create_2.setDefault(False)
        self.create_2.setFlat(False)
        self.create_2.setObjectName(_fromUtf8("create_2"))

        WizardPageH.setWindowTitle(_translate("WizardPage1", "FACE RECOGNITION SYSTEM", None))
        self.create.setText(_translate("WizardPage1", "ENTER SESSION", None))
        self.pushButton.setText(_translate("WizardPage1", "EXIT", None))
        self.create_2.setText(_translate("WizardPage1", "FACE RECOGNITION SYSTEM", None))
        QtCore.QMetaObject.connectSlotsByName(WizardPageH)

        self.fp1.addWidget(WizardPageH)

        self.pushButton.mousePressEvent = self.fp_close
        self.connect(self.create, QtCore.SIGNAL("clicked()"), self.fp_land_one)

    def fp_land_one(self):
        '''
        change Layout one to two
        '''
        self.remove_fp_land()
        self.fp_one()

    def fp_one(self):
        '''
        the first Layout
        '''

        WizardPage1 = QtGui.QWizardPage()
        WizardPage1.setGeometry(0,0,320,480)
        WizardPage1.setObjectName(_fromUtf8("WizardPage1"))
        WizardPage1.resize(960, 540)
        WizardPage1.setMinimumSize(QtCore.QSize(960, 540))
        WizardPage1.setMaximumSize(QtCore.QSize(960, 540))
        WizardPage1.setAutoFillBackground(False)
        WizardPage1.setStyleSheet(_fromUtf8("QWidget  {background: url(:/bg/images/bg.png);}"))

        self.widget1 = QtGui.QWidget(WizardPage1)
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.bgFrame = QtGui.QFrame(self.widget1)
        self.bgFrame.setGeometry(QtCore.QRect(-11, -1, 991, 551))
        self.bgFrame.setStyleSheet(_fromUtf8("background-image: url(:/mainBg/minor_back.png);"))
        self.bgFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.bgFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.bgFrame.setObjectName(_fromUtf8("bgFrame"))
        self.shadowFrame = QtGui.QFrame(self.widget1)
        self.shadowFrame.setGeometry(QtCore.QRect(-1, -1, 961, 551))
        self.shadowFrame.setStyleSheet(_fromUtf8("background-color: rgba(0, 0, 0, 100);"))
        self.shadowFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.shadowFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.shadowFrame.setObjectName(_fromUtf8("shadowFrame"))
        self.frame = QtGui.QFrame(self.widget1)
        self.frame.setGeometry(QtCore.QRect(-21, -12, 1001, 561))
        self.frame.setStyleSheet(_fromUtf8("#frame {background: transparent;}"))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.pushButton = QtGui.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(880, 500, 81, 31))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet(_fromUtf8("QPushButton {background: rgba(0, 0, 0, 50); font: 12pt \"Calibri\"; border-radius: 10px 0; color: #ffffff; }"))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(640, 280, 240, 50))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet(_fromUtf8("QPushButton {background: rgba(255,255,255,235); font: 15pt  \"Comic Sans\"; border-radius: 50px 0; color: black;}"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(630, 370, 270, 61))
        self.pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.setStyleSheet(_fromUtf8("QPushButton {background:black; font: 20pt \"Calibri\"; border-radius: 20px 0; color: #ffffff; }"))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.input_image_frame = QtGui.QFrame(self.frame)
        self.input_image_frame.setGeometry(QtCore.QRect(70, 140, 480, 330))
        self.input_image_frame.setStyleSheet(_fromUtf8("QFrame {background:rgba(255, 255, 255, 200);  }"))
        self.input_image_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.input_image_frame.setFrameShadow(QtGui.QFrame.Raised)
        self.input_image_frame.setObjectName(_fromUtf8("input_image_frame"))
        self.label = QtGui.QLabel(self.input_image_frame)
        self.label.setGeometry(QtCore.QRect(10, 10, 460, 310))
        self.label.setObjectName(_fromUtf8("label"))

        self.scanImage = QtGui.QLabel(self.input_image_frame)
        self.scanImage.setGeometry(QtCore.QRect(10, 10, 460, 310))
        self.scanImage.setText(_fromUtf8(""))
        # self.scanImage.setPixmap(QtGui.QPixmap(_fromUtf8("./back.png")))
        self.scanImage.setScaledContents(True)
        self.scanImage.setAlignment(QtCore.Qt.AlignCenter)
        self.scanImage.setObjectName(_fromUtf8("scanImage"))

        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 961, 91))
        self.label_2.setStyleSheet(_fromUtf8("QLabel {background:rgba(0, 0, 0, 180); font: 25pt \"Calibri\"; border-radius: 50px 0; color: white; }"))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.pushButton_4 = QtGui.QPushButton(self.frame)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 500, 81, 31))
        self.pushButton_4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_4.setStyleSheet(_fromUtf8("QPushButton {background: rgba(0, 0, 0, 50); font: 12pt \"Calibri\"; border-radius: 10px 0; color: #ffffff; }"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))

        # initialize Labels for the widgets
        WizardPage1.setWindowTitle(_translate("WizardPage1", "FACE RECOGNITION SYSTEM", None))
        self.pushButton.setText(_translate("WizardPage1", "EXIT", None))
        self.pushButton_2.setText(_translate("WizardPage1", "Browse", None))
        self.pushButton_3.setText(_translate("WizardPage1", "Recognize", None))
        self.label.setText(_translate("WizardPage1", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">No Image Selected</span></p></body></html>", None))
        self.label_2.setText(_translate("WizardPage1", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt;\">FETCH FACES</span></p></body></html>", None))
        self.pushButton_4.setText(_translate("WizardPage1", "HOME", None))
        self.scanImage.hide()
        self.pushButton_3.setDisabled(True)
        QtCore.QMetaObject.connectSlotsByName(WizardPage1)

        self.fp2.addWidget(WizardPage1) # add Wizard Page to Layout

        # initialize Click Events
        self.connect(self.pushButton_2, QtCore.SIGNAL("clicked()"), self.file_open)
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.recognize)
        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.fp_one_land)
        self.pushButton.mousePressEvent = self.fp_close


    def file_open(self):
        '''
        select the file to match
        '''
        self.found = 0
        filter_mask = "Music files (*.jpg *.jpeg *.JPG *.PNG *.png)"
        openPath = 'C:/'
        # get full filenames along with path
        self.fullFileName = QtGui.QFileDialog.getOpenFileName(self, 'Select Files', openPath , filter_mask)
        # get individual filename in list
        self.fileName = self.fullFileName.split('\\')[-1]
        #self.remainingSongs.setText(self.fileName)
        
        if not self.fullFileName == '': # check if the file is received
            # print(self.fullFileName)
            self.scanImage.setPixmap(QtGui.QPixmap(_fromUtf8(self.fullFileName)))
            self.scanImage.show()
            self.pushButton_3.setDisabled(False)
        else:
            self.scanImage.setPixmap(QtGui.QPixmap(_fromUtf8("./back.png")))
            self.scanImage.hide()
            self.pushButton_3.setDisabled(True)

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
    
    def recognize(self):
        print("Recognizing..")
        imgData = self.imgToFeature(str(self.fullFileName))
        # imgData = self.featuresToPCA(imgData)
        if max(clf.predict_proba(imgData)[0]) > 0.2: # THRESHOLD = 0.2 OUT OF 1.0
            self.classF = self.clf.predict(imgData)[0]
            print("Output: "+str(self.classF))
        else:
            self.classF = "Match Not Found"
        self.fp_one_two()

    
    def fp_one_two(self):
        '''
        change Layout one to two
        '''
        self.remove_fp_one()
        self.fp_two()

    def fp_one_land(self):
        '''
        change Layout one to three
        '''
        self.remove_fp_one()
        self.fp_land()

    def fp_two(self):
        '''
        the second Layout
        '''
        WizardPage2 = QtGui.QWizardPage()
        WizardPage2.setObjectName(_fromUtf8("WizardPage2"))
        WizardPage2.resize(960, 540)
        WizardPage2.setMinimumSize(QtCore.QSize(960, 540))
        WizardPage2.setMaximumSize(QtCore.QSize(960, 540))
        WizardPage2.setAutoFillBackground(False)
        WizardPage2.setStyleSheet(_fromUtf8("QWidget  {background: url(:/bg/images/bg.png);}"))

        self.widget2 = QtGui.QWidget(WizardPage2)
        self.widget2.setObjectName(_fromUtf8("widget2"))
        self.bgFrame = QtGui.QFrame(self.widget2)
        self.bgFrame.setGeometry(QtCore.QRect(-11, -1, 991, 551))
        self.bgFrame.setStyleSheet(_fromUtf8("background-image: url(:/mainBg/minor_back.png);"))
        self.bgFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.bgFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.bgFrame.setObjectName(_fromUtf8("bgFrame"))
        self.shadowFrame = QtGui.QFrame(self.widget2)
        self.shadowFrame.setGeometry(QtCore.QRect(-1, -1, 961, 551))
        self.shadowFrame.setStyleSheet(_fromUtf8("background-color: rgba(155,155,155, 100);"))
        self.shadowFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.shadowFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.shadowFrame.setObjectName(_fromUtf8("shadowFrame"))
        self.frame = QtGui.QFrame(self.widget2)
        self.frame.setGeometry(QtCore.QRect(-21, -12, 1001, 561))
        self.frame.setStyleSheet(_fromUtf8("#frame {background: transparent;}"))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.exitF2 = QtGui.QPushButton(self.frame)
        self.exitF2.setGeometry(QtCore.QRect(880, 500, 81, 31))
        self.exitF2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exitF2.setStyleSheet(_fromUtf8("QPushButton {background: rgba(0, 0, 0, 50); font: 12pt \"Calibri\"; border-radius: 10px 0; color: #ffffff; }"))
        self.exitF2.setObjectName(_fromUtf8("exitF2"))
        self.input_image_frame = QtGui.QFrame(self.frame)
        self.input_image_frame.setGeometry(QtCore.QRect(180, 160, 270, 320))
        self.input_image_frame.setStyleSheet(_fromUtf8("QFrame {background:rgba(255, 255, 255, 50);  }"))
        self.input_image_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.input_image_frame.setFrameShadow(QtGui.QFrame.Raised)
        self.input_image_frame.setObjectName(_fromUtf8("input_image_frame"))
        self.outImage = QtGui.QLabel(self.input_image_frame)
        self.outImage.setGeometry(QtCore.QRect(10, 10, 250, 300))
        self.outImage.setText(_fromUtf8(""))
        self.outImage.setPixmap(QtGui.QPixmap(_fromUtf8(self.fullFileName)))
        self.outImage.setScaledContents(True)
        self.outImage.setObjectName(_fromUtf8("outImage"))
        self.title = QtGui.QLabel(self.frame)
        self.title.setGeometry(QtCore.QRect(20, 30, 961, 91))
        self.title.setStyleSheet(_fromUtf8("QLabel {background:rgba(0, 0, 0, 180); font: 25pt \"Calibri\"; border-radius: 50px 0; color: white; }"))
        self.title.setObjectName(_fromUtf8("title"))
        self.homeF2 = QtGui.QPushButton(self.frame)
        self.homeF2.setGeometry(QtCore.QRect(40, 500, 81, 31))
        self.homeF2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.homeF2.setStyleSheet(_fromUtf8("QPushButton {background: rgba(0, 0, 0, 50); font: 12pt \"Calibri\"; border-radius: 10px 0; color: #ffffff; }"))
        self.homeF2.setObjectName(_fromUtf8("homeF2"))
        self.classOutput = QtGui.QLabel(self.frame)
        self.classOutput.setGeometry(QtCore.QRect(550, 310, 281, 71))
        self.classOutput.setObjectName(_fromUtf8("classOutput"))

        # initialize Labels for the widgets
        WizardPage2.setWindowTitle(_translate("WizardPage2", "WizardPage2", None))
        self.exitF2.setText(_translate("WizardPage2", "EXIT", None))
        self.title.setText(_translate("WizardPage2", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt;\">FETCH FACES</span></p></body></html>", None))
        self.homeF2.setText(_translate("WizardPage2", "HOME", None))
        self.classOutput.setText(_translate("WizardPage2", "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; color:#000000;\">"+str(self.classF)+"</span></p></body></html>", None))
        QtCore.QMetaObject.connectSlotsByName(WizardPage2)

        self.fp3.addWidget(WizardPage2) # add Wizard Page to Layout

        # initialize Click Events
        self.connect(self.homeF2, QtCore.SIGNAL("clicked()"), self.fp_two_land)
        self.exitF2.mousePressEvent = self.fp_close

    def fp_two_land(self):
        '''
        change Layout one to three
        '''
        self.remove_fp_two()
        self.fp_land()

    def remove_fp_land(self):
        for cnt in reversed(range(self.fp1.count())):
            # takeAt does both the jobs of itemAt and removeWidget
            # namely it removes an item and returns it
            widget = self.fp1.takeAt(cnt).widget()

            if widget is not None: 
                # widget will be None if the item is a layout
                widget.deleteLater()

    def remove_fp_one(self):
        '''
        remove nested widgets from Layout one
        '''
        for cnt in reversed(range(self.fp2.count())):
            # takeAt does both the jobs of itemAt and removeWidget
            # namely it removes an item and returns it
            widget = self.fp2.takeAt(cnt).widget()

            if widget is not None: 
                # widget will be None if the item is a layout
                widget.deleteLater()

    def remove_fp_two(self):
        '''
        remove nested widgets from Layout two
        '''
        for cnt in reversed(range(self.fp3.count())):
            # takeAt does both the jobs of itemAt and removeWidget
            # namely it removes an item and returns it
            widget = self.fp3.takeAt(cnt).widget()

            if widget is not None: 
                # widget will be None if the item is a layout
                widget.deleteLater()

    def fp_close(self, event):
        '''
        close Main Layout
        '''
        if event.button() == QtCore.Qt.LeftButton:
            sys.exit()
    
def run():
    app = QtGui.QApplication(sys.argv) # initialize QApplication
    app.setStyle(QtGui.QStyleFactory.create("Plastique")) # set theme
    ex = WindowLayout() # make object: Main Window

    # set Window Icon
    app_icon = QtGui.QIcon()
    app_icon.addFile('./icon.png', QtCore.QSize(16,16))
    app_icon.addFile('./icon.png', QtCore.QSize(24,24))
    app_icon.addFile('./icon.png', QtCore.QSize(32,32))
    app_icon.addFile('./icon.png', QtCore.QSize(48,48))
    app_icon.addFile('./icon.png', QtCore.QSize(256,256))
    # app_icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/icon/images/listen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    ex.setWindowIcon(app_icon)

    ex.show() # run the application
    sys.exit(app.exec_()) # terminate when all parsed

if __name__ == "__main__":
    '''
    main function
    '''
    run()
