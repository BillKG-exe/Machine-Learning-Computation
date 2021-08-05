import numpy as np
import sklearn
from sklearn import svm
from sklearn import metrics
from helperFunctions import *


class SVM:
    def __init__(self, filename, data, features, predict_features):
        self.filename = filename
        self.data = data
        self.features = features
        self.predict = predict_features
        self.testPredictions = None
        self.accuracy = None
        self.X = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.classifier = None
        self.prediction = None

        self.data = toNumerical(self.data, self.features)
        self.X = np.array(self.data.drop(self.predict, axis=1))
        self.y = np.array(self.data[self.predict])

    def loadModel(self):
        x_train, self.x_test, y_train, self.y_test = sklearn.model_selection.train_test_split(self.X, self.y,
                                                                                              test_size=0.2)
        try:
            self.classifier = loadModel()
            if self.classifier is None:
                self.createModel(x_train, y_train)
                saveModel(self.classifier)
            else:
                self.testPredictions = self.classifier.predict(self.x_test)
        except OSError:
            print('\nModel entered was not found')
            self.createModel(x_train, y_train)
            saveModel(self.classifier)

    def createModel(self, x_train, y_train):
        change_data = True
        while change_data:
            kernel = input('\nEnter the kernel of the SVM model: ')
            degree = int(input('Enter the degree of your model: '))
            self.classifier = svm.SVC(kernel=kernel, degree=degree)
            self.classifier.fit(x_train, y_train)
            self.testPredictions = self.classifier.predict(self.x_test)
            self.accuracy = metrics.accuracy_score(self.y_test, self.testPredictions)
            print('Accuracy of current model: ', self.accuracy, '\n')

            while True:
                try:
                    leave_test = int(input('Enter 1 to continue testing or 0 to keep current model: '))
                    if leave_test < 0 or leave_test > 1:
                        raise Exception('ERROR: Enter 1 or 0')
                    elif leave_test == 0:
                        change_data = False
                    break
                except ValueError:
                    print('ERROR: Please enter integer value')

    def printTest(self):
        printTestData(self.testPredictions, self.x_test, self.y_test)

    def getPrediction(self, data):
        print(self.classifier.predict([data]))

    def getAcc(self):
        return self.accuracy

    def getXTestSize(self):
        return len(self.x_test[0])
