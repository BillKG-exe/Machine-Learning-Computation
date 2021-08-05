import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from helperFunctions import *


class KNN:
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
        self.model = None
        self.prediction = None

        self.data = toNumerical(self.data, self.features)

        self.X = np.array(self.data.drop(self.predict, axis=1))
        self.y = np.array(self.data[self.predict])

    def loadModel(self):
        x_train, self.x_test, y_train, self.y_test = sklearn.model_selection.train_test_split(self.X, self.y,
                                                                                              test_size=0.1)
        try:
            self.model = loadModel()
            if self.model is None:
                self.createModel(x_train, y_train)
                saveModel(self.model)
        except OSError:
            print('\nERROR: The model was not found')
            self.createModel(x_train, y_train)
            saveModel(self.model)
        self.testPredictions = self.model.predict(self.x_test)

    def printTest(self):
        printTestData(self.testPredictions, self.x_test, self.y_test)

    def getPrediction(self, data):
        print(self.model.predict([data]))

    def getAcc(self):
        return self.accuracy

    def createModel(self, x_train, y_train):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(x_train, y_train)
        self.accuracy = self.model.score(self.x_test, self.y_test)

    def getXTestSize(self):
        return len(self.x_test[0])
