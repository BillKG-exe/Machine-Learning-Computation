import sklearn
import numpy as np
from sklearn import linear_model
from helperFunctions import *


class LinearModel:
    def __init__(self, filename, data, features, predict):
        self.filename = filename
        self.data = data
        self.features = features
        self.predict = predict
        self.x_test = None
        self.y_test = None
        self.model = None
        self.accuracy = 0
        self.predictions = None

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
            print('Model not found')
            self.createModel(x_train, y_train)
            saveModel(self.model)
        self.predictions = self.model.predict(self.x_test)

    def createModel(self, x_train, y_train):
        self.model = linear_model.LinearRegression()
        self.model.fit(x_train, y_train)
        self.accuracy = self.model.score(self.x_test, self.y_test)

    def printEquationData(self):
        print('\nRegression coef. : ', self.model.coef_)
        print('Regression int. : ', self.model.intercept_)

    def printTestData(self):
        for x in range(len(self.predictions)):
            print(self.predictions[x], end=' ')
            print(self.x_test[x], end=' ')
            print(self.y_test[x])

    def getAcc(self):
        return self.accuracy

    def getPrediction(self, data):
        print(self.model.predict([data]))

    def getXTestSize(self):
        return len(self.x_test[0])
