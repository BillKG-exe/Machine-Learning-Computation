import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
from helperFunctions import *


class File:
    def __init__(self, filename):
        self.filename = filename
        self.categories = None
        self.features = None
        self.data = None
        self.predicted_feature = None

    def fetchData(self, separator=','):
        file = open(self.filename, 'r')
        self.categories = file.readline().split(separator)
        file.close()

        self.features, predicted_feature = self.selectedCategories()
        self.features = self.convertFeatures(self.features)
        self.predicted_feature = self.categories[predicted_feature - 1].replace("\n", "")

        print('\n', self.features, self.predicted_feature)

        self.data = pd.read_csv(self.filename, sep=separator)
        self.data = self.data[self.features]

        print(self.data.head())

    def selectedCategories(self):
        print('\nSelect the features of your model from the following: ')

        displayMatrix(self.categories)

        options = validated_input('\nPlease enter a space separated list of integers: ', 'int_list',
                                  [len(self.categories)], 0)
        predict_feature = validated_input('Enter the digit of the feature to be predicted: ', 'int', [-1],
                                          len(self.categories))

        return options, predict_feature

    def getFilename(self):
        return self.filename

    def getFeatures(self):
        return self.features

    def getPredict(self):
        return self.predicted_feature

    def getX(self):
        try:
            return np.array(self.data.drop(self.predicted_feature, axis=1))
        except KeyError:
            print('ERROR: make sure the predicted feature is included in the list')

    def getY(self):
        try:
            return np.array(self.data[self.predicted_feature])
        except KeyError:
            print('ERROR: make sure the predicted feature is included in the list')

    def getHarshData(self):
        return self.data

    def printDataHead(self):
        print(self.data.head())

    def matrixDisplay(self):
        displayMatrix(self.categories)

    def convertFeatures(self, cat_selected):
        features = []
        for i in range(len(cat_selected)):
            if type(cat_selected[i]) == int:
                features.append(self.categories[cat_selected[i] - 1].replace("\n", ""))
        return features

    def graphAnalysis(self):
        for i in range(len(self.features)):
            if self.features[i] != self.predicted_feature:
                p = self.features[i]
                style.use("ggplot")
                pyplot.scatter(self.data[p], self.data[self.predicted_feature])
                pyplot.xlabel(p)
                pyplot.ylabel(self.predicted_feature)
                pyplot.show()
