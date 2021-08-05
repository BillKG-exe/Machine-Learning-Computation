from LinearRegressionModel import LinearModel
from helperFunctions import *
from KNearestNeighBor import KNN
from SVM_model import SVM


class MLComputation:
    def __init__(self, x, y, filename):
        self.filename = filename
        self.X = x
        self.y = y

    def linearRegressionComputation(self, filename, data, features, predict):
        computation = LinearModel(self.filename, data, features, predict)
        computation.loadModel()
        option = validated_input('\nEnter 1 to see test data or 0 to continue: ', 'int', [1, 0])

        if option == 1:
            computation.printEquationData()
            computation.printTestData()
            print('\nAccuracy:', computation.getAcc())
            print('NOTE: accuracy will show 0 if you used an existing model')

        self.testLoop(computation)

    def knnComputation(self, filename, data, features, predict_features):
        computation = KNN(filename, data, features, predict_features)
        computation.loadModel()

        option = validated_input('\nEnter 1 to see test data or 0 to continue: ', 'int', [1, 0])

        if option == 1:
            computation.printTest()
            print('\nAccuracy:', computation.getAcc())
            print('NOTE: accuracy will show 0 if you used an existing model')

        self.testLoop(computation)

    def svmComputation(self, filename, data, features, predict_features):
        svm = SVM(filename, data, features, predict_features)
        svm.loadModel()

        option = validated_input('\nEnter 1 to see test data or 0 to continue: ', 'int', [1, 0])

        if option == 1:
            svm.printTest()
            print('\nAccuracy:', svm.getAcc())
            print('NOTE: accuracy will show 0 or None if you used an existing model')

        self.testLoop(svm)

    def testLoop(self, computation):
        while True:
            data = validated_input('\nEnter a space separated list of your data for prediction: ', 'int_list',
                                   [-1], computation.getXTestSize())
            print('Data:', data, 'Prediction:', end=' ')
            computation.getPrediction(data)
            exit_loop = validated_input('\nEnter 1 to test again or 0 to quit testing: ', 'int', [1, 0])
            if exit_loop == 0:
                break
