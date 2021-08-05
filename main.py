from File import File
from ComputationHandler import MLComputation
from helperFunctions import validated_input

OPTIONS = ['Linear Regression', 'KNN Algorithm', 'Support Vector Machine']

algorithm_selected = -1


def inputFile():
    repeat = True
    filename = ""
    while repeat:
        try:
            filename = input('Enter the filename with its extension: ')
            file_test = open(filename, 'r')
            file_test.close()
            repeat = False
        except FileNotFoundError:
            print('\nNo such file or directory\n')

    return filename


def displayMenu():
    global algorithm_selected
    print('\nPlease select one option from the following: ')
    for i in range(len(OPTIONS)):
        print((i + 1), OPTIONS[i])
    algorithm_selected = validated_input('Option selected: ', 'int', [1, 2, 3])


def main():
    displayMenu()
    file = File(inputFile())

    sep = input('Enter the data separator of your file: ')

    file.fetchData(sep)
    model = MLComputation(file.getX(), file.getY(), file.filename)

    if algorithm_selected == 1:
        model.linearRegressionComputation(file.getFilename(), file.getHarshData(), file.getFeatures(),
                                          file.getPredict())
    elif algorithm_selected == 2:
        model.knnComputation(file.getFilename(), file.getHarshData(), file.getFeatures(), file.getPredict())
    elif algorithm_selected == 3:
        model.svmComputation(file.getFilename(), file.getHarshData(), file.getFeatures(), file.getPredict())

    analysis = validated_input('Would you like to analyze the relationship between the features and prediction? (yes, '
                               'no): ', "str", ["yes", "no"])
    if analysis == "yes":
        file.graphAnalysis()


main()
