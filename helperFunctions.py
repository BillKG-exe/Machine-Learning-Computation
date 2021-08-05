import pickle
from sklearn import preprocessing


def castArrayElements(array):
    new_array = []
    for i in range(len(array)):
        if not array[i].isdigit():
            continue
        else:
            new_array.append(int(array[i]))
    return new_array


def displayMatrix(array):
    index = 0
    while index < len(array):
        if index < len(array):
            if ((index + 1) % 3) != 0 or index == 0:
                print("%-3d %10s %10s" % ((index + 1), array[index], " "), end=" ")
            else:
                print("%-3d %10s" % ((index + 1), array[index]))
        else:
            print("%-3d %10s" % ((index + 1), array[index]))
        index += 1


def printTestData(predictions, x_test, y_test):
    for x in range(len(predictions)):
        print(predictions[x], end=' ')
        print(x_test[x], end=' ')
        print(y_test[x])


def saveModel(model):
    save = ''
    while save.lower() != 'yes' and save.lower() != 'no':
        save = input('\nWould you like to save your model?(yes/no): ')
        if save != 'yes' and save != 'no':
            print('\nERROR: Please enter yes or no ')
    save = save.lower()
    if save == 'yes':
        model_name = input('Enter the name of the model you\'d like to save: ')
        path = "C:\\Users\\ouatt\\PycharmProjects\\Machine Learning Tutorial\\existingModels\\"
        path += model_name + '.pickle'
        with open(path, "wb") as f:
            pickle.dump(model, f)


def loadModel():
    load = ''
    while load.lower() != 'yes' and load.lower() != 'no':
        load = input('\nwould you like to load an existing model?(yes/no): ')
        if load != 'yes' and load != 'no':
            print('\nERROR: Please enter yes or no ')
    load = load.lower()
    if load == 'yes':
        model_name = input('Enter the name of your existing model: ')
        path = "C:\\Users\\ouatt\\PycharmProjects\\Machine Learning Tutorial\\existingModels\\"
        path += model_name + '.pickle'
        pickle_in = open(path, "rb")
        return pickle.load(pickle_in)
    else:
        return None


def toNumerical(data, features):
    le = preprocessing.LabelEncoder()

    for i in range(len(features)):
        data[features[i]] = le.fit_transform(list(data[features[i]]))

    print('\nData conversion to numerical only:\n')
    print(data.head())
    return data


def convertTestInput(input_test):
    new_array = []
    le = preprocessing.LabelEncoder()
    for i in range(len(input_test)):
        if input_test[i].isdigit():
            print(input_test[i])
            input_test[i] = int(input_test[i])
        new_array.append(le.fit_transform([input_test[i]])[0])

    return input_test


def validated_input(msg, input_type, conditions, size=0):
    valid = True
    if input_type == "str":
        while True:
            try:
                str_selection = input(msg)
                str_selection = str_selection.lower()
                if str_selection in conditions:
                    return str_selection
                else:
                    print('\nInvalid input: Please try again')
            except ValueError:
                print('\nERROR: Please enter string')
    elif input_type == "int":
        while True:
            try:
                int_selection = int(input(msg))
                if conditions[0] == -1:
                    if 0 < int_selection <= size:
                        return int_selection
                elif int_selection in conditions:
                    return int_selection
                else:
                    print('\nInvalid Input: Please enter integer value between:', conditions)
            except ValueError:
                print('\nERROR: Please enter an integer value')
    elif input_type == "int_list":
        while True:
            try:
                list_selections = input(msg)
                list_selections = castArrayElements(list_selections.split(' '))
                if size != 0 and len(list_selections) > size:
                    print('ERROR: Size of list exceed the required size,', size)
                    continue
                elif size != 0 and len(list_selections) < size:
                    print('ERROR: Size of list is less than the required size,', size)
                    continue
                else:
                    if conditions[0] != -1:
                        for i in range(len(list_selections)):
                            if list_selections[i] > conditions[0]:
                                valid = False
                                break
                if valid:
                    return list_selections
                else:
                    print('\nERROR: Please enter integers between range 0 to', conditions[0])
                    valid = True
            except ValueError:
                print('VALUE ERROR: Please enter integers')
