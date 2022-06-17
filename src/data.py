import datetime
import random

import matplotlib.pyplot as plt
import numpy as np

import model as md

WAR_BEGIN_DATE = datetime.datetime(2022, 2, 24)
DATAFILE = "../data.csv"

EARLIEST_PREDICT_DAY = 10
PERCENTAGE_OF_TESTS = 20

NUM_OF_COLUMNS = 12
COLUMNS = ["troops", "troops with wounded", "tanks", "military vehicle", "artillery", "MLRS", "AA", "planes",
           "helicopters", "autos", "boats", "drones"]

COUNT_DAY_TO_PREDICT = 30

FILELINES = []
DATATABLE = []

"""
    data structure:
        0 - day of start (first day 0 value)
        1 - killed troops
        2 - troops with wounded
        3 - tanks
        4 - military vehicle
        5 - artillery
        6 - MLRS (reactive artillery)
        7 - AA (anti-aircraft)
        8 - planes
        9 - helicopters
        10 - autos
        11 - boats
        12 - drones
"""


# load file into filelines
def load_file(filename):
    global FILELINES
    file = open(filename, 'r')

    FILELINES = separate_data(file)


# get raw data from filelines
def separate_data(file):
    lines = []
    for line in file.read().splitlines():
        line = line.split('\t')
        lines.append(line)
    return lines[1::]


# create data as described datastructure
def make_dataset():
    global DATATABLE
    day = 0
    for line in FILELINES:
        x = []
        x.append(day)
        for i in range(1, len(line)):
            x.append(int(line[i]))
        DATATABLE.append(x)
        day += 1


# generate list of random indexes to test model
def generate_test_indexes():
    numtest = int(len(FILELINES) * PERCENTAGE_OF_TESTS / 100)
    testingIndexes = []

    while numtest > 0:
        rmd = random.randrange(EARLIEST_PREDICT_DAY, len(FILELINES) - 1)
        if rmd not in testingIndexes:
            testingIndexes.append(rmd)
            numtest -= 1

    testingIndexes.sort()
    return testingIndexes


# split data into train and test samples using generated sequence of indexes
def split(test_sequence):
    train = []
    test = []

    for k in range(len(DATATABLE)):
        if k in test_sequence:
            test.append(DATATABLE[k])
        else:
            train.append(DATATABLE[k])

    return train, test


# generate trained model structure
def generate_model(train):
    models = []
    for x in range(12):
        model = md.initmodel()
        model = md.train(model, train, x + 1)
        models.append(model)

    return models


# make a raw prediction for one test
# NOT THE SAME STRUCTURE AS DATATABLE
def prediction(models, test):
    answer = [test[0] + 1]

    for i in range(12):
        answer.append(int(models[i].predict([test])))
    return answer


# get a prediction for sequence of tests
# same structure as datatable
def test_model(models, tests):
    answers = []
    for test in tests:
        answers.append(prediction(models, test))
    return answers


# measure error for each prediction in {pred}
def errors(pred):
    errors = []
    for p in pred:
        day = p[0]
        error = md.squarederror(DATATABLE[day], p)
        errors.append(error)
    return errors


# create plot with error measure for every category
def create_plot(test, error, title):
    x = np.array(DATATABLE)
    x_data = list(x[:, 0])
    x_test = list(np.array(test)[:, 0])

    for i in range(NUM_OF_COLUMNS):
        y_data = list(np.array(DATATABLE)[:, i + 1])
        y_test = list(np.array(test)[:, i + 1])

        # calculate error
        y_error = []
        for j in range(len(test)):
            y_error.append(round(y_test[j] * error[j], 2))

        # build
        plt.plot(x_data, y_data, label="actual")
        plt.errorbar(x_test, y_test, label="test results", yerr=y_error, fmt='o', color='orange',
                     ecolor='red')
        plt.legend(loc="upper left")
        # legends
        plt.xlabel('day')
        plt.ylabel(COLUMNS[i])

        plt.title(COLUMNS[i] + " " + title)
        plt.grid(visible=True, axis='both', drawstyle='steps-mid')

        plt.show()


# testing of a model using generated 20% of tests
def standard_test():
    print("==== standard test ====")
    test_sequence = generate_test_indexes()
    (train, test) = split(test_sequence)
    model = generate_model(train)
    predictions = test_model(model, test)
    error = errors(predictions)

    print("predictions:")
    for p in predictions:
        print(p)
    print("\nerrors:")
    print(error)

    create_plot(predictions, error, "by day standard test")


def multiprediction():
    print("\n==== multi prediction test ====")
    test_sequence = generate_test_indexes()

    testsize = len(DATATABLE) - COUNT_DAY_TO_PREDICT
    data = DATATABLE[0:testsize - 1]
    model = generate_model(data)

    results = []
    num_predict = COUNT_DAY_TO_PREDICT
    while num_predict > 0:
        result = prediction(model, data[-1])
        print(result)
        data.append(result)
        results.append(result)
        num_predict -= 1

    error = errors(results)

    create_plot(results, error, "day by day multiprediction")

    print("predictions:")
    for p in results:
        print(p)
    print("\nerrors:")
    print(error)


# prepare dataset
def datainit():
    load_file(DATAFILE)
    make_dataset()


def main():
    datainit()
    standard_test()
    multiprediction()


if __name__ == '__main__':
    main()
