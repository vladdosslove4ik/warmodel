from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

NUMBEROFVALUES = 12


def initmodel():
    model = LinearRegression()
    return model


def train(model, data, column):
    train = []
    test = []
    for x in range(1, len(data)):
        train.append(data[x - 1])
        test.append(data[x][column])
    model = fit_model(model, train, test)
    return model


def fit_model(model, data, test):
    return model.fit(data, test)


def predict(model, data):
    return model.predict(data)


def squarederror(Y_test, Y_pred):
    return round(mean_absolute_percentage_error(Y_test, Y_pred), 4)
