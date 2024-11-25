from sklearn.linear_model import *
from .utils import *
import pickle


class LinearTreeBase():
    def __init__(self, parameters, X_train, y_train, X_test, y_test, time_pred):
        self.parameters = parameters
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time_pred = time_pred


    def LinearTree(self, model):
        regr = model(LinearRegression())
        regr.set_params(**self.parameters)
        regr.fit(self.X_train, self.y_train)
        return regr


