import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils import convert_numpy_arr_to_df


class DataWrapper:
    def __init__(self, train_set=None, test_set=None, label=None):
        if label is None:
            self.label = {1: 'Red soil',
                          2: 'Cotton crop',
                          3: 'Grey soil',
                          4: 'Damp grey soil',
                          5: 'Soil with vegetation stubble',
                          6: 'Mixture class',
                          7: 'Very damp grey soil'}
        else:
            self.label = label
        self.train_set = convert_numpy_arr_to_df(train_set, 36, self.label)
        self.test_set = convert_numpy_arr_to_df(test_set, 36, self.label)

    def import_train_set_from_txt(self, path_train):
        """Import the training set from a text file.

        :param str path_train: Path to the text file containing the training set.
        """

        self.train_set = convert_numpy_arr_to_df(np.loadtxt(path_train), 36, self.label)

    def import_test_set_from_txt(self, path_test):
        """Import the testing set from a text file.

        :param str path_test: Path to the text file containing the testing set.
        """

        self.test_set = convert_numpy_arr_to_df(np.loadtxt(path_test), 36, self.label)

    def import_train_set(self, train_set):
        """Import the training set from a numpy.array.

        :param numpy.array train_set: Matrix containing the training data.
        """

        self.train_set = convert_numpy_arr_to_df(train_set, 36, self.label)

    def import_test_set(self, test_set):
        """Import the testing set from a numpy.array.

        :param numpy.array test_set: Matrix containing the testing data.
        """

        self.test_set = convert_numpy_arr_to_df(test_set, 36, self.label)

    def apply_scaler(self, scaler):
        self.unscaled_train_set = self.train_set
        self.unscaled_test_set = self.test_set
        self.train_set = scaler.fit_transform(self.train_set)
        self.test_set = scaler.fit_transform(self.test_set)

    def scale(self, scaler_type="standard", **kwargs):
        scaler = None
        if scaler_type == "standard":
            scaler = preprocessing.StandardScaler(**kwargs)
        elif scaler_type == "minmax":
            scaler = preprocessing.MinMaxScaler(**kwargs)
        elif scaler_type == "normalize":
            scaler = preprocessing.Normalizer(**kwargs)
        self._scale(scaler)
