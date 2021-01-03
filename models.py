from datawrapper import DataWrapper
from sklearn.ensemble import RandomForestClassifier

class Model:
    """Class used to test diverse prediction model.

    :param DataWrapper data: DataWrapper object to work with.
    """

    def __init__(self, data=None):
        self.data = DataWrapper() if data is None else data


    def build_rf_boostrsap(self, **kwargs):
        features = self.train_set.iloc[:, :36].to_numpy()
        classes = self.train_set.iloc[:, 36].to_numpy()
        clf = RandomForestClassifier(bootstrap=True, **kwargs)
        clf.fit(features, classes)
        return clf

    def predict(self, model):
        return model.predict(self.test_set.iloc[:, :36].to_numpy())

    def __getattr__(self, key):
        if key == 'train_set':
            return self.data.train_set
        if key == 'test_set':
            return self.data.test_set
        if key == 'label':
            return self.data.label
