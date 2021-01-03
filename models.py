from datawrapper import DataWrapper


class Model:
    """Class used to test diverse prediction model.

    :param DataWrapper data: DataWrapper object to work with.
    """

    def __init__(self, data=None):
        self.data = DataWrapper() if data is None else data


    def __getattr__(self, key):
        if key == 'train_set':
            return self.data.train_set
        if key == 'test_set':
            return self.data.test_set
        if key == 'label':
            return self.data.label
