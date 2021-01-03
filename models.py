from datawrapper import DataWrapper


class Model:
    """Class used to test diverse prediction model.

    :param DataWrapper data: DataWrapper object to work with.
    """

    def __init__(self, data=None):
        self.data = DataWrapper() if data is None else data
