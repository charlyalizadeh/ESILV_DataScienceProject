from datawrapper import DataWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score


class Model:
    """Class used to test diverse prediction model.

    :param DataWrapper data: DataWrapper object to work with.
    """

    def __init__(self, data=None):
        self.data = DataWrapper() if data is None else data

    def build_rf_boostrsap(self, fit=False, train=None, **kwargs):
        model = RandomForestClassifier(bootstrap=True, **kwargs)
        if fit:
            if train is None:
                train = self.train_set
            features = train.iloc[:, :36].to_numpy()
            classes = train.iloc[:, 36].to_numpy()
            model.fit(features, classes)
        return model

    def build_rf_bagging(self, fit=False, train=None, **kwargs):
        model = BaggingClassifier(**kwargs)
        if fit:
            if train is None:
                train = self.train_set
            features = train.iloc[:, :36].to_numpy()
            classes = train.iloc[:, 36].to_numpy()
            model.fit(features, classes)
        return model

    def predict(self, model, test=None):
        if test is None:
            test = self.test_set.iloc[:, :36].to_numpy()
        return model.predict(test)

    def cross_validate(self, models, cv=10):
        print(f"Cross Validation of {len(models)} models.")
        print(f"Number of training observations: {len(self.train_set)}\n-----")
        index_fitted = [i for i in range(len(models)) if hasattr(models[i], "classes_")]
        if index_fitted:
            print(f"Warning: model {index_fitted} is fitted.") if len(
                index_fitted
            ) == 1 else print(f"Warning: models {index_fitted} are fitted.")

        features = self.train_set.iloc[:, :36]
        classes = self.train_set.iloc[:, 36]
        best_model = models[0]
        print(f" -> Cross validation of model 0: {models[0]}")
        best_accuracy = cross_val_score(models[0], features, classes, cv=cv)
        best_accuracy = sum(best_accuracy) / len(best_accuracy)
        accuracies = [best_accuracy]
        for i in range(1, len(models)):
            print(f" -> Cross validation of model {i}: {models[i]}")
            accuracy = cross_val_score(models[i], features, classes, cv=cv)
            accuracy = sum(accuracy) / len(accuracy)
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = models[i]
        return accuracies, best_model, best_accuracy

    def __getattr__(self, key):
        if key == "train_set":
            return self.data.train_set
        if key == "test_set":
            return self.data.test_set
        if key == "label":
            return self.data.label
