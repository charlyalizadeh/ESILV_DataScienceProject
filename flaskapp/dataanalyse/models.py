from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def build_rf(train_set=None, **kwargs):
    model = RandomForestClassifier(**kwargs)
    if train_set:
        features = train_set.iloc[:, :36].to_numpy()
        targets = train_set.iloc[:, 36].to_numpy()
        model.fit(features, targets)
    return model


def build_rf_bagging(train_set=None, **kwargs):
    model = BaggingClassifier(**kwargs)
    if train_set:
        features = train_set.iloc[:, :36].to_numpy()
        targets = train_set.iloc[:, 36].to_numpy()
        model.fit(features, targets)
    return model


def build_decisiontree(train_set=None, **kwargs):
    model = DecisionTreeClassifier(**kwargs)
    if train_set:
        features = train_set.iloc[:, :36].to_numpy()
        targets = train_set.iloc[:, :36].to_numpy()
        model.fit(features, targets)
    return model


def build_logisticregression(train_set=None, **kwargs):
    model = LogisticRegression(**kwargs)
    if train_set:
        features = train_set.iloc[:, :36].to_numpy()
        targets = train_set.iloc[:, :36].to_numpy()
        model.fit(features, targets)
    return model


def build_model(model_type, train_set=None, **kwargs):
    if model_type == "Random Forest":
        return build_rf(train_set, **kwargs)
    elif model_type == "Random Forest Bagging":
        return build_rf_bagging(train_set, **kwargs)
    elif model_type == "Decision Tree":
        return build_decisiontree(train_set, **kwargs)
    elif model_type == "Logistic Regression":
        return build_logisticregression(train_set, **kwargs)


def cross_validate(models, train_set, cv=10):
    if not models:
        print("No models.")
        return [], None, None
    print(f"Cross Validation of {len(models)} models.")
    print(f"Number of training observations: {len(train_set)}\n-----")
    index_fitted = [i for i in range(len(models)) if hasattr(models[i], "targets_")]
    if index_fitted:
        print(f"Warning: model {index_fitted} is fitted.") if len(index_fitted) == 1 else \
            print(f"Warning: models {index_fitted} are fitted.")

    features = train_set.iloc[:, :36]
    targets = train_set.iloc[:, 36]
    best_model = models[0]
    print(f" -> Cross validation of model 0: {models[0]}")
    best_accuracy = cross_val_score(models[0], features, targets, cv=cv)
    best_accuracy = sum(best_accuracy) / len(best_accuracy)
    accuracies = [best_accuracy]
    for i in range(1, len(models)):
        print(f" -> Cross validation of model {i}: {models[i]}")
        accuracy = cross_val_score(models[i], features, targets, cv=cv)
        accuracy = sum(accuracy) / len(accuracy)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = models[i]
    return accuracies, best_model, best_accuracy


def predict(model, data, fit=True):
    features = data.get_features(False)
    targets = data.get_targets(False)
    if fit:
        model.fit(data.get_features(), data.get_targets())
    prediction = model.predict(features)
    return accuracy_score(targets, prediction)
