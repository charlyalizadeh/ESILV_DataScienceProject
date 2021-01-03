from dataviz import DataViz
from models import Model
import matplotlib.pyplot as plt


def main():
    model = Model()
    model.data.import_train_set_from_txt("./sat.tst")
    model.data.import_test_set_from_txt("./sat.trn")
    mymodel = model.build_rf_boostrsap()
    print(model.predict(mymodel))


if __name__ == '__main__':
    main()
