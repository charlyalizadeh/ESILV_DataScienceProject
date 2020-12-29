import cProfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


class Model:
    def __init__(self, train_set=None, test_set=None):
        self.train_set = train_set
        self.test_set = test_set
        sns.set_theme(style="darkgrid")

    def import_test_set(self, path_test):
        self.test_set = np.loadtxt(path_test)

    def import_train_set(self, path_train):
        self.train_set = np.loadtxt(path_train)

    def plot_spectrum(self, pixel_index=16, spectrum_index=(0, 1, 2, 3), train=True, **kwargs):
        spectrum_tot = {0:'green',
                        1:'red',
                        2:'indra_red1',
                        3:'infra_red2'}
        spectrum = {key:spectrum_tot[key] for key in spectrum_index}
        mat_value = self.train_set if train else self.test_set
        values = {}
        for index in spectrum_index:
            unique, counts = np.unique(mat_value[:,pixel_index + index], return_counts=True)
            temp_dict = defaultdict(lambda: 0)
            temp_dict.update(dict(zip(unique, counts)))
            values[spectrum_tot[index]] = [temp_dict[i] for i in range(256)]
        values['index'] = range(256)
        df = pd.DataFrame.from_dict(values)
        palette = ['green', 'red', 'orange', 'navy']
        palette = palette[:len(spectrum_index)]
        sns.lineplot(x='index',
                     y='value',
                     hue='variable',
                     data=pd.melt(df, ['index']),
                     palette=palette,
                     **kwargs)

def test():
    mymodel = Model()
    mymodel.import_train_set('sat.trn')
    mymodel.import_test_set('sat.tst')

    fig, axis = plt.subplots(ncols=2)
    mymodel.plot_spectrum(16, [0, 1, 2, 3], False, ax=axis[0])
    mymodel.plot_spectrum(16, [0, 1, 2, 3], ax = axis[1])
    plt.show()
