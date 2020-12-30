import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict


class DataViz:
    def __init__(self, train_set=None, test_set=None):
        self.train_set = train_set
        self.test_set = test_set
        sns.set_theme(style="whitegrid")

    def import_test_set(self, path_test):
        self.test_set = np.loadtxt(path_test)

    def import_train_set(self, path_train):
        self.train_set = np.loadtxt(path_train)

    def plot_spectrum(self,
                      pixel_index=4,
                      spectrum_index=(0, 1, 2, 3),
                      train=True,
                      **kwargs):
        pixel_index *= 4
        spectrum_tot = {0: 'green',
                        1: 'red',
                        2: 'indra_red1',
                        3: 'infra_red2'}
        mat_value = self.train_set if train else self.test_set
        values = {}
        for index in spectrum_index:
            unique, counts = np.unique(mat_value[:, pixel_index + index],
                                       return_counts=True)
            temp_dict = defaultdict(lambda: 0)
            temp_dict.update(dict(zip(unique, counts)))
            values[spectrum_tot[index]] = [temp_dict[i] for i in range(256)]
        values['index'] = range(256)
        df = pd.DataFrame.from_dict(values)
        palette = ['green', 'red', 'orange', 'navy']
        palette = palette[:len(spectrum_index)]
        return sns.lineplot(x='index',
                            y='value',
                            hue='variable',
                            data=pd.melt(df, ['index']),
                            palette=palette,
                            **kwargs)

    def plot_line_pixel_value(self,
                              pixel_index=range(8),
                              spectrum_index=[0, 1, 2, 3],
                              train=True,
                              **kwargs):
        x = []
        y = []
        h = []
        mat_value = self.train_set if train else self.test_set
        pixel_index = [p * 4 for p in pixel_index]
        for row in mat_value:
            for pindex in pixel_index:
                x.append(row[-1])
                y.append(sum(row[pindex + i] for i in spectrum_index) /
                         len(spectrum_index))
                h.append(int(pindex / 4))
        return sns.stripplot(x=x, y=y, hue=h)
