import cProfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


class DataViz:
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


    def bar_plot_label(self, label_index = 36, train=True) :
        #label_index is the 37th column in the training and testing set
        labels = {1 : 'Red soil',
                  2 : 'Cotton crop',
                  3 : 'Grey soil',
                  4 : 'Damp grey \n soil',
                  5 : 'Soil with \n vegetation stubble',
                  6 : 'Mixture\n class',
                  7 : 'Very damp\n grey soil'}
        
        mat_value = self.train_set if train else self.test_set
        
        for _ in range(len(mat_value)) :
            unique, counts = np.unique(mat_value[:,label_index], return_counts=True)
        
        #Add class 6 as there are no examples with class 6 in our dataset
        unique = np.insert(unique,5, 6.0)
        counts = np.insert(counts,5, 0)

        #Changing font of labels
        hfont = {'fontname' : 'Microsoft New Tai Lue'}

        #Plot barcharts
        fig = plt.figure(figsize=(10,5))
        plt.bar(unique,counts, color = 'blue', width=1)
        plt.xlabel("Label", **hfont)
        plt.ylabel("No. of pixels labelled", **hfont)
        plt.title("Bar Plot of classes")
        #Change xticks to plot labels names instead of numbers
        plt.xticks(unique, labels.values(), **hfont)
        
        plt.tight_layout()
        plt.show()
            

def test () :
    test = DataViz()
    test.import_train_set("sat.trn")
    test.bar_plot_label()

test()