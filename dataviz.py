import cProfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    def pca (self, train = True, number_components = 5) :
        mat_value = self.train_set if train else self.test_set

        #Store the value in a DataFrame
        df = pd.DataFrame(data=mat_value, index = [i for i in range(len(mat_value))])
        
        #Store the features values
        df_features = df.iloc[:, 0:36].values
        #Store the target values
        df_target = df.iloc[:,36].values

        #Scale the features
        df_features = StandardScaler().fit_transform(df_features)

        #Apply PCA on features
        pca = PCA(n_components=number_components)
        principal_components = pca.fit_transform(df_features)

        #Store the PCA values in a DataFrame
        principal_comp_df = pd.DataFrame(data=principal_components, columns=['pc_1','pc_2','pc_3','pc_4','pc_5'])

        #Add the target values and rename it
        final_df = pd.concat([principal_comp_df, df.iloc[:,36]], axis = 1)
        final_df = final_df.rename(columns={36:"target"})

        print(final_df.head())
        

        # Explained variance plot
        plt.bar(range(1,len(pca.explained_variance_)+1),pca.explained_variance_)
        plt.ylabel("Explained variance")
        plt.xlabel("Components")
        plt.title("Explained variance Plot")
        plt.plot(range(1,len(pca.explained_variance_ )+1),
                np.cumsum(pca.explained_variance_),
                c='red',
                label="Cumulative Explained Variance")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        #Scree plot
        plt.plot(pca.explained_variance_)
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.tight_layout()
        plt.show()
        #First component is plotted at 0 in xscale


        #2D projection
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('PCA with 2 Components', fontsize = 20)

        targets =[1.0,2.0,3.0,4.0,5.0,6.0,7.0]
        #targets= ['Red soil','Cotton crop','Grey soil','Damp grey soil','Soil with vegetation stubble','Mixture class','Very damp grey soil']
        
        #colors for our different labels
        colors = ['crimson', 'bisque', 'silver', 'dimgrey', 'limegreen', 'royalblue', 'blueviolet']
        
        #We plot with only the two first components
        for target, color in zip(targets,colors):
            indicesToKeep = final_df['target'] == target
            ax.scatter(final_df.loc[indicesToKeep, 'pc_1']
               , final_df.loc[indicesToKeep, 'pc_2']
               , c = color
               , s = 50)

        ax.legend(targets)
        ax.grid()
        plt.tight_layout()
        plt.show()
        

def test_1 () :
    test = DataViz()
    test.import_train_set("sat.trn")
    test.bar_plot_label()


def test_2 () :
    test = DataViz()
    test.import_train_set("sat.trn")
    test.pca()

test_1()
test_2()