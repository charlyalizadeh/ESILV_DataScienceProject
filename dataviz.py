import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataViz:
    def __init__(self, train_set=None, test_set=None):
        self.label = {1: 'Red soil',
                      2: 'Cotton crop',
                      3: 'Grey soil',
                      4: 'Damp grey soil',
                      5: 'Soil with vegetation stubble',
                      6: 'Mixture class',
                      7: 'Very damp grey soil'}
        self.train_set = self.convert_numpy_arr_to_df(train_set)
        self.test_set = self.convert_numpy_arr_to_df(test_set)

    def import_train_set_from_txt(self, path_train):
        self.train_set = self.convert_numpy_arr_to_df(np.loadtxt(path_train))

    def import_test_set_from_txt(self, path_test):
        self.test_set = self.convert_numpy_arr_to_df(np.loadtxt(path_test))

    def import_train_set(self, train_set):
        self.train_set = self.convert_numpy_arr_to_df(train_set)

    def import_test_set(self, test_set):
        self.test_set = self.convert_numpy_arr_to_df(test_set)

    def convert_numpy_arr_to_df(self, np_array):
        df = pd.DataFrame(
                data=np_array,
                columns=[f'p{int(p / 4)}_sp{p % 4}'
                         for p in range(36)] + ['Class']
                )
        df['Class'] = df['Class'].astype(int)
        df['Class'] = df['Class'].replace(self.label)
        return df

    def plot_spectrum(self, pixel_index=4, spectrum_index=range(4),
                      train=True, **kwargs):

        query_df = None
        if train:
            query_df = self.train_set[[
                f'p{pixel_index}_sp{sp}' for sp in spectrum_index
                ]]
        else:
            query_df = self.test_set[[
                f'p{pixel_index}_sp{sp}' for sp in spectrum_index
                ]]
        counts = [
                query_df[col].astype(int).value_counts()
                for col in query_df.columns
                ]
        for i in range(len(counts)):
            counts[i] = [counts[i][j] if j in counts[i] else 0
                         for j in range(256)]
        df = pd.DataFrame(dict(zip([f'p{pixel_index}_sp{sp}'
                                    for sp in spectrum_index], counts)))
        df['Pixel Value'] = range(256)
        palette = ['green', 'red', 'orange', 'navy']
        palette = palette[:len(spectrum_index)]
        return sns.lineplot(x='Pixel Value',
                            y='value',
                            hue='variable',
                            data=pd.melt(df, ['Pixel Value']),
                            **kwargs)

    def plot_line_pixel_value(self, pixel_index=range(8),
                              spectrum_index=range(4), train=True, **kwargs):
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

    def plot_bar_class(self, train=True):
        """Bar plot the number of observations by class.

        :param bool train: Boolean indicating whether to use the training
                           or testing set. (Default True)
        """

        counts = self.train_set['Class'].value_counts() \
            if train else \
            self.test_set['Class'].value_counts()
        for i in self.label.values():
            if i not in counts.index:
                counts.at[i] = 0
        ax = sns.barplot(x=counts.index, y=counts)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set(xlabel='Label',
               ylabel='No. of pixels labelled',
               title='Bar Plot of classes')
        return ax

    def get_pca(self, train=True, number_components=2):
        """

        source: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
        """
        df = self.train_set if train else self.test_set

        # Store the features values
        df_features = df.iloc[:, 0:36].values
        # Store the class values

        # Scale the features
        df_features = StandardScaler().fit_transform(df_features)

        # Apply PCA on features
        pca = PCA(n_components=number_components)
        principal_components = pca.fit_transform(df_features)
        return pca, principal_components

    def plot_explained_variance(self, pca):
        ax = plt.subplot()
        ax.bar(range(1, len(pca.explained_variance_) + 1),
               pca.explained_variance_)
        ax.set_ylabel('Explained variance')
        ax.set_xlabel('Components')
        ax.set_title('Explained variance Plot')
        ax.plot(
            range(1, len(pca.explained_variance_) + 1),
            np.cumsum(pca.explained_variance_),
            c='red',
            label='Cumulative Explained Variance',
        )
        plt.legend(loc='upper left')
        plt.tight_layout()
        return ax

    def plot_scree(self, pca):
        ax = plt.subplot()
        ax.plot(range(1, len(pca.explained_variance_) + 1),
                pca.explained_variance_)
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative explained variance')
        plt.tight_layout()
        return ax

    def plot_pca(self, pca, principal_components, classes):
        # Store the PCA values in a DataFrame
        principal_comp_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)]
        )

        # Add the target values and rename it
        final_df = pd.concat([principal_comp_df, classes], axis=1)

        # 2D projection
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('PCA with 2 Components', fontsize=20)

        targets = list(self.label.values())

        # Colors for our different labels
        colors = [
            'crimson',
            'bisque',
            'silver',
            'dimgrey',
            'limegreen',
            'royalblue',
            'blueviolet',
        ]

        # We plot with only the two first components
        for target, color in zip(targets, colors):
            indicesToKeep = final_df['Class'] == target
            ax.scatter(
                final_df.loc[indicesToKeep, 'PC1'],
                final_df.loc[indicesToKeep, 'PC2'],
                c=color,
                s=50,
            )

        ax.legend(targets)
        ax.grid()
        plt.tight_layout()
        return ax
