import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datawrapper import DataWrapper


class DataViz:
    """Class used to visualize the data we work with

    :param DataWrapper data: DataWrapper object to work with.
    """
    def __init__(self, data=None):
        self.data = DataWrapper() if data is None else data

    def plot_spectrum(self, pixel_index=4, spectrum_index=range(4), train=True, **kwargs):
        """Plot the histogram of a specified data set.

        :param int pixel_index: Index corresponding to the pixel position in the 3x3 matrix. (Default 4)
        :param iterable spectrum_index: Indexes corresponding to the spectrum index. (Default range(4))
        :param bool train: Boolean specifying whether to use training or testing set. (Default True)
        :param dict **kwargs: Dictionary of keyword arguments sent to the seaborn plot function.
        """

        query_df = None
        if train:
            query_df = self.train_set[[f'p{pixel_index}_sp{sp}' for sp in spectrum_index]]
        else:
            query_df = self.test_set[[f'p{pixel_index}_sp{sp}' for sp in spectrum_index]]
        counts = [query_df[col].astype(int).value_counts() for col in query_df.columns]
        for i in range(len(counts)):
            counts[i] = [counts[i][j] if j in counts[i] else 0 for j in range(256)]
        df = pd.DataFrame(dict(zip([f'p{pixel_index}_sp{sp}'
                                    for sp in spectrum_index], counts)))
        max_pixel_value = int(max(query_df.max())) + 1
        df['Pixel Value'] = range(256)
        palette = ['green', 'red', 'orange', 'navy']
        palette = palette[:len(spectrum_index)]
        ax = sns.lineplot(x='Pixel Value', y='value', hue='variable', data=pd.melt(df, ['Pixel Value']), **kwargs)
        ax.set(xlim=(0, max_pixel_value))
        return ax

    def plot_spectrum_by_pixel_index(self, spectrum_index=range(4), train=True, **kwargs):
        fig, axs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                self.plot_spectrum(i * 3 + j,
                                   spectrum_index,
                                   train=train,
                                   ax=axs[i, j],
                                   **kwargs)
        return fig

    def plot_pixel_value(self, pixel_index=range(9), spectrum_index=range(4), train=True, labels=True, **kwargs):
        """Plot the pixel value in function of the pixel position and the spectrum.

        :param iterable pixel_index: Indexes corresponding to the pixels position in the 3x3 matrix. (Default range(9))
        :param iterable spectrum_index: Indexes corresponding to the spectrum index. (Default range(4))
        :param bool train: Boolean specifying whether to use training or testing set. (Default True)
        :param bool labels: Boolean specifying whether to display title and axes labels or not. (Default True)
        :param dict **kwargs: Dictionary of keyword arguments sent to the seaborn plot function.
        """

        df = self.train_set if train else self.test_set
        classes = df['Class'].tolist() * len(pixel_index)
        pixel_index = [p * 4 for p in pixel_index]
        means = []
        pixel_position = []
        for p in pixel_index:
            temp = df.iloc[:, [p + s for s in spectrum_index]].mean(axis=1)
            means.extend(temp.tolist())
            pixel_position.extend([int(p / 4) for i in range(len(df))])
        ax = sns.stripplot(x=classes, y=means, hue=pixel_position, **kwargs)
        if labels:
            ax.set(xlabel='Class',
                   ylabel='Pixel Value',
                   title='Plot of classes by pixel value and pixel position.')
        ax.legend(title='Pixel position')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.legend(loc='upper left')
        return ax

    def plot_pixel_value_by_pixel_index(self, spectrum_index=range(4)):
        """Plot side by side the pixel value in function of the pixel position.

        :param iterable spectrum_index: Indexes corresponding to the spectrum index. (Default range(4))
        """

        fig, axs = plt.subplots(3, 3)
        palette = sns.color_palette('husl', 9)
        for i in range(3):
            for j in range(3):
                self.plot_pixel_value([i * 3 + j],
                                      spectrum_index,
                                      labels=False,
                                      ax=axs[i, j],
                                      palette=[palette[i * 3 + j]]
                                      )
        return fig

    def plot_pixel_value_by_spectrum_index(self, pixel_index=range(9)):
        """Plot side by side the pixel value in function of the spectrum index

        :param iterable pixel_index: Indexes corresponding to the pixels position in the 3x3 matrix. (Default range(9))
        """

        fig, axs = plt.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                self.plot_pixel_value(pixel_index, [i * 2 + j], labels=False, ax=axs[i, j])
                return fig

    def plot_bar_class(self, train=True):
        """Bar plot the number of observations by class.

        :param bool train: Boolean indicating whether to use the training or testing set. (Default True)
        """

        counts = self.train_set['Class'].value_counts() if train else self.test_set['Class'].value_counts()
        for i in self.label.values():
            if i not in counts.index:
                counts.at[i] = 0
        ax = sns.barplot(x=counts.index, y=counts)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set(xlabel='Label',
               ylabel='No. of pixels labelled',
               title='Bar Plot of classes')
        return ax

    def get_pca(self, train=True, number_components=5):
        """Build and return a sklearn.decomposition.PCA object applied on the specified data set.

        source: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

        :param bool train: Wheter to use training set or testing set. (Default True)
        :param int number_components: Number of components used in the pca. (Default 5)
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
        """Plot the graph of explained variance.

        :param sklearn.decomposition.PCA pca: PCA object to plot the explained variance from.
        """

        ax = plt.subplot()
        ax.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
        ax.set_ylabel('Explained variance')
        ax.set_xlabel('Components')
        ax.set_title('Explained variance Plot')
        ax.plot(range(1, len(pca.explained_variance_) + 1),
                np.cumsum(pca.explained_variance_),
                c='red',
                label='Cumulative Explained Variance'
                )
        plt.legend(loc='upper left')
        plt.tight_layout()
        return ax

    def plot_scree(self, pca):
        """Plot the scree graph of a PCA.

        :param sklearn.decomposition.PCA pca: PCA object to plot the scree graph from.
        """

        ax = plt.subplot()
        ax.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative explained variance')
        plt.tight_layout()
        return ax

    def plot_pca(self, pca, principal_components, classes):
        """Plot the 2D visualization of a pca.

        :param sklearn.decomposition.PCA pca: PCA object to plot from.
        :param pandas.DataFrame principal_components: Principal components of the pca.
        :param iterable classes: List of unique class.
        """

        # Store the PCA values in a DataFrame
        principal_comp_df = pd.DataFrame(data=principal_components,
                                         columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

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
        colors = ['crimson',
                  'bisque',
                  'silver',
                  'dimgrey',
                  'limegreen',
                  'royalblue',
                  'blueviolet']

        # We plot with only the two first components
        for target, color in zip(targets, colors):
            indicesToKeep = final_df['Class'] == target
            ax.scatter(final_df.loc[indicesToKeep, 'PC1'],
                       final_df.loc[indicesToKeep, 'PC2'],
                       c=color,
                       s=50)

        ax.legend(targets)
        ax.grid()
        plt.tight_layout()
        return ax

    def __getattr__(self, key):
        if key == 'train_set':
            return self.data.train_set
        if key == 'test_set':
            return self.data.test_set
        if key == 'label':
            return self.data.label
