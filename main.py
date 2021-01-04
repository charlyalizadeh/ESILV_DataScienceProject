from dataviz import DataViz
import matplotlib.pyplot as plt


def main():
    test = DataViz()
    test.import_train_set_from_txt("./sat.trn")
    test.import_test_set_from_txt("./sat.tst")
    pca, principal_components = test.get_pca(True, 6)
    #test.plot_scree(pca)
    #plt.show()
    #test.plot_explained_variance(pca)
    #plt.show()
    #test.plot_pca(pca, principal_components, test.train_set.iloc[:, 36])
    #plt.show()
    #test.plot_pixel_value(spectrum_index=[3])
    #plt.show()
    #test.plot_pixel_value_by_pixel_index()
    #test.plot_pixel_value_by_spectrum_index()
    #plt.show()


if __name__ == '__main__':
    main()
