import pandas as pd


def convert_numpy_arr_to_df(np_array, classcol, label=None):
    """Convert a numpy array to a pandas DataFrame.

    Assumption about the numpy array:
        - The class column is the last one.
        - Any column with an index greater than classcol will be ignored.

    :param numpy.array np_array: Numpy array to convert.
    :param int classcol: The index of the class column.
    :param dict label: Dictionary containing the replacement labels. (Default None)
    """

    df = pd.DataFrame(
            data=np_array,
            columns=[f'p{int(p / 4)}_sp{p % 4}'
                     for p in range(classcol)] + ['Class']
            )
    df['Class'] = df['Class'].astype(int)
    if label is not None:
        df['Class'] = df['Class'].replace(label)
    return df
