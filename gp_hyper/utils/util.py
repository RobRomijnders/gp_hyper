from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def pandas_to_numpy(train, test_size=0.2):
    """
    Note that this function contains many BAD data science practices.
      * We actually drop all the non-numeric columns
      * We fill in the missing values using the means.

    However, we want this project to focus on Gaussian Processes, so we take these BAD steps for granted
    :param train:
    :param test_size:
    :return:
    """
    # Fill missing values with the mean per column
    # Note that this is BAD data science practise, but here we want to focus on the Gaussian Processes part
    train = train.fillna(train.mean())

    # Only use all numeric columns
    numeric_column_indicators = list(map(lambda x: pd.api.types.is_numeric_dtype(x), train.dtypes.tolist()))
    data = np.array(train)[:, numeric_column_indicators]
    data = data.astype(np.float32)
    X, y = data[:, :-1], data[:, -1]

    print('We have %.0f samples and %.0f numeric features' % X.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Calculate all the statistics
    mean_X, mean_y, std_X, std_y = np.mean(X_train, axis=0), \
                                   np.mean(y_train, axis=0), \
                                   np.std(X_train, axis=0), \
                                   np.std(y_train, axis=0)

    # Transform all variables
    def transform(X, mean, std):
        return (X-mean) / (std + 1E-9)
    return transform(X_train, mean_X, std_X), \
           transform(X_val, mean_X, std_X), \
           transform(y_train, mean_y, std_y), \
           transform(y_val, mean_y, std_y)





