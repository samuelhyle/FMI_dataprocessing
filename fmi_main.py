import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():

    FMIRawData = pd.read_csv('FMIData_Assignment1.csv')

    print(FMIRawData.head(5))

    assert FMIRawData.shape == (1418, 11)

    print(FMIRawData.columns)

    data = FMIRawData.drop(['Time zone', 'Precipitation amount (mm)', 'Snow depth (cm)'], axis=1)


    data.columns = ['Year', 'Month', 'Day', 'Time', 'AirTemp', 'GroundMinTmp', 'MaxTemp', 'MinTemp']

    date_column = data["Year"].astype(str) + '-' + data["Month"].astype(str) + '-' + data["Day"].astype(str)
    data.insert(0, "Date", date_column)
    data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

    print(data.head())

    new_data = data[data['Time'] == '00:00']

    print("First five rows of the new FMI dataframe are as: \n", new_data.head())

    features = new_data['MinTemp']
    labels = new_data['MaxTemp']

    X = np.array(features).reshape(-1,1)
    y = np.array(labels)

    assert np.isclose(X[0, 0], -2.6), "Feature matrix value is incorrect"
    assert np.isclose(y[0], 3.4), "Label vector value is incorrect"

    assert X.shape == (713, 1), "X is incorrect"
    assert y.shape == (713,), "y is incorrect"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(data['MinTemp'],
                    data['MaxTemp'])
    axes[0].set_xlabel("mintemp", size=15)
    axes[0].set_ylabel("maxtemp", size=15)
    axes[0].set_title("mintemp vs maxtemp ", size=15)

    axes[1].hist(data['MaxTemp'])
    axes[1].set_title('distribution of maxtemp', size=15)
    axes[1].set_ylabel("count of datapoints", size=15)
    axes[1].set_xlabel("maxtemp intervals", size=15)
    plt.show()


main()