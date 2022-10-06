import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy as np
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Columns(Enum):
    DATE = "Date"
    USDNOK = "TWI NKSP Index  (R1)"
    OIL = "CO1 Comdty  (L1)"
    STOCKS = "MXWO Index -  on 9/8/22  (R2)"


def regression(df: pd.DataFrame):

    X = df[[Columns.OIL.value, Columns.STOCKS.value]]
    Y = df[Columns.USDNOK.value]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.01, random_state=101)

    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    print("STD", X_train.std(axis=0))
    print("coef.", lm.coef_)
    print("norm. coef ", lm.coef_ * X_train.std(axis=0))
    return lm.coef_ * X_train.std(axis=0)

    print(lm.coef_)

    prediction = lm.predict(X_test)
    plt.scatter(Y_test, prediction)
    plt.show()


if __name__ == "__main__":
    # Load data from excel
    df = pd.read_excel("regression data.xlsx", sheet_name="2000-2022")

    # date_column = df[Columns.DATE.value]

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # first order differencing
    df = df.diff()
    df = df.dropna()

    # Undo date differencing
    # df[Columns.DATE.value] = date_column

    # Show
    # df.plot.scatter(x=Columns.STOCKS.value, y=Columns.USDEUR.value)
    # plt.sh-

    stocks = []
    oil = []

    coefs = regression(df)
    print("coeffs full data: ", coefs)

    # set to 1 means two year intervals, 2 is 3 etc.
    interval = 5

    # split data into years
    for i in range(2000, 2023, interval):
        # year_df = df[f"{i}-01-01":f"{i}-12-31"]
        year_df = df.sort_index(
        ).loc[f"{i}-01-01":f"{i+(interval-1)}-12-31", :]
        coefs = regression(year_df)
        print(f"{i}-{i+(interval-1)}, {coefs[0]}, {coefs[1]}")
        oil.append(coefs[0])
        stocks.append(coefs[1])
        break

    # print(oil)
    # print(stocks)
