from ast import List
from datetime import date
import datetime
from typing import Dict
from numpy import inner
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pysr import PySRRegressor

from pandas import DataFrame
from CustomTypes.Dataseries import CustomDataseries, DataFrequency, Dataseries
from CustomTypes.Prefixes import Prefixes


class Model(object):
    def __init__(self, name: str,
                 authors: "list[str]",
                 publish_year: int,
                 page: str,
                 weights: dict,
                 model_start_date: str,
                 model_end_date: str,
                 dependent_variable: str,
                 frequency: DataFrequency,
                 stds: dict,
                 stats: dict,
                 lags: int = 0,
                 original_start_date: str = "1900-01-01",
                 original_end_date: str = "1901-01-01",
                 ):
        self.name = name
        self.authors = authors
        self.publish_year = publish_year
        self.page = page
        self.weights = weights
        self.original_start_date = datetime.datetime.fromisoformat(
            original_start_date)
        self.original_end_date = datetime.datetime.fromisoformat(
            original_end_date)
        self.model_start_date = datetime.datetime.fromisoformat(
            model_start_date)
        self.model_end_date = datetime.datetime.fromisoformat(model_end_date)
        self.dependent_variable = dependent_variable
        self.frequency = DataFrequency.get_frequency_enum(frequency)
        self.stds = stds
        self.stats = stats
        self.lags = lags
        self.results = {}

    def __str__(self):
        return '   '.join("%s: %s\n" % item for item in vars(self).items())

    def get_dataseries(self) -> "list[Dataseries]":
        dataseries = set()
        for coeff in self.weights:
            prefixes, name = Prefixes.process_prefixes(coeff)
            if Prefixes.CUSTOM in prefixes:
                custom_dataseries = CustomDataseries.getCustomDataseries(name)
                custom_source = custom_dataseries.get_dataseries()
                dataseries = dataseries.union(custom_source)
            else:
                dataseries_ = Dataseries.get_dataseries(name)
                dataseries.add(dataseries_)
        return dataseries

    def get_data(self):
        return

    def run_model(self, from_date: datetime, to_date: datetime):
        df = DataFrame()
        self.weights[self.dependent_variable] = 0
        for series, weight in self.weights.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)

            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)

            inner_df = dataseries.get_df(self.frequency, from_date, to_date)

            inner_df = Prefixes.apply_prefixes(
                prefixes, inner_df, source_series_name)

            df[series] = inner_df

        # Remove lagged rows
        df = df.iloc[self.lags:, :]

        print("mika var her")
        print(list(self.weights.keys()))
        model, prediction = symbolic_regression(df, list(self.weights.keys()), self.dependent_variable)
        return model, prediction

        for index, row in df.iterrows():
            prediction = 0
            for series, weight in self.weights.items():
                prediction += row[series]*weight

            df.loc[index, 'OUTPUT'] = prediction


        r2 = r2_score(df[self.dependent_variable].tolist(),
                      df["OUTPUT"].tolist())

        # calculate standard error of residuals
        residuals = df[self.dependent_variable] - df["OUTPUT"]
        std_err = residuals.std()

        # extra -1 adjusts for alpha not being a parameter
        adjusted_r2 = 1 - (1-r2)*(len(df)-1)/(len(df)-len(self.weights)-1-1)

        # remove key in dict
        self.weights.pop(self.dependent_variable, None)

        return r2, adjusted_r2, std_err

    def get_source_df(self, frequency: DataFrequency, from_date: datetime, to_date: datetime) -> DataFrame:
        df = DataFrame()
        for series, weight in self.weights.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)

            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)

            inner_df = dataseries.get_df(frequency, from_date, to_date)

            inner_df = Prefixes.apply_prefixes(
                prefixes, inner_df, source_series_name)

            df[series] = inner_df
        return df

    def reestimate(self, from_date: datetime, to_date: datetime):
        # recalculate relevant dataseries
        for series, weight in self.weights.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)
            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)
            # print("reestimating ", source_series_name)
            dataseries.reestimate(from_date, to_date, self.frequency)

        df = self.get_source_df(self.frequency, from_date, to_date)

        dep_prefix, dep_name = Prefixes.process_prefixes(
            self.dependent_variable)
        dep_series = Dataseries.get_dataseries(dep_name)
        dep_series_df = dep_series.get_df(self.frequency, from_date, to_date)
        dep_series_df = Prefixes.apply_prefixes(
            dep_prefix, dep_series_df, dep_name)

        df[self.dependent_variable] = dep_series_df

        # Remove lagged rows
        df = df.iloc[self.lags:, :]

        # print(df)
        # print(list(self.weights.keys()))
        if df.isnull().values.any():
            print(f"WARNING: df has NAN")
            print(df[df.isna().any(axis=1)])
            print("END NAN")

        lm = regression(df, list(self.weights.keys()), self.dependent_variable)

        for index, key in enumerate(self.weights.keys()):
            self.weights[key] = lm.coef_[index]
        self.weights["ALPHA"] = lm.intercept_
        # print(f"Reestimated model {self.name}")


        return lm, df

def create_windows(df, window_size=10):
    X = []
    for i in range(len(df) - window_size):
        X.append(df[i:i + window_size])
    X = np.array(X)
    return X

def symbolic_regression(df: pd.DataFrame, X_names: "list[str]", Y_name: str):
    # df.drop("ALPHA", axis=1, inplace=True)

    X = df[X_names]
    Y = df[Y_name]

    X.drop("ALPHA", axis=1, inplace=True)
    X.drop(Y_name, axis=1, inplace=True)

    #Remove dashes from column names
    X.columns = [x.replace("-", "_") for x in X.columns]


    print("names")
    print(X_names)
    print(Y_name)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, shuffle=False)

    # Scale data
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    # X_train = scaler_x.fit_transform(X_train)
    # Y_train = scaler_y.fit_transform(Y_train)

    # Create model
    model = PySRRegressor(
    niterations=50000,  # < Increase me for better results
    binary_operators=["+","-", "*", "/"],
    unary_operators=[
        "sqrt",
        # "log",
        "abs",
        "cube",
        # "pow"
        "square",
        # "cos",
        "exp",
        # "sin",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    model_selection = 'best'
    )

    model.fit(X_train, Y_train)

    print(model)

    # Make predictions
    # X_test = scaler_x.transform(X_test)
    # Y_test = scaler_y.transform(Y_test)
    Y_pred_is = model.predict(X_train)
    Y_pred_oos = model.predict(X_test)

    # Evaluate model
    print("Symbolic In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is))


    
    print("Symbolic Out of Sample")
    print("R2: ", r2_score(Y_test, Y_pred_oos))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos))

    # Linear regression
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    Y_pred_is_lm = lm.predict(X_train)
    print("Linear Regression In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is_lm))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is_lm))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is_lm))

    Y_pred_oos_lm = lm.predict(X_test)
    print("Linear Regression Out of Sample")
    print("R2: ", r2_score(Y_test, Y_pred_oos_lm))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos_lm))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos_lm))

    return model, Y_pred_oos_lm


def random_forrest_regression(df: pd.DataFrame, X_names: "list[str]", Y_name: str):
    # df.drop("ALPHA", axis=1, inplace=True)

    X = df[X_names]
    Y = df[Y_name]

    X.drop("ALPHA", axis=1, inplace=True)
    X.drop(Y_name, axis=1, inplace=True)

    print("names")
    print(X_names)
    print(Y_name)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, shuffle=False)

    # Scale data
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    # X_train = scaler_x.fit_transform(X_train)
    # Y_train = scaler_y.fit_transform(Y_train)

    # Create model
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=1, max_depth=5)
    model.fit(X_train, Y_train)

    # Make predictions
    # X_test = scaler_x.transform(X_test)
    # Y_test = scaler_y.transform(Y_test)
    Y_pred_is = model.predict(X_train)
    Y_pred_oos = model.predict(X_test)

    # Evaluate model
    print("Forrest In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is))


    
    print("Forrest Out of Sample")
    print("R2: ", r2_score(Y_test, Y_pred_oos))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos))

    # Linear regression
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    Y_pred_is_lm = lm.predict(X_train)
    print("Linear Regression In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is_lm))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is_lm))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is_lm))

    Y_pred_oos_lm = lm.predict(X_test)
    print("Linear Regression Out of Sample")
    print("R2: ", r2_score(Y_test, Y_pred_oos_lm))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos_lm))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos_lm))

    return model, Y_pred_oos_lm


def xgboost_regression(df: pd.DataFrame, X_names: "list[str]", Y_name: str):
    # df.drop("ALPHA", axis=1, inplace=True)

    
    X = df[X_names]
    Y = df[Y_name]

    X.drop("ALPHA", axis=1, inplace=True)
    X.drop(Y_name, axis=1, inplace=True)

    print("names")
    print(X_names)
    print(Y_name)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, shuffle=False)

    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    Y_train = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
    Y_test = scaler_y.transform(Y_test.values.reshape(-1, 1))

    # Create model
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Evaluate model
    Y_pred_is = model.predict(X_train)
    print("XGBoost In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is))


    Y_pred_oos = model.predict(X_test)
    print("XGBoost OOS")
    print("R2: ", r2_score(Y_test, Y_pred_oos))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos))

    # Linear regression
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    Y_pred_is_lm = lm.predict(X_train)
    print("Linear Regression In sample")
    print("R2: ", r2_score(Y_train, Y_pred_is_lm))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is_lm))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is_lm))

    Y_pred_oos_lm = lm.predict(X_test)
    print("Linear Regression OOS")
    print("R2: ", r2_score(Y_test, Y_pred_oos_lm))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos_lm))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos_lm))

    return model, Y_pred_oos


def lstm_regression2(df: pd.DataFrame, X_names: "list[str]", Y_name: str):
    # df.drop("ALPHA", axis=1, inplace=True)

    
    X = df[X_names]
    Y = df[Y_name]

    X.drop("ALPHA", axis=1, inplace=True)
    X.drop(Y_name, axis=1, inplace=True)

    print("names")
    print(X_names)
    print(Y_name)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, shuffle=False)

    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    Y_train = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
    Y_test = scaler_y.transform(Y_test.values.reshape(-1, 1))

    # Create windows
    window_size = 52
    X_train_window = create_windows(X_train, window_size)
    Y_train_window = Y_train[window_size:]
    X_test_window = create_windows(X_test, window_size)
    Y_test_window = Y_test[window_size:]

    X_train_lin = X_train[window_size:]
    Y_train_lin = Y_train[window_size:]
    X_test_lin = X_test[window_size:]
    Y_test_lin = Y_test[window_size:]

    print("X_train")
    print(X_train)
    print("Y_train")
    print(Y_train)

    model = keras.Sequential([
            # keras.layers.Input(batch_shape=(None, None, 1)),
            # keras.layers.LSTM(5, return_sequences=True),
            keras.layers.LSTM(50, input_shape=(window_size, 1)),
            # keras.layers.LSTM(10),
            # keras.layers.Dense(8),
            keras.layers.Dense(1),
        ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # train_data = keras.preprocessing.sequence.TimeseriesGenerator(X_train, Y_train, length=1, batch_size=1)
    histroy = model.fit(X_train_window, Y_train_window, epochs=20, verbose=1, validation_data=(X_test_window, Y_test_window))

    # Predict on test data
    predictions = model.predict(X_train_window)
    mse_lstm = mean_squared_error(Y_train_window, predictions)
    r2_lstm = r2_score(Y_train_window, predictions)
    oos_predictions = model.predict(X_test_window)
    oos_mse_lstm = mean_squared_error(Y_test_window, oos_predictions)
    oos_r2_lstm = r2_score(Y_test_window, oos_predictions)

    # compare with linear regression
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    #reshape X
    # X_train = np.array(X_train).reshape(-1, 1)
    # print(X_train)
    # print(Y_train)
    lr_model.fit(X_train_lin, Y_train_lin)
    predictions = lr_model.predict(X_train_lin)
    mse_lin = mean_squared_error(Y_train_lin, predictions)
    r2_lin = r2_score(Y_train_lin, predictions)
    oos_predictions = lr_model.predict(X_test_lin)
    oos_mse_lin = mean_squared_error(Y_test_lin, oos_predictions)
    oos_r2_lin = r2_score(Y_test_lin, oos_predictions)

    # Base mse
    base = np.mean(Y_train)
    base = np.repeat(base, len(Y_train))
    mse_base = mean_squared_error(Y_train, base)
    base = np.mean(Y_test)
    base = np.repeat(base, len(Y_test))
    oos_mse_base = mean_squared_error(Y_test, base)

    print('Base MSE: ', mse_base)
    print('Linear MSE: ', mse_lin)
    print('LSTM MSE: ', mse_lstm)
    print("")
    print('Base OOS MSE: ', oos_mse_base)
    print('Linear OOS MSE: ', oos_mse_lin)
    print('LSTM OOS MSE: ', oos_mse_lstm)
    print("")
    print('Linear R2: ', r2_lin)
    print('LSTM R2: ', r2_lstm)
    print("")
    print('Linear OOS R2: ', oos_r2_lin)
    print('LSTM OOS R2: ', oos_r2_lstm)

    return

def lstm_regression(df: pd.DataFrame, X_names: "list[str]", Y_name: str):
    # df.drop("ALPHA", axis=1, inplace=True)

    X = df[X_names]
    Y = df[Y_name]

    X.drop("ALPHA", axis=1, inplace=True)
    X.drop(Y_name, axis=1, inplace=True)

    print("names")
    print(X_names)
    print(Y_name)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=101)

    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    Y_train = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
    Y_test = scaler_y.transform(Y_test.values.reshape(-1, 1))

    print("X_train")
    print(X_train)
    print("Y_train")
    print(Y_train)

    model = keras.Sequential([
            # keras.layers.Input(batch_shape=(None, None, 1)),
            # keras.layers.LSTM(5, return_sequences=True),
            keras.layers.Dense(8, input_dim=4),
            keras.layers.Dense(4),
            keras.layers.Dense(2),
            keras.layers.Dense(1),
        ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # train_data = keras.preprocessing.sequence.TimeseriesGenerator(X_train, Y_train, length=1, batch_size=1)
    histroy = model.fit(X_train, Y_train, epochs=10000)

    # Predict on test data
    predictions = model.predict(X_train)
    mse_lstm = mean_squared_error(Y_train, predictions)
    oos_predictions = model.predict(X_test)
    oos_mse_lstm = mean_squared_error(Y_test, oos_predictions)

    # compare with linear regression
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    #reshape X
    # X_train = np.array(X_train).reshape(-1, 1)
    # print(X_train)
    # print(Y_train)
    lr_model.fit(X_train, Y_train)
    predictions = lr_model.predict(X_train)
    mse_lin = mean_squared_error(Y_train, predictions)
    oos_predictions = lr_model.predict(X_test)
    oos_mse_lin = mean_squared_error(Y_test, oos_predictions)

    # Base mse
    base = np.mean(Y_train)
    base = np.repeat(base, len(Y_train))
    mse_base = mean_squared_error(Y_train, base)
    base = np.mean(Y_test)
    base = np.repeat(base, len(Y_test))
    oos_mse_base = mean_squared_error(Y_test, base)

    print('Base MSE: ', mse_base)
    print('Linear MSE: ', mse_lin)
    print('LSTM MSE: ', mse_lstm)
    print("")
    print('Base OOS MSE: ', oos_mse_base)
    print('Linear OOS MSE: ', oos_mse_lin)
    print('LSTM OOS MSE: ', oos_mse_lstm)

    return

    return model, model


def regression(df: pd.DataFrame, X_names: "list[str]", Y_name: str) -> LinearRegression:

    X = df[X_names]
    Y = df[Y_name]

    # print(X)
    # print(Y)

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.01, random_state=101)

    lm = LinearRegression()
    lm.fit(X, Y)

    lm.normalize

    return lm

    # return lm.coef_ * X_train.std(axis=0)

    print(lm.coef_)

    prediction = lm.predict(X_test)
    plt.scatter(Y_test, prediction)
    plt.show()
