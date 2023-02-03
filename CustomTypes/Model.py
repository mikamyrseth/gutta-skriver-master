from ast import List
from datetime import date
import datetime
from typing import Dict
from numpy import inner
import pandas as pd
from sklearn.decomposition import PCA
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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pysr import PySRRegressor
import sympy as sp
import symfit
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt

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
        model, prediction = symbolic_regression(
            df, list(self.weights.keys()), self.dependent_variable)
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

    names = {"LAGGED-DELTA-LOG-NB-KKI": "LAG-KKI",
             "DELTA-CUSTOM-MYRSTUEN-INTEREST-DIFFERENCE-12M": "INTEREST-DIFF",
             "DELTA-LOG-ICE-BRENT": "OIL",
             "LAGGED-CUSTOM-MYRSTUEN-EQULIBIRUM": "EQULIBIRUM",
             }

    # replace column names
    X.columns = [names.get(x, x) for x in X.columns]

    # Add time column
    X["TIME"] = range(0, len(X))

    # Remove dashes from column names
    X.columns = [x.replace("-", "_") for x in X.columns]
    X.columns = [x.replace("&", "") for x in X.columns]

    variable_names = X.columns
    print(variable_names)

    # X = PCA(n_components=4).fit_transform(X)
    X = X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, shuffle=False)

    print("Train")
    print(X_train)
    print("Test")
    print(X_test)

    # X_validate, X_test, Y_validate, Y_test = train_test_split(
    # X_test, Y_test, test_size=0.5, shuffle=False)

    # X_train_con = pd.concat([X_train, X_validate])
    # Y_train_con = pd.concat([Y_train, Y_validate])

    # Scale data
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    # X_train = scaler_x.fit_transform(X_train)
    # Y_train = scaler_y.fit_transform(Y_train)

    # calculate linear r2 vaidation
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    linear_mse = mean_squared_error(Y_train, lr.predict(X_train))
    stopping_criteria = linear_mse*0.20
    print("Linear MSE: ", linear_mse)
    print("Stopping criteria: ", stopping_criteria)
    # Create model

    model = PySRRegressor(

        # populations=8,
        # ^ 2 populations per core, so one is always running.
        # population_size=200,
        niterations=1000000,  # < Increase me for better results
        maxsize=30,  # default 20
        # adaptive_parsimony_scaling=100,  # default 20
        ncyclesperiteration=1000,  # dedfault 550
        # procs=20,
        multithreading=True,
        # populations=40*2,
        turbo=False,
        binary_operators=[
            "+",
            "-",
            "/",
            "*",
            "pow",
            # "mod",
            # "greater",
            # "pow",
            # "coeff(x, y) = x*y"
        ],
        complexity_of_constants=1,
        complexity_of_variables=1,
        select_k_features=18,
        # precision=64,
        constraints={
            'mult': (4, 4),
            "/": (4, 4),
            # "pow": (2, 2),
            # 'coeff': (1, 3),
            'cube': 4,
            'square': 4,
            'square_abs': 4,
            'sqrt': 4,
            'abs': 4,
            'exp': 4,
            'erf': 4,
            'cerf': 4,
            'log': 4,
            # 'isnegative': 2,
            # 'ispositive': 2,
        },
        nested_constraints={
            "mult": {
                "+": 0,
                "-": 0,
            },
            "/": {
                "+": 0,
                "-": 0,
            },
            "sqrt": {
                "sqrt": 0,
                "square": 0,
                "square_abs": 0,
            },
            "square": {
                "sqrt": 0,
                "square": 0,
                "square_abs": 0,
            },
            "square_abs": {
                "sqrt": 0,
                "square": 0,
                "square_abs": 0,
            },
        },
        # parsimony=0.0002,  # 0.0032
        # turbo=True,
        warm_start=True,
        unary_operators=[
            "neg",
            "square",
            "cube",
            "exp",
            "abs",
            "sqrt",
            "log",
            "erf",
            "erfc",
            # "relu",
            # "round",
            # "floor",
            # "ceil",
            # "gamma",
            # "sign",
            # "sqrt",
            # "log",
            # "abs",
            # "cube",
            # "pow",
            # "square",
            "square_abs(x) = x*abs(x)",
            # sqrt_abs = sqrt(x)"
            # "cos",
            # "exp",
            # "isnegative(x) = (1-abs(x)/x)/2",
            # "ispositive(x) = (abs(x)/x+1)/2",
            # "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        complexity_of_operators={
            # "mult": 1,
            # "/": 1,
            # "coeff": 0,
            # "+": 1,
            # "-": 1,
            # "sqrt": 0.1,
            # "cube": 1,
            # "square": 1,
            # "squaresign": 1,
            # "abs": 1,
            # "exp": 1,
            # "isnegative": 1,
            # "ispositive": 1,
        },
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
            # "coeff": lambda x, y: x * y,
            # "isnegative": lambda x: (1 - abs(x) / x) / 2,
            # "ispositive": lambda x: (abs(x) / x + 1) / 2,
            "square_abs": lambda x: x * abs(x),
        },
        # ^ Define operator for SymPy as well
        loss="L2DistLoss()",
        early_stop_condition=f"f(loss, complexity) = (loss < {stopping_criteria}) && (complexity < 15)",
        # loss="loss(x, y) = abs(x - y)",
        # ^ Custom loss function (julia syntax)
        model_selection='best',
        temp_equation_file=False,
        tempdir="storage/users/mikam/",
        equation_file="storage/users/mikam/eqs.csv",
    )

    """
    model = PySRRegressor.from_file("hall_of_fame_2023-01-31_212025.384.pkl")
    model.set_params(extra_sympy_mappings={
        # "inv": lambda x: 1 / x,
        # "coeff": lambda x, y: x * y,
        # "isnegative": lambda x: x - abs(x),
        # "ispositive": lambda x: (abs(x) / x + 1) / 2,
        # "squaresign": lambda x: x * abs(x),
    },)
    model.warm_start = True
    # model.early_stop_condition=f"f(loss, complexity) = (loss < {stopping_criteria}) && (complexity < 15)",
    # model.adaptive_parsimony_scaling = 30
    # model.niterations = 1000000

    """

    model.set_params(
        population_size=75,  # default 33
        tournament_selection_n=23,  # default 10
        tournament_selection_p=0.8,  # default 0.86
        ncyclesperiteration=100,  # default 550
        parsimony=3.2e-3,  # default 0.0032
        fraction_replaced_hof=0.08,  # default 0.035
        optimizer_iterations=25,  # default 8
        crossover_probability=0.12,  # default 0.066
        weight_optimize=0.06,  # default 0.0
        populations=50,  # default 15
        adaptive_parsimony_scaling=20000.0,  # default 20
    )

    # model = PySRRegressor(niterations=1000000)
    model.fit(X_train, Y_train)

    # fill dict with equation indexes and 0
    equation_dict = {}
    for index, row in model.equations_.iterrows():
        equation_dict[index] = []

    # Cross validation

    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(df):
        # print(f"Cross validation from {train_index} to {test_index}")
        X_train_cv, X_validate_cv = X[train_index,
                                      :], X[test_index, :]
        Y_train_cv, Y_validate_cv = Y[train_index], Y[test_index]

        # print("Training")
        # print(X_train)
        # print(Y_train)
        # print("Validating")
        # print(Y_train)
        # print(Y_validate)

        for index, row in model.equations_.iterrows():
            jax_moddel = model.jax(index)
            jax_callable = jax_moddel['callable']
            # print("Jax callable")
            # print(jax_callable)
            jax_params = jax_moddel['parameters']
            # print("jax params")
            # print(jax_params)
            if index == 0:
                continue
            if len(jax_params) == 0:
                continue

            def loss(params, x, y):
                return jnp.mean((jax_callable(x, params) - y) ** 2)
            jax_params = jopt.minimize(
                fun=loss,
                x0=jax_params,
                args=(X_train_cv, Y_train_cv),
                method="BFGS",
                tol=0.001,
            ).x

            # print("new params")
            # print(jax_params)

            # print("Numpy format")
            # print(X_validate.to_numpy())

            prediction = np.nan_to_num(jax_callable(X_validate_cv, jax_params))
            # print("Prediction")
            # print(prediction)
            r2 = r2_score(Y_validate_cv, prediction)
            print("Processed equation: ", index, " with score: ", r2)
            # append result
            equation_dict[index].append(r2)

    # convert scores into averages
    for key, value in equation_dict.items():
        if len(value) == 0:
            equation_dict[key] = 0
            continue
        equation_dict[key] = sum(value) / len(value)

    print("Equations:")
    for index, row in model.equations_.iterrows():
        eq = row["equation"]
        prediction_os = np.nan_to_num(model.predict(X_test, index))
        r2_os = r2_score(Y_test, prediction_os)
        prediction_is = np.nan_to_num(model.predict(X_train, index))
        r2_is = r2_score(Y_train, prediction_is)
        print(
            f"Equation {index} total score: {equation_dict[index]} is score: {r2_is} and OOS score: {r2_os}: {eq}")

    best_equation = max(equation_dict, key=equation_dict.get)
    best_equation_eq = model.equations_.loc[best_equation]["equation"]
    best_score = equation_dict[best_equation]
    print(f"Best equation: {best_equation} with score: {best_score}")

    prediction = np.nan_to_num(model.predict(X_test, best_equation))
    r2 = r2_score(Y_test, prediction)
    print(f"Best equation R2 in test sample: {r2}")

    """
    Y_pred_lr = lr.predict(X_validate)
    r2_score_lr = r2_score(Y_validate, Y_pred_lr)
    print(f"Linear Regression: R2 validation: {r2_score_lr}")

    best_equation = 0
    best_equation_eq = 0
    best_r2 = 0
    alt_best_equation = 0
    alt_best_equation_eq = 0
    alt_best_r2 = 0
    for i, row in model.equations_.iterrows():
        eq = row["equation"]
        Y_pred_oos = model.predict(X_validate, i)
        y_pred_super_oos = model.predict(X_test, i)
        r2_score_oos = r2_score(Y_validate, Y_pred_oos)
        r2_score_super_oos = r2_score(Y_test, y_pred_super_oos)
        print(
            f"Equation {i}: R2 validation: {r2_score_oos}, ({r2_score_super_oos}))")
        if r2_score_oos > best_r2:
            best_r2 = r2_score_oos
            best_equation = i
            best_equation_eq = eq
        if r2_score_oos > r2_score_lr:
            alt_best_r2 = r2_score_oos
            alt_best_equation = i
            alt_best_equation_eq = eq
    print(f"Best equation: {best_equation}: {best_equation_eq}")
    print(f"Alt best equation: {alt_best_equation}: {alt_best_equation_eq}")
    print("Using, best")

    # recalculate constants of best equation
    X_train = pd.concat([X_train, X_validate])
    Y_train = pd.concat([Y_train, Y_validate])
    best_equation_eq_sympy = model.equations_.loc[best_equation,
                                                  "sympy_format"]
    best_equation_eq_lambda = model.equations_.loc[best_equation,
                                                   "lambda_format"]

    print("Best equation sympy")
    print(best_equation_eq_sympy)
    print("Best equation lambda")
    print(best_equation_eq_lambda)

    jax_moddel = model.jax(best_equation)
    print("Jax")
    print(jax_moddel)
    jax_callablle = jax_moddel['callable']
    jax_params = jax_moddel['parameters']
    print("Jax callable")
    print(jax_callablle)
    print("Jax params")
    print(jax_params)

    # convert to numpy
    def loss(params, x, y):
        return jnp.mean((jax_callablle(x, params) - y) ** 2)
    jax_params = jopt.minimize(
        fun=loss,
        x0=jax_params,
        args=(X_train_con.to_numpy(), Y_train_con.to_numpy()),
        method="BFGS",
        tol=0.000001
    ).x

    print("Reestimate Jax params")
    print(jax_params)

    # Make predictions
    # X_test = scaler_x.transform(X_test)
    # Y_test = scaler_y.transform(Y_test)
    Y_pred_is = model.predict(X_train, best_equation)
    Y_pred_oos = model.predict(X_test, best_equation)

    # Evaluate model
    print("Symbolic In sample train only")
    print("R2:  ", r2_score(Y_train, Y_pred_is))
    print("MSE: ", mean_squared_error(Y_train, Y_pred_is))
    print("MAE: ", mean_absolute_error(Y_train, Y_pred_is))

    print("Symbolic Out of Sample")
    print("R2:  ", r2_score(Y_test, Y_pred_oos))
    print("MSE: ", mean_squared_error(Y_test, Y_pred_oos))
    print("MAE: ", mean_absolute_error(Y_test, Y_pred_oos))

    prediction = jax_callablle(X_train_con.to_numpy(), jax_params)
    print("Symbolic+JAX In sample train with validate")
    print("R2: ", r2_score(Y_train_con, prediction))
    print("MSE: ", mean_squared_error(Y_train_con, prediction))
    print("MAE: ", mean_absolute_error(Y_train_con, prediction))

    prediction = jax_callablle(X_test.to_numpy(), jax_params)
    print("Symbolic+JAX Out of Sample")
    print("R2  ", r2_score(Y_test, prediction))
    print("MSE: ", mean_squared_error(Y_test, prediction))
    print("MAE: ", mean_absolute_error(Y_test, prediction))

    # Linear regression

    # Remove Time
    X_train = X_train.drop(columns=["TIME"])
    X_test = X_test.drop(columns=["TIME"])

    lm = LinearRegression()
    # combine train and validation
    # X_train = X_train_con
    # Y_train = Y_train_con
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

    print("HIHIHI")
    """

    return model, best_equation


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
    model = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, verbose=1, max_depth=5)
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
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_squared_error'])

    # train_data = keras.preprocessing.sequence.TimeseriesGenerator(X_train, Y_train, length=1, batch_size=1)
    histroy = model.fit(X_train_window, Y_train_window, epochs=20,
                        verbose=1, validation_data=(X_test_window, Y_test_window))

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
    # reshape X
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
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_squared_error'])

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
    # reshape X
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
