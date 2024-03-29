import datetime
from enum import Enum
from typing import Dict
import numpy as np
import math
from datetime import date, datetime

from pandas import DataFrame
import pandas as pd
from CustomTypes.Prefixes import Prefixes
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class DataFrequency(str, Enum):
    DAILY = "d"
    WEEKLY = "7d"
    MONTHLY = "M"
    QUARTERLY = "Q"

    def get_frequency_enum(str):
        match str:
            case "DAILY":
                return DataFrequency.DAILY
            case "WEEKLY":
                return DataFrequency.WEEKLY
            case "MONTHLY":
                return DataFrequency.MONTHLY
            case "QUARTERLY":
                return DataFrequency.QUARTERLY


class Dataseries(object):
    data = []

    def __init__(self, name: str, description: str, publisher: str, frequency: DataFrequency, bbg_ticker: str, link: str):
        self.name = name
        self.description = description
        self.publisher = publisher
        self.frequency = DataFrequency.get_frequency_enum(frequency)
        self.bbg_ticker = bbg_ticker
        self.link = link
        self.df = self.init_df()

    def __str__(self):
        return '   '.join("%s: %s\n" % item for item in vars(self).items())

    def get_dataseries(name: str) -> 'Dataseries':
        for series in Dataseries.data:
            if series.name == name:
                return series
        raise Exception("Could not get dataseries with name ", name)

    def init_df(self) -> DataFrame:
        # Generate ALPHA
        if self.name == "ALPHA":
            date_index = pd.date_range(
                date(1999, 12, 31), date(2022, 10, 21), freq=DataFrequency.DAILY)
            ones = np.ones(len(date_index))
            df = pd.DataFrame({'Date': date_index, 'ALPHA': ones})
            df = df.set_index('Date')
            return df

        # Generate DUMMY
        if "DUMMY" in self.name:
            date_index = pd.date_range(
                date(1999, 12, 31), date(2022, 10, 21), freq=DataFrequency.DAILY)
            zeros = np.zeros(len(date_index))
            df = pd.DataFrame({'Date': date_index, self.name: zeros})
            df = df.set_index('Date')

            # get dates from string on format "DUMMY-FROM-20080801-TO-20081231"
            dates = self.name.split("-")
            from_date = datetime.strptime(dates[2], "%Y%m%d")
            to_date = datetime.strptime(dates[4], "%Y%m%d")
            df.loc[from_date:to_date, self.name] = 1
            return df

        # Load
        try:
            if self.bbg_ticker == "NA":
                df = pd.read_excel("NON_BLOOMBERG_DATA.xls",
                                   sheet_name=self.name)
                df["Date"] = pd.to_datetime(
                    df['Date'], infer_datetime_format=True)
            else:
                df = pd.read_excel("IFOE1_DATA_221010.xlsx",
                                   sheet_name=self.bbg_ticker, header=None, names=["Date", self.name])

                df["Date"] = pd.to_datetime(
                    df['Date'], unit='D', origin='1899-12-30')

            # Process and cut
            df = df.set_index(['Date'])
            return df
        except Exception as e:
            print(
                f"***WARNING*** Data with name {self.name} could not be loaded: {e}")
            return DataFrame()

    def get_df(self, frequency: DataFrequency, from_date: datetime, to_date: datetime) -> DataFrame:
        df = self.df

        if df.index.min() > pd.Timestamp(from_date):
            warnings.warn(
                f"{self.name} does not have data from {from_date} first data point is {df.index.min()}")
            print(1/0)
        if df.index.max() < pd.Timestamp(to_date):
            warnings.warn(
                f"{self.name} does not have data from {to_date} last data point is {df.index.max()}")
            print(1/0)

        df = df.asfreq('D')
        df = df.interpolate()
        df = df.loc[from_date:to_date]
        df = df.resample(frequency.value, origin=from_date,).mean()

        if df.isnull().values.any():
            print(f"WARNING: {self.name} has NAN")
            print(df)

        return df

    def reestimate(self, from_date: datetime, to_date: datetime, frequency: DataFrequency):
        # print("Skipped reestimating ", self.name)
        return


class CustomSeriesType(Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"
    EXPONENT = "EXPONENT"


def get_custom_series_type_enum(str):
    match str:
        case "ADD":
            return CustomSeriesType.ADD
        case "MULTIPLY":
            return CustomSeriesType.MULTIPLY
        case "EXPONENT":
            return CustomSeriesType.EXPONENT


class CustomDataseries(object):
    data = []

    def __init__(self, name: str, page: str, weights: dict, type: CustomSeriesType, recalculate: bool = False, dependent_variable: str = None):
        self.name = name
        self.page = page
        self.weights = weights
        self.type = get_custom_series_type_enum(type)
        self.recalculate = recalculate
        self.dependent_variable = dependent_variable

    def __str__(self) -> str:
        return '   '.join("%s: %s\n" % item for item in vars(self).items())

    def getCustomDataseries(name: str) -> 'CustomDataseries':
        for series in CustomDataseries.data:
            if series.name == name or series.name == "CUSTOM-"+name:
                return series
        raise Exception("Could not get custom series with name ", name)

    def get_dataseries(self) -> "list[Dataseries]":
        dataseries = set()
        for weight in self.weights:
            prefixes, name = Prefixes.process_prefixes(weight)
            if Prefixes.CUSTOM in prefixes:
                custom_dataseries = CustomDataseries.getCustomDataseries(name)
                custom_source = custom_dataseries.get_dataseries()
                dataseries = dataseries.union(custom_source)
            else:
                dataseries_ = Dataseries.get_dataseries(name)
                dataseries.add(dataseries_)
        return dataseries

    def get_df(self, frequency: DataFrequency, from_date: datetime, to_date: datetime) -> DataFrame:
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

        if self.type == CustomSeriesType.ADD:
            for index, row in df.iterrows():
                prediction = 0
                for series, weight in self.weights.items():
                    prediction += row[series]*weight

                df.loc[index, self.name] = prediction

        elif self.type == CustomSeriesType.MULTIPLY:
            for index, row in df.iterrows():
                prediction = 1
                for series, weight in self.weights.items():
                    prediction *= row[series]*weight

                df.loc[index, self.name] = prediction

        elif self.type == CustomSeriesType.EXPONENT:
            for index, row in df.iterrows():
                prediction = 0
                for series, weight in self.weights.items():
                    prediction += row[series]**weight

                df.loc[index, self.name] = prediction

        # print("PROCESSED MODEL!: ")
        # print(df)

        # save to xlsx
        df.to_excel(
                    f'results/df/custom_dataseries/{self.name}.xlsx', index=True, index_label="Date")



        # pd.set_option('display.max_columns', None)
        # pd.reset_option(“max_columns”)
        # print(df.head())
        return df.loc[:, [self.name]]

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

    def reestimate(self, from_date: datetime, to_date: datetime, frequency: DataFrequency):
        # print("reestimating ", self.name)

        # recalculate relevant dataseries
        for series, weight in self.weights.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)
            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)
            dataseries.reestimate(from_date, to_date, frequency)

        if self.recalculate == False:
            # print("Skipped reestimating", self.name)
            return

        df = self.get_source_df(frequency, from_date, to_date)

        dep_prefix, dep_name = Prefixes.process_prefixes(
            self.dependent_variable)
        dep_series = Dataseries.get_dataseries(dep_name)
        dep_series_df = dep_series.get_df(frequency, from_date, to_date)
        dep_series_df = Prefixes.apply_prefixes(
            dep_prefix, dep_series_df, dep_name)

        df[self.dependent_variable] = dep_series_df
        # print(df)
        # print(list(self.weights.keys()))
        if df.isnull().values.any():
            print(f"WARNING: df has NAN")
            print(df[df.isna().any(axis=1)])
            print("END NAN")

        lm = regression(df, list(self.weights.keys()), self.dependent_variable)
        old_coeffs = list(self.weights.values()).copy()
        for index, key in enumerate(self.weights.keys()):
            self.weights[key] = lm.coef_[index]
        self.weights["ALPHA"] = lm.intercept_
        # print(f"Reestimated dataseries {self.name} to: ")

        new_coeffs = list(self.weights.values())
        # print("comparing")
        # print(old_coeffs)
        # print(new_coeffs)
        mse = mean_squared_error(old_coeffs, new_coeffs)
        r2 = r2_score(old_coeffs, new_coeffs)
        # print("dataseries deviance is (MSE)", mse)
        # print("R2", r2)
        # print(self)


def regression(df: pd.DataFrame, X_names: list[str], Y_name: str):

    X = df[X_names]
    Y = df[Y_name]

    # print(X)
    # print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.01, random_state=101)

    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    #print("STD", X_train.std(axis=0))
    #print("coef.", lm.coef_)
    #print("norm. coef ", lm.coef_* X_train.std(axis=0))
    print(X_names)
    print(lm.coef_)
    print(lm.intercept_)
    return lm

    # return lm.coef_ * X_train.std(axis=0)

    print(lm.coef_)

    prediction = lm.predict(X_test)
    plt.scatter(Y_test, prediction)
    plt.show()
