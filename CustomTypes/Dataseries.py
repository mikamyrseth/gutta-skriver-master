import datetime
from enum import Enum
from typing import Dict
import numpy as np
import math

from pandas import DataFrame
import pandas as pd
from CustomTypes.Prefixes import Prefixes
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class DataFrequency(str, Enum):
    DAILY = "d"
    WEEKLY = "W"
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

    def __str__(self):
        return '   '.join("%s: %s\n" % item for item in vars(self).items())

    def get_dataseries(name: str) -> 'Dataseries':
        for series in Dataseries.data:
            if series.name == name:
                return series
        raise Exception("Could not get dataseries with name ", name)

    def get_df(self, frequency: DataFrequency, from_date: datetime, to_date: datetime) -> DataFrame:
        # if større frequcny ---> agreggering, men hvordan???
        # if mindre frequency --> split data lineært elns...

        # Generate ALPHA
        if self.name == "ALPHA":
            date_index = pd.date_range(
                from_date, to_date, freq=frequency.value)
            ones = np.ones(len(date_index))
            df = pd.DataFrame({'Date': date_index, 'ALPHA': ones})
            df = df.set_index('Date')
            return df

        

        #Load
        if self.bbg_ticker == "NA":
            print(f"Attempting to find non-BBG data with name {self.name}")
            df = pd.read_excel("NON_BLOOMBERG_DATA.xls", sheet_name=self.name)
            df["Date"] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            print(df)
        else:
            print(
                f"Attempting to find data {self.name} with ticker {self.bbg_ticker}")
            df = pd.read_excel("IFOE1_DATA_221010.xlsx",
                            sheet_name=self.bbg_ticker, header=None, names=["Date", self.name])

            df["Date"] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
        
        # Process and cut
        df = df.set_index(['Date'])
        df = df.resample(frequency.value).ffill()
        df = df.loc[from_date:to_date]
        if df.loc[from_date:from_date].empty:
            warnings.warn(
                f"Series {self.name} does not have data from {to_date}. First data is {df.iloc[0]}")
        if df.loc[to_date:to_date].empty:
            warnings.warn(
                f"Series {self.name} does not have data to {to_date}. Last data is {df.iloc[-1]}")

        if df.isnull().values.any():
            print(f"WARNING: {self.name} has NAN")
            print(df)

        return df

    def reestimate(self, from_date: datetime, to_date: datetime):
        return


class CustomSeriesType(Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"


class CustomDataseries(object):
    data = []

    def __init__(self, name: str, page: str, weights: dict, type: CustomSeriesType, recalculate: bool = False, dependent_variable: str = None):
        self.name = name
        self.page = page
        self.weights = weights
        self.type = CustomSeriesType
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

        for index, row in df.iterrows():
            prediction = 0
            for series, weight in self.weights.items():
                prediction += row[series]*weight

            df.loc[index, self.name] = prediction


        print("PROCESSED MODEL!: ")
        print(df)
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

    def reestimate(self, from_date: datetime, to_date: datetime):
        if(not self.recalculate):
            return

        # recalculate relevant dataseries
        for series, weight in self.weights.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)
            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)
            dataseries.reestimate(from_date, to_date)

        df = self.get_source_df(DataFrequency.MONTHLY, from_date, to_date)
        df[self.dependent_variable] = Dataseries.get_dataseries(
            self.dependent_variable).get_df(DataFrequency.MONTHLY, from_date, to_date)
        print(df)
        print(list(self.weights.keys()))
        regression(df, list(self.weights.keys()), self.dependent_variable)
        return


def regression(df: pd.DataFrame, X_names: list[str], Y_name: str):

    X = df[X_names]
    Y = df[Y_name]

    print(X)
    print(Y)

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
    return lm.coef_

    # return lm.coef_ * X_train.std(axis=0)

    print(lm.coef_)

    prediction = lm.predict(X_test)
    plt.scatter(Y_test, prediction)
    plt.show()
