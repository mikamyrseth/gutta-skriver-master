import datetime
from enum import Enum
from typing import Dict
import numpy as np

from pandas import DataFrame
import pandas as pd
from CustomTypes.Prefixes import Prefixes
import warnings
import logging


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
        self.frequency = frequency
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

        print(
            f"Attempting to find data {self.name} with ticker {self.bbg_ticker}")
        df = pd.read_excel("IFOE1_DATA_221010.xlsx",
                           sheet_name=self.bbg_ticker, header=None, names=["Date", self.bbg_ticker])

        df["Date"] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
        df = df.set_index(['Date'])
        df = df.resample(frequency.value).interpolate()
        df = df.loc[from_date:to_date]
        if df.loc[from_date:from_date].empty:
            warnings.warn(f"Series {self.name} does not have data from {to_date}. First data is {df.iloc[0]}")
        if df.loc[to_date:to_date].empty:
            warnings.warn(f"Series {self.name} does not have data to {to_date}. Last data is {df.iloc[-1]}")
        return df


class CustomSeriesType(Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"


class CustomDataseries(object):
    data = []

    def __init__(self, name: str, page: str, weights: dict, type: CustomSeriesType):
        self.name = name
        self.page = page
        self.weights = weights
        self.type = CustomSeriesType

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

            # Process prefixes in correct order
            prev_prefix = None
            for prefix in prefixes:
                if prev_prefix != None:
                    if prefix.isdigit():
                        inner_df = Prefixes.process_df(
                            prev_prefix, inner_df, source_series_name, prefix)
                        prev_prefix = None
                    else:
                        inner_df = Prefixes.process_df(
                            prev_prefix, inner_df, source_series_name
                        )
                        prev_prefix = prefix
            if prev_prefix != None:
                inner_df = Prefixes.process_df(
                    prev_prefix, inner_df, source_series_name
                )
            df[series] = inner_df

        for index, row in df.iterrows():
            prediction = 0
            for series, weight in self.weights.items():
                prediction += row[series]*weight

            df.loc[index, 'OUTPUT'] = prediction

        print("Calculated custom dataseries ", self.name)
        print(df)
        return df['OUTPUT']
