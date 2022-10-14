from ast import List
from datetime import date
import datetime
from typing import Dict
import pandas as pd

from pandas import DataFrame
from CustomTypes.Dataseries import CustomDataseries, DataFrequency, Dataseries
from CustomTypes.Prefixes import Prefixes


class Model(object):
    def __init__(self, name: str,
                 authors: "list[str]",
                 publish_year: int,
                 page: str,
                 coeffs: dict,
                 model_start_date: str,
                 model_end_date: str,
                 dependent_variable: str,
                 frequency: str,
                 stds: dict,
                 stats: dict
                 ):
        self.name = name
        self.authors = authors
        self.publish_year = publish_year
        self.page = page
        self.coeffs = coeffs
        self.model_start_date = model_start_date
        self.model_end_date = model_end_date
        self.dependent_variable = dependent_variable
        self.frequency = frequency
        self.stds = stds
        self.stats = stats

    def __str__(self):
        return '   '.join("%s: %s\n" % item for item in vars(self).items())

    def get_dataseries(self) -> "list[Dataseries]":
        dataseries = set()
        for coeff in self.coeffs:
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
        self.frequency = DataFrequency.get_frequency_enum(self.frequency)
        df = DataFrame()
        for series, weight in self.coeffs.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)

            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)

            inner_df = dataseries.get_df(self.frequency, from_date, to_date)

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
            for series, weight in self.coeffs.items():
                prediction += row[series]*weight

            df.loc[index, 'OUTPUT'] = prediction

        # AD hoc fasit
        dep_series = Dataseries.get_dataseries("NB-KKI")
        dep_df = dep_series.get_df(self.frequency, from_date, to_date)
        #dep_df = Prefixes.process_df("DELTA", dep_df, "NOIWTOT Index")
        df["FASIT"] = dep_df

        print("PROCESSED MODEL!: ")
        print(df)
        # pd.set_option('display.max_columns', None)
        # pd.reset_option(“max_columns”)
        # print(df.head())
        return df['OUTPUT']
