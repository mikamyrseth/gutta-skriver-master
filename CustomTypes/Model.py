from ast import List
from datetime import date
import datetime
from typing import Dict
from numpy import inner
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
                 frequency: DataFrequency,
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
        self.coeffs[self.dependent_variable] = 0
        for series, weight in self.coeffs.items():
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

        for index, row in df.iterrows():
            prediction = 0
            for series, weight in self.coeffs.items():
                prediction += row[series]*weight

            df.loc[index, 'OUTPUT'] = prediction

        # AD hoc fasit
        # dep_series = Dataseries.get_dataseries("NB-KKI")
        #dep_df = dep_series.get_df(self.frequency, from_date, to_date)
        #dep_df = Prefixes.process_df("DELTA", dep_df, "NOIWTOT Index")
        #df["FASIT"] = dep_df

        print("PROCESSED MODEL!: ")
        print(df)
        # pd.set_option('display.max_columns', None)
        # pd.reset_option(“max_columns”)
        # print(df.head())
        return df['OUTPUT']


    def get_source_df(self, frequency: DataFrequency, from_date: datetime, to_date: datetime) -> DataFrame:
        df = DataFrame()
        for series, weight in self.coeffs.items():
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
        self.frequency = DataFrequency.get_frequency_enum(self.frequency)
        # recalculate relevant dataseries
        for series, weight in self.coeffs.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)
            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)
            dataseries.reestimate(from_date, to_date)

        # recalculate relevant dataseries
        for series, weight in self.coeffs.items():
            prefixes, source_series_name = Prefixes.process_prefixes(series)
            if Prefixes.CUSTOM in prefixes:
                dataseries = CustomDataseries.getCustomDataseries(
                    source_series_name)
            else:
                dataseries = Dataseries.get_dataseries(source_series_name)
            dataseries.reestimate(from_date, to_date)

        df = self.get_source_df(DataFrequency.MONTHLY, from_date, to_date)

        dep_prefix, dep_name = Prefixes.process_prefixes(self.dependent_variable)
        dep_series = Dataseries.get_dataseries(dep_name)
        dep_series_df = dep_series.get_df(self.frequency, from_date, to_date)
        dep_series_df = Prefixes.apply_prefixes(
                dep_prefix, dep_series_df, dep_name)


        df[self.dependent_variable] = dep_series_df
        print(df)
        print(list(self.coeffs.keys()))
        regression(df, list(self.coeffs.keys()), self.dependent_variable)


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
