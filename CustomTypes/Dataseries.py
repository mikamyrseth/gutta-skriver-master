from enum import Enum
from typing import Dict

from pandas import DataFrame
from CustomTypes.Prefixes import Prefixes


class DataFrequency(Enum):
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "QUARTERLY"


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

    def get_df(self, frequency: DataFrequency) -> DataFrame:
        # if større frequcny ---> agreggering, men hvordan???
        # if mindre frequency --> split data lineært elns...
        raise Exception("Not implemented")


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

    def get_dataseries(self) -> list[Dataseries]:
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
