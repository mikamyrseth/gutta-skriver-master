from enum import Enum
from typing import Dict


class DataFrequency(Enum):
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"


class Dataseries(object):
    def __init__(self, name: str, description: str, publisher: str, frequency: DataFrequency, bbg_ticker: str, link: str):
        self.name = name
        self.description = description
        self.publisher = publisher
        self.frequency = frequency
        self.bbg_ticker = bbg_ticker
        self.link = link

    def __str__(self):
        return '   '.join("%s: %s\n" % item for item in vars(self).items())


class CustomDataseries(object):
    def __init__(self, name: str, page: str, weights: dict):
        self.name = name
        self.page = page
        self.weights = weights

    def __str__(self) -> str:
        return '   '.join("%s: %s\n" % item for item in vars(self).items())
