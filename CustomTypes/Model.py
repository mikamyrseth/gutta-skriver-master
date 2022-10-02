from ast import List
from typing import Dict


class Model(object):
    def __init__(self, name: str,
                 authors: list[str],
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
