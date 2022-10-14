

class Dataspecification(object):
    def __init__(self, dataseries: Dataseries, coefficients:list, transformations:list[]):
        self.dataseries = dataseries
        self.coefficients = coefficients
        self.transormations = transformations
