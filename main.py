from CustomTypes.Dataseries import *
from CustomTypes.Model import *
from CustomTypes.Prefixes import *
import json


def load_json() -> tuple[list[Dataseries], list[CustomDataseries], list[Model]]:
    dataseries_json = open('dataseries.json')
    dataseries = json.load(dataseries_json)
    dataseries = list(map(lambda json: Dataseries(**json), dataseries))
    dataseries_json.close()

    custom_dataseries_json = open("custom_dataseries.json")
    custom_dataseries = json.load(custom_dataseries_json)
    custom_dataseries = list(
        map(lambda json: CustomDataseries(**json), custom_dataseries))
    custom_dataseries_json.close()

    models_json = open('models.json')
    models = json.load(models_json)
    models = list(map(lambda json: Model(**json), models))
    models_json.close()

    for model in models:
        print(model)


load_json()
