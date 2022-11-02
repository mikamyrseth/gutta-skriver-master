from datetime import datetime
from pandas import DatetimeIndex
from CustomTypes.Dataseries import *
from CustomTypes.Model import *
from CustomTypes.Prefixes import *
import json
import os
from datetime import datetime


def load_json() -> "tuple[ list[Dataseries], list[CustomDataseries], list[Model]] ":
    # Load data
    all_dataseries = []
    directory = "input/dataseries"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print("Loading", f)
            dataseries_json = open(f)
            dataseries = json.load(dataseries_json)
            dataseries = list(map(lambda json: Dataseries(**json), dataseries))
            dataseries_json.close()
            all_dataseries = all_dataseries+dataseries
    Dataseries.data = all_dataseries

    # Check values
    for dataseries in all_dataseries:
        if dataseries.df.empty:
            print(f"{dataseries.name}: MISSING")
        else:
            print(
                f"{dataseries.name}:{dataseries.bbg_ticker} has data from {dataseries.df.index[0]} to {dataseries.df.index[-1]}")

    all_custom_dataseries = []
    directory = "input/custom_dataseries"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print("Loading", f)
            custom_dataseries_json = open(f)
            custom_dataseries = json.load(custom_dataseries_json)
            custom_dataseries = list(
                map(lambda json: CustomDataseries(**json), custom_dataseries))
            custom_dataseries_json.close()
            all_custom_dataseries = all_custom_dataseries + custom_dataseries
    CustomDataseries.data = all_custom_dataseries

    all_models: list[Model] = []
    directory = "input/models"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print("Loading", f)
            model_json = open(f)
            model = json.load(model_json)
            model = Model(**model)
            model_json.close()
            all_models.append(model)

    for model in all_models:
        if model.name == "Bjørnstad korttidsmodell":
            model.reestimate(model.model_start_date, model.model_end_date)
            # model.reestimate(model.model_end_date, date(2021, 12, 31))
            # model.reestimate(model.model_start_date, date(2021, 12, 31))
            # model.run_model(model.model_start_date, model.model_end_date)
            # model.run_model(model.model_end_date, date(2021, 12, 31))
            # model.run_model(model.model_start_date, date(2021, 12, 31))


load_json()
