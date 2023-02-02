import json
import os
from CustomTypes.Dataseries import CustomDataseries, Dataseries
from CustomTypes.Model import Model


def main():
    all_models = load_data()

    # get model where
    # model = next(
    # model for model in all_models if model.name == "Myrstuen&Sylte2020 long")

    # for all models where "benchmark not in name"
    run_models = [
        model for model in all_models if "benchmark" not in model.name.lower()]

    for model in run_models:
        # estimate from 2002-03-31 to 2014-08-31
        lm, df = model.reestimate(from_date="2002-03-31", to_date="2014-08-31")
        X = df[list(model.weights.keys())]
        Y = df[model.dependent_variable]
        prediction_r_2 = lm.score(X, Y)
        print(f"{model.name} IS R2: \t\t{prediction_r_2}")

        base_r2, adjusted_base_r2, base_std_err = model.run_model(
            from_date="2014-08-31", to_date="2020-01-31")

        # print results
        print(f"{model.name}  OOS R2: \t\t{base_r2}")

    """
    # Get model where name is "Benchmark ICE-BRENT short"
    model = next(
        model for model in all_models if model.name == "Symbolic regression selected")
    model.run_model(
        model.model_start_date, model.model_end_date)
    """


def load_data():
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

    # Data debugging
    if True:
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

    long_model_dirs = [
        "input/models_long",
        "input/models_benchmarks_long",
    ]

    short_model_dirs = [
        "input/models_short",
        "input/models_benchmarks_short",
    ]

    all_long_models: list[Model] = []
    all_short_models: list[Model] = []

    for model_dir in long_model_dirs:
        directory = model_dir
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print("Loading", f)
                model_json = open(f)
                model = json.load(model_json)
                model = Model(**model)
                model_json.close()
                all_long_models.append(model)

    for model_dir in short_model_dirs:
        directory = model_dir
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print("Loading", f)
                model_json = open(f)
                model = json.load(model_json)
                model = Model(**model)
                model_json.close()
                all_short_models.append(model)

    all_models = all_long_models + all_short_models
    return all_models


main()
