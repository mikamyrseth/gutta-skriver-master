from datetime import datetime
from pandas import DatetimeIndex
from CustomTypes.Dataseries import *
from CustomTypes.Model import *
from CustomTypes.Prefixes import *
import json
import os
from datetime import datetime, timedelta


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

    runSandbox = False
    runTest1 = True
    runTest2 = True
    runTest3 = True
    

    # Sandbox testing
    if runSandbox:
        for model in all_models:
            if model.name == "Ellen linear model":
                lm, df = model.reestimate(date(2015, 1, 1), date(2020, 1, 1))
                X = df[list(model.weights.keys())]
                Y = df[model.dependent_variable]
                prediction_r_2 = lm.score(X, Y) 
                print("MODEL HAS R2 of: " , prediction_r_2)

    # test 1 - Model reestimation
    if runTest1:
        for model in all_models:
            if True:  # model.name == "Akram (2022) Aggregate oil prices":
                # reestimate
                print("Running model:", model.name)
                old_coeffs = list(model.weights.values()).copy()
                lm, df = model.reestimate(model.model_start_date, model.model_end_date)

                # calculate statistics
                new_coeffs = list(model.weights.values())
                r2 = r2_score(old_coeffs, new_coeffs)
                error = mean_absolute_percentage_error(old_coeffs, new_coeffs)

                X = df[list(model.weights.keys())]
                Y = df[model.dependent_variable]
                normalized_coefficients = lm.coef_ * X.std(axis=0)   
                normalized_coefficients = normalized_coefficients.abs()
                prediction_r_2 = lm.score(X, Y)      

                # Plot coefficients
                normalized_coefficients.plot(kind="barh")
                plt.savefig(f'normalized-coefficients-{model.name}.png')    
                
                # save stats
                model.results["test1"] = {}
                model.results["test1"]["Model"] = model.name
                model.results["test1"]["dependent variable"] = model.dependent_variable
                model.results["test1"]["Prediction interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{model.model_end_date.strftime('%Y-%m-%d')}"
                model.results["test1"]["Top Coefficient"] = normalized_coefficients.idxmax(axis=0)
                model.results["test1"]["Model Similarity (R2)"] = r2.round(3)
                model.results["test1"]["Model Deviance (MAPE)"] = error.round(3)
                model.results["test1"]["Prediction Strength (R2)"] = prediction_r_2.round(3)
                # model.results["test1"]["normalized_coefficients"] = normalized_coefficients.to_dict()

                # model.reestimate(model.model_end_date, date(2021, 12, 31))
                # model.reestimate(model.model_start_date, date(2021, 12, 31))
                # model.run_model(model.model_start_date, model.model_end_date)
                # model.run_model(model.model_end_date, date(2021, 12, 31))
                # model.run_model(model.model_start_date, date(2021, 12, 31))

    # test 2 - forward test
    if runTest2:
        for model in all_models:
            #calculate extended time period
            delta_time = model.model_end_date-model.model_start_date
            new_end_date = model.model_end_date+timedelta(days=int(delta_time.days*0.15))

            base_r2 = model.run_model(model.model_start_date, model.model_end_date)
            new_r2 = model.run_model(model.model_start_date, new_end_date)
            delta_r2 = new_r2-base_r2
            percentage_change = delta_r2/base_r2
            model.results["test2"] = {}
            model.results["test2"]["Model"] = model.name
            model.results["test2"]["Base Interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{model.model_end_date.strftime('%Y-%m-%d')}"
            model.results["test2"]["Base R2"] = base_r2.round(3)
            model.results["test2"]["New R2"] = new_r2.round(3)
            model.results["test2"]["New interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{new_end_date.strftime('%Y-%m-%d')}"
            model.results["test2"]["Delta R2"] = delta_r2.round(3)
            model.results["test2"]["R2 percent change"] = percentage_change.round(3)

    # test 3 - time intervals
    if runTest3:
        time_intervals = [(date(2002, 12, 31), date(2006, 1, 1)), 
        (date(2006, 1, 1), date(2010, 1, 1)), 
        (date(2010, 1, 1), date(2015, 1, 1)), 
        (date(2015, 1, 1), date(2020, 1, 1)), 
        (date(2020, 1, 1), date(2021, 12, 30)),
        (date(2002, 12, 31), date(2021, 12, 30))]

        for model in all_models:
            model.results["test3"] = []
            for start_date, end_date in time_intervals:
                lm, df = model.reestimate(start_date, end_date)
                X = df[list(model.weights.keys())]
                Y = df[model.dependent_variable]
                prediction_r_2 = lm.score(X, Y)   
                print(df)
                print(list(model.weights.keys()))
                print(X)
                normalized_coefficients = lm.coef_ * X.std(axis=0)   
                normalized_coefficients = normalized_coefficients.abs()
                time_result = {}
                time_result["Model"] = model.name
                time_result["Prediction interval"] = f"{start_date}-{end_date}"
                time_result[f"R2"] = prediction_r_2.round(3)
                # model.results[f"test3:{start_date}-{end_date}_coeffs"] = normalized_coefficients.to_dict()
                time_result[f"Top Coefficient"] = normalized_coefficients.idxmax(axis=0)
                model.results["test3"].append(time_result)
  

    # Save results
    if True:
        all_test_1 = []
        all_test_2 = []
        all_test_3 = []
        for model in all_models:
            with open("results/" + model.name + ".json", "w+") as outfile:
                json.dump(model.results, outfile, indent=4)
            all_test_1.append(model.results["test1"])
            all_test_2.append(model.results["test2"])
            for time_result in model.results["test3"]:
                all_test_3.append(time_result)

        with open("results/all_test_1.json", "w+") as outfile:
            json.dump(all_test_1, outfile, indent=4)
        with open("results/all_test_2.json", "w+") as outfile:
            json.dump(all_test_2, outfile, indent=4)
        with open("results/all_test_3.json", "w+") as outfile:
            json.dump(all_test_3, outfile, indent=4)


            # df = pd.DataFrame(data=model.results, index=[0])
            # df = (df.T)
            # print(df)

            # df = pd.concat([results, normalized_coefficients])

            # df.to_excel(f'results-{model.name}.xlsx')
            # with pd.ExcelWriter("results.xlsx", sheet_name=self.name, engine="openpyxl", mode="a", on_sheet_exists="replace") as writer:
            # df.to_excel(writer, sheet_name=self.name, index=False)
            # pd.write_excel(writer, df)

            # print(self)

            # save dict to file and format it nicely
            # with open("results/" + self.name + ".json", "w+") as outfile:
            #     json.dump(self.results, outfile, indent=4)
            

            # get key of dict with max value


load_json()
