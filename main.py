from datetime import datetime
import itertools
from pandas import DatetimeIndex
from CustomTypes.Dataseries import *
from CustomTypes.Model import *
from CustomTypes.Prefixes import *
import json
import os
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.stats.mstats import zscore


def is_stationary(df: pd.DataFrame, cutoff=0.05) -> bool:
    pvalue = adfuller(df)[1]
    return pvalue < cutoff


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

    runSandbox = False
    runTest1 = True
    runTest2 = False
    runTest3 = False

    # Sandbox testing
    if runSandbox:
        for model in all_models:
            if model.name == "Benchmark ICE-BRENT long":
                model.frequency = DataFrequency.QUARTERLY
                print(model)

                _ = model.reestimate(
                    from_date=model.model_start_date,
                    to_date=model.model_end_date,
                )

                print(model)

                base_r2, adjusted_base_r2, base_std_err = model.run_model(
                    model.model_start_date, model.model_end_date)

                print(
                    f"interval: {model.model_start_date} - {model.model_end_date}")
                print(f"Base R2: {base_r2}")
                print(f"Adjusted Base R2: {adjusted_base_r2}")
                print(f"Base Std Err: {base_std_err}")

    # test 1 - Model reestimation
    if runTest1:
        for model in all_models:
            if True:  # model.name == "Akram (2022) Aggregate oil prices":
                # reestimate
                print("Running model:", model.name)
                old_coeffs = list(model.weights.values()).copy()[1:]
                old_coeffs_dict = model.weights.copy()
                lm, df = model.reestimate(
                    model.model_start_date, model.model_end_date)

                # calculate statistics
                new_coeffs = list(model.weights.values())[1:]

                if "Benchmark" in model.name:
                    error = 0
                    r2 = 1
                else:
                    error = mean_absolute_percentage_error(
                        old_coeffs, new_coeffs)
                    r2 = r2_score(old_coeffs, new_coeffs)

                # calculate new dict with deviance from old to new coeffs
                coeff_deviance = model.weights.copy()
                for key in coeff_deviance:
                    delta = coeff_deviance[key] - old_coeffs_dict[key]
                    deviance = abs(delta)/abs(old_coeffs_dict[key])
                    deviance = deviance.round(3)
                    coeff_deviance[key] = deviance

                X = df[list(model.weights.keys())]
                Y = df[model.dependent_variable]
                normalized_coefficients = lm.coef_ * X.std(axis=0)
                normalized_coefficients = normalized_coefficients.abs()
                prediction_r_2 = lm.score(X, Y)

                # extra -1 adjusts for alpha not being a parameter
                adjusted_r2 = 1 - \
                    (1-prediction_r_2)*(len(Y)-1)/(len(Y)-X.shape[1]-1-1)

                # Calculate standard error of residuals
                residuals = Y - lm.predict(X)
                std_error = residuals.std()

                # save stats
                model.results["test1"] = {}
                model.results["test1"]["Model"] = model.name
                model.results["test1"]["dependent variable"] = model.dependent_variable
                model.results["test1"][
                    "Prediction interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{model.model_end_date.strftime('%Y-%m-%d')}"
                model.results["test1"]["Top Coefficient"] = normalized_coefficients.idxmax(
                    axis=0)
                model.results["test1"]["old coefficients"] = old_coeffs_dict
                model.results["test1"]["new coefficients"] = model.weights
                model.results["test1"]["coefficient deviance"] = coeff_deviance
                model.results["test1"]["Model Similarity (R2)"] = round(r2, 3)
                model.results["test1"]["Model Deviance (MAPE)"] = round(
                    error, 3)
                model.results["test1"]["Prediction Strength (R2)"] = prediction_r_2.round(
                    3)
                model.results["test1"]["Adjusted Prediction Strength (Adjusted R2)"] = adjusted_r2.round(
                    3)
                model.results["test1"]["Standard Error of Residuals"] = std_error.round(
                    3)
                model.results["test1"]["Sample size"] = len(Y)

                # test_stationarity
                all_stationary = True
                not_stationary = []
                for col in model.weights.keys():
                    if "Random Walk" in model.name:
                        continue
                    if col == "ALPHA":
                        continue
                    stationary = is_stationary(df[col])
                    if not stationary:
                        all_stationary = False
                        not_stationary.append(col)
                model.results["test1"]["Stationary"] = all_stationary
                model.results["test1"]["Non-stationary variables"] = not_stationary

                # test coefficient significance
                all_significant = True
                not_significant = {}
                for col in model.weights.keys():
                    if "Random Walk" in model.name:
                        continue
                    if col == "ALPHA":
                        continue
                    # calculate p-value
                    est = sm.OLS(Y, X).fit()
                    p_value = est.pvalues[col]
                    # print(est.summary())
                    # print(f"{col} p-value: {p_value}")
                    significant = p_value < 0.05
                    if not significant:
                        all_significant = False
                        not_significant[col] = p_value
                model.results["test1"]["Significant"] = all_significant
                model.results["test1"]["Non-significant variables"] = not_significant

                # save df to xlsx with dates on format yyyy-mm-dd
                df.to_excel(
                    f'results/df/models/{model.name}.xlsx', index=True, index_label="Date")

    # test 2 - forward test
    if runTest2:
        for model in all_models:
            # calculate extended time period
            delta_time = model.model_end_date-model.model_start_date
            new_end_date = model.model_end_date + \
                timedelta(days=int(delta_time.days*0.15))

            base_r2, adjusted_base_r2, base_std_err = model.run_model(
                model.model_start_date, model.model_end_date)
            new_r2, adjusted_new_r2, new_std_err = model.run_model(
                model.model_start_date, new_end_date)
            delta_r2 = new_r2-base_r2
            delta_adjusted_r2 = adjusted_new_r2-adjusted_base_r2
            delta_std_err = new_std_err-base_std_err
            percentage_change = delta_r2/base_r2
            percentage_change_adjusted = delta_adjusted_r2/adjusted_base_r2
            percentage_change_std_err = delta_std_err/base_std_err

            model.results["test2"] = {}
            model.results["test2"]["Model"] = model.name
            model.results["test2"][
                "Base Interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{model.model_end_date.strftime('%Y-%m-%d')}"
            model.results["test2"]["Base R2"] = base_r2.round(3)
            model.results["test2"]["Base standard error of residuals"] = base_std_err.round(
                3)
            model.results["test2"]["Base Adjusted R2"] = adjusted_base_r2.round(
                3)
            model.results["test2"]["New R2"] = new_r2.round(3)
            model.results["test2"]["New Adjusted R2"] = adjusted_new_r2.round(
                3)
            model.results["test2"]["New standard error of residuals"] = new_std_err.round(
                3)
            model.results["test2"][
                "New interval"] = f"{model.model_start_date.strftime('%Y-%m-%d')}-{new_end_date.strftime('%Y-%m-%d')}"
            model.results["test2"]["Delta R2"] = delta_r2.round(3)
            model.results["test2"]["Delta Adjusted R2"] = delta_adjusted_r2.round(
                3)
            model.results["test2"]["Delta standard error of residuals"] = delta_std_err.round(
                3)
            model.results["test2"]["R2 percent change"] = percentage_change.round(
                3)
            model.results["test2"]["Adjusted R2 percent change"] = percentage_change_adjusted.round(
                3)
            model.results["test2"]["Standard error of residuals percent change"] = percentage_change_std_err.round(
                3)

    # test 3 - time intervals
    if runTest3:
        time_intervals = [(date(2003, 1, 1), date(2007, 1, 1)),
                          (date(2007, 1, 1), date(2011, 1, 1)),
                          (date(2011, 1, 1), date(2016, 1, 1)),
                          (date(2016, 1, 1), date(2021, 12, 30)),
                          (date(2003, 1, 1), date(2021, 12, 30)),
                          ]

        model.results
        for model in all_models:
            model.results["test3"] = []
            model.results["test3-by-interval"] = {}
            print("Running test 3 for model: ", model.name)

            if "short" in model.name:
                model.dependent_variable = "DELTA-NB-KKI"
            elif "long" in model.name:
                model.dependent_variable = "NB-KKI"
            else:
                print("neither long or short")
                print(1/0)

            frequencies = [DataFrequency.WEEKLY,
                           DataFrequency.MONTHLY, DataFrequency.QUARTERLY]
            for frequency in frequencies:
                model.frequency = frequency

                for start_date, end_date in time_intervals:
                    print("Running test 3 for model: ", model.name,
                          " and interval: ", start_date, end_date)

                    if "Random Walk" in model.name:
                        prediction_r_2, adjusted_prediction_r_2, std_error = model.run_model(
                            start_date, end_date)
                    else:
                        lm, df = model.reestimate(start_date, end_date)
                        X = df[list(model.weights.keys())]
                        Y = df[model.dependent_variable]
                        prediction_r_2 = lm.score(X, Y)

                        # extra -1 adjusts for alpha not being a parameter
                        adjusted_prediction_r_2 = 1 - \
                            (1-prediction_r_2)*(len(Y)-1) / \
                            (len(Y)-X.shape[1]-1-1)

                        # Calculate stanbdard error of residuals
                        residuals = Y-lm.predict(X)
                        std_error = residuals.std()

                    print("\tfound std error: ", std_error,
                          "Adj. r2 ", adjusted_prediction_r_2)

                    # print(df)
                    # print(list(model.weights.keys()))
                    # print(X)
                    normalized_coefficients = lm.coef_ * X.std(axis=0)
                    # normalized_coefficients = normalized_coefficients*100
                    normalized_coefficients = normalized_coefficients.abs()
                    normalized_coefficients = normalized_coefficients.round(3)
                    normalized_coefficients = normalized_coefficients.drop(
                        "ALPHA")
                    time_result = {}
                    time_result["Frequency"] = frequency.name
                    time_result["Model"] = model.name
                    # convert date to two digit year
                    time_result["Start Date"] = start_date.strftime(
                        '%Y-%m-%d')
                    time_result["Prediction interval"] = f"{start_date.strftime('%y')}-{end_date.strftime('%y')}"
                    time_result[f"R2"] = prediction_r_2.round(3)
                    time_result[f"Adjusted R2"] = adjusted_prediction_r_2.round(
                        3)
                    time_result[f"Standard error of residuals"] = std_error.round(
                        4)
                    # model.results[f"test3:{start_date}-{end_date}_coeffs"] = normalized_coefficients.to_dict()
                    time_result[f"Standardized coefficients"] = normalized_coefficients.to_dict(
                    )
                    model.results["test3"].append(time_result)
                    model.results["test3-by-interval"][f"{start_date}-{end_date}"] = time_result

                    # test_stationarity
                    all_stationary = True
                    not_stationary = []
                    for col in model.weights.keys():
                        if "Random Walk" in model.name:
                            continue
                        if col == "ALPHA":
                            continue
                        stationary = is_stationary(df[col])
                        if not stationary:
                            all_stationary = False
                            not_stationary.append(col)
                    model.results["test3-by-interval"][f"{start_date}-{end_date}"]["Stationary"] = all_stationary
                    model.results["test3-by-interval"][f"{start_date}-{end_date}"]["Non-stationary variables"] = not_stationary

                    # test coefficient significance
                    all_significant = True
                    not_significant = {}
                    for col in model.weights.keys():
                        if "Random Walk" in model.name:
                            continue
                        if col == "ALPHA":
                            continue
                        # calculate p-value
                        est = sm.OLS(Y, X).fit()
                        p_value = est.pvalues[col]
                        # print(est.summary())
                        # print(f"{col} p-value: {p_value}")
                        significant = p_value < 0.05
                        if not significant:
                            all_significant = False
                            not_significant[col] = p_value
                    model.results["test3-by-interval"][f"{start_date}-{end_date}"]["Significant"] = all_significant
                    model.results["test3-by-interval"][f"{start_date}-{end_date}"]["Non-significant variables"] = not_significant

                    # if on last interval
                    # if start_date == time_intervals[-1][0] and end_date == time_intervals[-1][1] and "Benchmark" not in model.name and frequency == DataFrequency.MONTHLY:
                    # X = zscore(X).drop("ALPHA", axis=1)
                    # Y = zscore(Y)
                    # print(X)
                    # print(Y)
                    # print(sm.OLS(Y, X).fit().summary())
                    # normalized_coefficients = abs(est.params.drop("ALPHA"))
                    # Plot coefficients
                    # normalized_coefficients.plot(kind="barh")

                    # round to 3 decimal places
                    # normalized_coefficients = normalized_coefficients.round(
                    #     3)

                    # Increase canvas size
                    # fig = plt.gcf()
                    # fig.set_size_inches(18.5, 10.5)

                    # ensure plot labels are showing
                    # plt.tight_layout()

                    # save plot
                    # fig.savefig(
                    #     f'results/normalized-coefficients/{model.name}.png')

                    # save nornalized coefficients to json
                    # with open(f'results/tables/coeffs_{model.name}.json', 'w') as fp:
                    #     json.dump(
                    #         normalized_coefficients.to_dict(), fp, indent=4)

                    # Save results
    if True:
        all_test_1 = []
        all_test_2 = []
        all_test_3 = []
        table_model_time_intervals = []
        table_recreation_performance = []
        table_frequency = []
        table_standardized_coefficients = {}
        all_test_3_by_interval = {}
        for model in all_models:
            with open("results/" + model.name + ".json", "w+") as outfile:
                json.dump(model.results, outfile, indent=4)
            if runTest1:
                if "Benchmark" not in model.name:
                    test_1 = model.results["test1"]
                    all_test_1.append(test_1)
                    interval_overlap = (model.model_end_date - model.model_start_date).days / (
                        model.original_end_date - model.original_start_date).days
                    interval_overlap = int(round(interval_overlap*100, 0))
                    table_recreation_performance.append(
                        {"Model": model.name,
                         "Model Deviance (MAPE)": test_1["Model Deviance (MAPE)"]
                         })
                    table_model_time_intervals.append(
                        {"Model": model.name,
                         "Original interval": f"{model.original_start_date.strftime('%Y-%m-%d')}-{model.original_end_date.strftime('%Y-%m-%d')}",
                         "Recreation interval": f"{model.model_start_date.strftime('%Y-%m-%d')}-{model.model_end_date.strftime('%Y-%m-%d')}",
                         "Overlap %": interval_overlap
                         })
                    table_frequency.append(
                        {"Model": model.name,
                         "Frequency": model.frequency.name,
                         "Sample size": test_1["Sample size"]
                         })
            if runTest2:
                all_test_2.append(model.results["test2"])
            if runTest3:
                stand_coeffs_model = []
                for time_result in model.results["test3"]:
                    all_test_3.append(time_result)
                    if "Benchmark" in model.name:
                        continue
                    if time_result["Frequency"] != DataFrequency.MONTHLY.name:
                        continue
                    coeffs = time_result["Standardized coefficients"]
                    coeffs["Interval"] = time_result["Prediction interval"]
                    # parse prediction interval
                    stand_coeffs_model.append(
                        coeffs)
                if "Benchmark" in model.name:
                    continue
                table_standardized_coefficients[model.name] = stand_coeffs_model
                for interval, time_result in model.results["test3-by-interval"].items():
                    if interval not in all_test_3_by_interval:
                        all_test_3_by_interval[interval] = []
                    all_test_3_by_interval[interval].append(time_result)

        if runTest1:
            with open("results/all_test_1.json", "w+") as outfile:
                json.dump(all_test_1, outfile, indent=4)
            with open("results/tables/table_recreation_performance.json", "w+") as outfile:
                json.dump(table_recreation_performance, outfile, indent=4)
            with open("results/tables/table_model_info.json", "w+") as outfile:
                json.dump(table_model_time_intervals, outfile, indent=4)
            with open("results/tables/table_frequency.json", "w+") as outfile:
                json.dump(table_frequency, outfile, indent=4)
        if runTest2:
            with open("results/all_test_2.json", "w+") as outfile:
                json.dump(all_test_2, outfile, indent=4)
        if runTest3:
            with open("results/all_test_3.json", "w+") as outfile:
                json.dump(all_test_3, outfile, indent=4)
            with open("results/all_test_3_by_interval.json", "w+") as outfile:
                json.dump(all_test_3_by_interval, outfile, indent=4)

            # dump all test_3_by_interval where frequency is MONTHLY
            big_interval = [
                res for res in all_test_3 if res["Prediction interval"] == "03-21"]
            results_w = [
                result for result in big_interval if result["Frequency"] == "WEEKLY"]
            results_m = [
                result for result in big_interval if result["Frequency"] == "MONTHLY"]
            results_q = [
                result for result in big_interval if result["Frequency"] == "QUARTERLY"]
            print("WEEKLY", results_w)
            print("MONTHLY", results_m)
            print("QUARTERLY", results_q)
            table_results_r2 = []
            table_results_std = []
            for result_m, result_q, result_w in zip(results_m, results_q, results_w):
                if result_m["Model"] != result_q["Model"]:
                    print(1/0)
                table_results_r2.append(
                    {"Model": result_m["Model"],
                     "adj R2_W": result_w["Adjusted R2"],
                     "adj R2_M": result_m["Adjusted R2"],
                     "adj R2_Q": result_q["Adjusted R2"],
                     })
                table_results_std.append(
                    {"Model": result_m["Model"],
                     "std_W": result_w["Standard error of residuals"],
                     "std_M": result_m["Standard error of residuals"],
                     "std_Q": result_q["Standard error of residuals"],
                     })

            with open("results/tables/table_results.json", "w+") as outfile:
                json.dump(table_results_r2, outfile, indent=4)

            table_results_r2_short = [
                res for res in table_results_r2 if "Benchmark" not in res["Model"] and "short" in res["Model"]]
            table_result_r2_short_benchmark = [
                res for res in table_results_r2 if "Benchmark" in res["Model"] and "short" in res["Model"]]
            table_result_r2_long = [
                res for res in table_results_r2 if "Benchmark" not in res["Model"] and "long" in res["Model"]]
            table_result_r2_long_benchmark = [
                res for res in table_results_r2 if "Benchmark" in res["Model"] and "long" in res["Model"]]

            table_results_std_short = [
                res for res in table_results_std if "Benchmark" not in res["Model"] and "short" in res["Model"]]
            table_result_std_short_benchmark = [
                res for res in table_results_std if "Benchmark" in res["Model"] and "short" in res["Model"]]
            table_result_std_long = [
                res for res in table_results_std if "Benchmark" not in res["Model"] and "long" in res["Model"]]
            table_result_std_long_benchmark = [
                res for res in table_results_std if "Benchmark" in res["Model"] and "long" in res["Model"]]

            with open("results/tables/table_results_r2_short.json", "w+") as outfile:
                json.dump(table_results_r2_short, outfile, indent=4)
            with open("results/tables/table_result_r2_short_benchmark.json", "w+") as outfile:
                json.dump(table_result_r2_short_benchmark, outfile, indent=4)
            with open("results/tables/table_result_r2_long.json", "w+") as outfile:
                json.dump(table_result_r2_long, outfile, indent=4)
            with open("results/tables/table_result_r2_long_benchmark.json", "w+") as outfile:
                json.dump(table_result_r2_long_benchmark, outfile, indent=4)

            with open("results/tables/table_results_std_short.json", "w+") as outfile:
                json.dump(table_results_std_short, outfile, indent=4)
            with open("results/tables/table_result_std_short_benchmark.json", "w+") as outfile:
                json.dump(table_result_std_short_benchmark, outfile, indent=4)
            with open("results/tables/table_result_std_long.json", "w+") as outfile:
                json.dump(table_result_std_long, outfile, indent=4)
            with open("results/tables/table_result_std_long_benchmark.json", "w+") as outfile:
                json.dump(table_result_std_long_benchmark, outfile, indent=4)

            for model, coeffs in table_standardized_coefficients.items():
                with open(f"results/tables/coeffs_standardized_{model}.json", "w+") as outfile:
                    json.dump(coeffs, outfile, indent=4)

            # table for bechmarks over time
            table_short_benchmark_over_time = [
                res for res in all_test_3 if "Benchmark" in res["Model"] and "short" in res["Model"]]
            table_short_benchmark_over_time = [
                res for res in table_short_benchmark_over_time if res["Frequency"] == "MONTHLY"]
            table_short_benchmark_over_time = [
                {
                    "Model": dict_["Model"],
                    dict_["Prediction interval"]: dict_["Adjusted R2"]
                } for dict_ in table_short_benchmark_over_time
            ]
            # combine all dicts in table_short_benchmark_over_time with same value of "Model"
            grouped = itertools.groupby(
                table_short_benchmark_over_time, key=lambda x: x["Model"])
            print(grouped)
            table_short_benchmark_over_time = [
                {k: v for d in g for k, v in d.items()} for _, g in grouped]
            print(table_short_benchmark_over_time)
            with open("results/tables/table_short_benchmark_over_time.json", "w+") as outfile:
                json.dump(table_short_benchmark_over_time, outfile, indent=4)

            # Plot the adjusted r2 for all long models for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Standard error of residuals"])
                # drop models in all_short_models
                df = df.drop(
                    [model.name for model in all_short_models], errors="ignore")

                # order by adjusted r2
                df = df.sort_values(by="Adjusted R2", ascending=False)

                df.plot.bar()
                plt.title(time_interval+" All Long Models")

                # color bars of models containing "Benchmark" in their label
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Adjusted R2"]):
                    plt.text(i, v, str(round(v, 3)), color='black',
                             fontweight='bold', ha="center")

                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/r2-intervals/{time_interval}_long.png")
                plt.close()

            # Plot the adjusted r2 for all short models for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Standard error of residuals"])
                # drop models in all_short_models
                df = df.drop(
                    [model.name for model in all_long_models], errors="ignore")

                # order by adjusted r2
                df = df.sort_values(by="Adjusted R2", ascending=False)

                df.plot.bar()

                # color bars of models containing "Benchmark" in the name
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Adjusted R2"]):
                    plt.text(i, v, str(round(v, 3)), color='black',
                             fontweight='bold', ha="center")

                plt.title(time_interval+" All Short Models")
                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/r2-intervals/{time_interval}_short.png")
                plt.close()

            # make a plot of the stand error for all long models for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Adjusted R2"])
                # drop models in all_short_models
                df = df.drop(
                    [model.name for model in all_short_models], errors="ignore")

                # order by standard error
                df = df.sort_values(by="Standard error of residuals")

                df.plot.bar()

                # color bars of models containing "Benchmark" in the name
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Standard error of residuals"]):
                    plt.text(i, v, str(round(v, 4)), color='black',
                             fontweight='bold', ha="center")

                plt.title(time_interval+" All Long Models")
                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/std-err-intervals/{time_interval}_long.png")
                plt.close()

            # make a plot of the stand error for all short models for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Adjusted R2"])
                # drop models in all_short_models
                df = df.drop(
                    [model.name for model in all_long_models], errors="ignore")

                # order by standard error
                df = df.sort_values(by="Standard error of residuals")

                df.plot.bar()

                # color bars of models containing "Benchmark" in the name
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Standard error of residuals"]):
                    plt.text(i, v, str(round(v, 4)), color='black',
                             fontweight='bold', ha="center")

                plt.title(time_interval+" All Short Models")
                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/std-err-intervals/{time_interval}_short.png")
                plt.close()

            # make a plot of count of non-stationary variables for all short model for for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Adjusted R2", "Standard error of residuals"])
                # transform list to count
                df["Non-stationary variables"] = df["Non-stationary variables"].apply(
                    lambda x: len(x))

                # drop long models
                df = df.drop(
                    [model.name for model in all_long_models], errors="ignore")

                # order by count
                df = df.sort_values(by="Non-stationary variables")

                df.plot.bar()

                # color bars of models containing "Benchmark" in the name
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Non-stationary variables"]):
                    plt.text(i, v, str(round(v, 0)), color='black',
                             fontweight='bold', ha="center")

                plt.title(time_interval+" All Short Models")
                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/stationarity/{time_interval}_short.png")
                plt.close()

            # make a plot of count of non-stationary variables for all long model for for each time interval
            for time_interval, time_results in all_test_3_by_interval.items():
                df = pd.DataFrame(time_results)
                df = df.set_index("Model")
                df = df.drop(
                    columns=["Prediction interval", "R2", "Top Coefficient", "Adjusted R2", "Standard error of residuals"])
                # transform list to count
                df["Non-stationary variables"] = df["Non-stationary variables"].apply(
                    lambda x: len(x))

                # drop short models
                df = df.drop(
                    [model.name for model in all_short_models], errors="ignore")

                # order by count
                df = df.sort_values(by="Non-stationary variables")

                df.plot.bar()

                # color bars of models containing "Benchmark" in the name
                for i, label in enumerate(df.index):
                    if "Benchmark" in label:
                        plt.gca().get_children()[i].set_color("red")

                # add data labels
                for i, v in enumerate(df["Non-stationary variables"]):
                    plt.text(i, v, str(round(v, 0)), color='black',
                             fontweight='bold', ha="center")

                plt.title(time_interval+" All Long Models")
                # Increase canvas size
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)

                fig.tight_layout()
                fig.savefig(
                    f"results/stationarity/{time_interval}_long.png")
                plt.close()

            # make a line plot of the stand error for all short models with time intervals as x asis
            df = pd.DataFrame(all_test_3)

            # pivot the data
            df = df.pivot(index="Prediction interval",
                          columns="Model", values="Standard error of residuals")

            # remove first interval
            df = df.drop(df.index[0])

            # remove long models
            df = df.drop(
                [model.name for model in all_long_models], axis=1, errors="ignore")

            # remove all benchmark models
            df = df.drop(
                [model.name for model in all_short_models if "Benchmark" in model.name], axis=1, errors="ignore")

            df.plot.line()

            plt.title("All Short Models")
            # Increase canvas size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)

            fig.tight_layout()
            fig.savefig(
                f"results/std-err-intervals/all_short.png")
            plt.close()

            # make a line plot of the stand error for all short benchmark models with time intervals as x asis
            df = pd.DataFrame(all_test_3)

            # pivot the data
            df = df.pivot(index="Prediction interval",
                          columns="Model", values="Standard error of residuals")

            # remove first interval
            df = df.drop(df.index[0])

            # remove long models
            df = df.drop(
                [model.name for model in all_long_models], axis=1, errors="ignore")

            # remove all benchmark models
            df = df.drop(
                [model.name for model in all_short_models if "Benchmark" not in model.name], axis=1, errors="ignore")

            df.plot.line()

            plt.title("All Short Models")
            # Increase canvas size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)

            fig.tight_layout()
            fig.savefig(
                f"results/std-err-intervals/all_short_bench.png")
            plt.close()

            # make a line plot of the adjusted r2 for all short models with time intervals as x asis
            df = pd.DataFrame(all_test_3)

            # pivot the data
            df = df.pivot(index="Prediction interval",
                          columns="Model", values="Adjusted R2")

            # remove first interval
            df = df.drop(df.index[0])

            # remove long models
            df = df.drop(
                [model.name for model in all_long_models], axis=1, errors="ignore")

            # remove all benchmark models
            df = df.drop(
                [model.name for model in all_short_models if "Benchmark" in model.name], axis=1, errors="ignore")

            df.plot.line()

            plt.title("All Short Models")
            # Increase canvas size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)

            fig.tight_layout()
            fig.savefig(
                f"results/r2-intervals/all_short.png")
            plt.close()

            # make a line plot of the adjusted r2 for all short becnhmark models with time intervals as x asis
            df = pd.DataFrame(all_test_3)

            # pivot the data
            df = df.pivot(index="Prediction interval",
                          columns="Model", values="Adjusted R2")

            # remove first interval
            df = df.drop(df.index[0])

            # remove long models
            df = df.drop(
                [model.name for model in all_long_models], axis=1, errors="ignore")

            # remove all benchmark models
            df = df.drop(
                [model.name for model in all_short_models if "Benchmark" not in model.name], axis=1, errors="ignore")

            df.plot.line()

            plt.title("All Short Models")
            # Increase canvas size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)

            fig.tight_layout()
            fig.savefig(
                f"results/r2-intervals/all_short_bench.png")
            plt.close()


load_json()
