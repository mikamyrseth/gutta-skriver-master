from enum import Enum
import enum
import numpy as np


class Prefixes(Enum):
    CUSTOM = "CUSTOM"
    LOG = "LOG"
    DELTA = "DELTA"
    LAGGED = "LAGGED"

    def process_prefixes(name: str) -> "tuple[list, str]":
        prefixes_in_name = []
        term_list = name.split("-")

        # Find matching prefixes
        for term in term_list:
            if term in [e.value for e in Prefixes]:
                prefixes_in_name.append(Prefixes[term])
            if term.isdigit() and int(term) < 10:
                prefixes_in_name.append(term)

        # Remove matches from dataseries name
        for prefix in prefixes_in_name:
            if isinstance(prefix, Prefixes):
                term_list.remove(prefix.value)
            else:
                term_list.remove(prefix)
        dataseries_name = "-".join(term_list)

        if Prefixes.CUSTOM in prefixes_in_name:
            dataseries_name = "CUSTOM-" + dataseries_name

        return prefixes_in_name, dataseries_name

    def process_df(prefix, df, column_name: str, step=1):
        print(f"Processed prefix for {column_name}, {prefix} with step {step}")
        print("Before:", df)
        if str(prefix).isdigit():
            return df
        match prefix:
            case Prefixes.CUSTOM:
                print("After:", df)
                return df
            case Prefixes.LOG:
                df[column_name] = np.log(df[column_name])
                print("After:", df)
                if df.isnull().values.any():
                    print(f"WARNING: Series has NAN")
                    print(df)
                return df
            case Prefixes.DELTA:
                df = df.diff(step)
                print("After:", df)
                return df
            case Prefixes.LAGGED:
                lagged = df[df.columns.values[0]].shift(step)
                df[df.columns.values[0]] = lagged
                print("After:", df)
                return df

    def apply_prefixes(prefixes: list, df, column_name: str):
        number_indexes = [i for i, x in enumerate(
            prefixes) if str(x).isdigit()]
        for i, prefix in enumerate(prefixes):
            if i+1 in number_indexes:
                df = Prefixes.process_df(
                    prefix, df, column_name, prefixes[i+1])
            else:
                df = Prefixes.process_df(prefix, df, column_name)
        return df
