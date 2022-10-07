from enum import Enum


class Prefixes(Enum):
    CUSTOM = "CUSTOM"
    LOG = "LOG"
    DELTA = "DELTA"
    LAGGED = "LAGGED"

    def process_prefixes(name: str) -> tuple[list, str]:
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

        return prefixes_in_name, dataseries_name
