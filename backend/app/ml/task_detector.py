import pandas as pd


def detect_task(
    target_series: pd.Series
) -> str:

    # Numerical target
    if pd.api.types.is_numeric_dtype(target_series):

        unique_values = target_series.nunique()

        # Few unique numerical values
        if unique_values <= 10:
            return "classification"

        # Many numerical values
        return "regression"

    # Non-numerical target
    return "classification"