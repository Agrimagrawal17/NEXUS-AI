import pandas as pd
from app.ml.target_detector import detect_target
from app.ml.task_detector import detect_task


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the semantic type of a dataset column.
    """

    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Numerical
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Object / text columns
    if series.dtype == "object":

        non_null = series.dropna()

        if len(non_null) == 0:
            return "unknown"

        # Try datetime detection
        converted = pd.to_datetime(
            non_null,
            errors="coerce"
        )

        datetime_ratio = converted.notna().mean()

        if datetime_ratio >= 0.8:
            return "datetime"

        # Check unique ratio
        unique_ratio = series.nunique() / max(len(series), 1)

        # Potential ID
        if unique_ratio >= 0.95:
            return "id"

        # High uniqueness usually indicates text
        if unique_ratio > 0.5:
            return "text"

        return "categorical"

    return "unknown"


def calculate_data_quality(df: pd.DataFrame) -> dict:
    """
    Calculate overall dataset quality information.
    """

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(df.isnull().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    if total_cells > 0:
        missing_percentage = (
            missing_cells / total_cells
        ) * 100
    else:
        missing_percentage = 0

    if len(df) > 0:
        duplicate_percentage = (
            duplicate_rows / len(df)
        ) * 100
    else:
        duplicate_percentage = 0

    quality_score = 100

    # Penalize missing values
    quality_score -= min(missing_percentage, 40)

    # Penalize duplicates
    quality_score -= min(duplicate_percentage, 20)

    quality_score = round(
        max(quality_score, 0),
        2
    )

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(
            duplicate_percentage,
            2
        ),
        "quality_score": quality_score
    }


def profile_dataset(file_path: str) -> dict:

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Only CSV and Excel files are supported."
        )

    columns = {}

    for column in df.columns:

        series = df[column]

        columns[column] = {
            "type": detect_column_type(series),
            "missing_values": int(
                series.isnull().sum()
            ),
            "unique_values": int(
                series.nunique()
            ),
            "percentage_missing": round(
                (
                    series.isnull().sum()
                    / len(df)
                ) * 100,
                2
            )
        }

    # Target detection
    target_info = detect_target(df)

    # Task detection
    task_info = None

    if target_info["target_found"]:
        target_column = target_info["target_column"]

        task = detect_task(
            df[target_column]
        )

        task_info = {
            "task": task
        }

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "columns_info": columns,
        "data_quality": calculate_data_quality(df),

        "target_detection": target_info,
        "task_detection": task_info
    }

    return profile