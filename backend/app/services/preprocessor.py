import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for column in df.columns:

        # Numerical column
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

        # Categorical / text column
        else:
            df[column] = df[column].fillna("Unknown")

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Remove completely duplicate rows
    df = df.drop_duplicates()

    return df


def handle_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for column in df.columns:

        # Try to detect datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[column]):

            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek

            # Remove original datetime column
            df = df.drop(columns=[column])

    return df

def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str
) -> dict:

    # Step 1: Handle missing values
    df = handle_missing_values(df)

    # Step 2: Remove duplicate rows
    df = remove_duplicates(df)

    # Step 3: Handle datetime columns
    df = handle_datetime_columns(df)

    # Step 4: Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return {
        "feature_columns": X.columns.tolist(),
        "target_column": target_column,
        "feature_count": len(X.columns),
        "target_count": len(y)
    }