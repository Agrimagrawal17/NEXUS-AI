import pandas as pd

from sklearn.model_selection import train_test_split


def prepare_data(df, target_column):
    """
    Prepare dataset for machine learning.
    """

    # Copy dataset
    data = df.copy()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Separate target and features
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # -----------------------------------------
    # Convert date/datetime columns
    # -----------------------------------------

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(X[column]):

            X[column + "_year"] = X[column].dt.year
            X[column + "_month"] = X[column].dt.month
            X[column + "_day"] = X[column].dt.day
            X[column + "_dayofweek"] = X[column].dt.dayofweek

            X = X.drop(columns=[column])

    # -----------------------------------------
    # Try detecting date columns stored as text
    # -----------------------------------------

    for column in X.columns:

        if X[column].dtype == "object":

            converted = pd.to_datetime(
                X[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.8:

                X[column + "_year"] = converted.dt.year
                X[column + "_month"] = converted.dt.month
                X[column + "_day"] = converted.dt.day
                X[column + "_dayofweek"] = converted.dt.dayofweek

                X = X.drop(columns=[column])

    # -----------------------------------------
    # Handle missing values
    # -----------------------------------------

    for column in X.columns:

        if X[column].dtype == "object":

            mode_value = X[column].mode()

            if not mode_value.empty:
                X[column] = X[column].fillna(
                    mode_value[0]
                )
            else:
                X[column] = X[column].fillna(
                    "Unknown"
                )

        else:

            X[column] = X[column].fillna(
                X[column].median()
            )

    # -----------------------------------------
    # Remove missing target rows
    # -----------------------------------------

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # -----------------------------------------
    # Remove high-cardinality categorical columns
    # -----------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        unique_ratio = X[column].nunique() / len(X)

        if unique_ratio > 0.5:

            X = X.drop(columns=[column])

    # -----------------------------------------
    # Convert categorical columns
    # -----------------------------------------

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # Convert boolean columns to numbers
    X = X.astype(float)

    # -----------------------------------------
    # Train/Test split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test