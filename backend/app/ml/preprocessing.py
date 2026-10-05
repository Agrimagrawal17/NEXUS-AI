import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer


def prepare_data(df, target_column):
    """
    Prepare dataset for machine learning.

    Returns:
        X_train_processed
        X_test_processed
        y_train
        y_test
        preprocessor
        feature_names
        preprocessing_metadata
    """

    # =====================================================
    # STEP 1: COPY DATA
    # =====================================================

    data = df.copy()

    data = data.drop_duplicates()

    # =====================================================
    # STEP 2: SEPARATE TARGET AND FEATURES
    # =====================================================

    X = data.drop(columns=[target_column])

    y = data[target_column]

    # =====================================================
    # STEP 3: REMOVE MISSING TARGET ROWS
    # =====================================================

    valid_rows = y.notna()

    X = X.loc[valid_rows]

    y = y.loc[valid_rows]

    # =====================================================
    # STEP 4: DETECT DATE COLUMNS
    # =====================================================

    datetime_columns = []

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(
            X[column]
        ):

            datetime_columns.append(column)

            continue

        if X[column].dtype == "object":

            column_name = column.lower()

            date_keywords = [
                "date",
                "time",
                "datetime",
                "timestamp"
            ]

            looks_like_date = any(
                keyword in column_name
                for keyword in date_keywords
            )

            if looks_like_date:

                converted = pd.to_datetime(
                    X[column],
                    errors="coerce",
                    format="mixed"
                )

                valid_ratio = (
                    converted.notna().mean()
                )

                if valid_ratio >= 0.8:

                    X[column] = converted

                    datetime_columns.append(
                        column
                    )

    # =====================================================
    # STEP 5: CREATE DATE FEATURES
    # =====================================================

    for column in datetime_columns:

        X[column + "_year"] = (
            X[column].dt.year
        )

        X[column + "_month"] = (
            X[column].dt.month
        )

        X[column + "_day"] = (
            X[column].dt.day
        )

        X[column + "_dayofweek"] = (
            X[column].dt.dayofweek
        )

        X = X.drop(
            columns=[column]
        )

    # =====================================================
    # STEP 6: REMOVE HIGH-CARDINALITY CATEGORICAL
    # =====================================================

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    columns_to_remove = []

    for column in categorical_columns:

        if len(X) == 0:
            continue

        unique_ratio = (
            X[column].nunique()
            / len(X)
        )

        if unique_ratio > 0.5:

            columns_to_remove.append(
                column
            )

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    # =====================================================
    # STEP 7: TRAIN / TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # STEP 8: IDENTIFY COLUMN TYPES
    # =====================================================

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # =====================================================
    # STEP 9: NUMERICAL PIPELINE
    # =====================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # =====================================================
    # STEP 10: CATEGORICAL PIPELINE
    # =====================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # =====================================================
    # STEP 11: COMBINE PREPROCESSING
    # =====================================================

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # =====================================================
    # STEP 12: FIT ONLY ON TRAINING DATA
    # =====================================================

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # =====================================================
    # STEP 13: TRANSFORM TEST DATA
    # =====================================================

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # =====================================================
    # STEP 14: GET FEATURE NAMES
    # =====================================================

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # =====================================================
    # STEP 15: CONVERT TO DATAFRAME
    # =====================================================

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    # =====================================================
    # STEP 16: FINAL SAFETY CHECK
    # =====================================================

    X_train_processed = (
        X_train_processed.astype(float)
    )

    X_test_processed = (
        X_test_processed.astype(float)
    )

    # =====================================================
    # STEP 17: PREPROCESSING METADATA
    # =====================================================

    preprocessing_metadata = {

        "original_feature_columns": (
            X.columns.tolist()
        ),

        "numeric_columns": (
            numeric_columns
        ),

        "categorical_columns": (
            categorical_columns
        ),

        "datetime_columns": (
            datetime_columns
        ),

        "removed_columns": (
            columns_to_remove
        ),

        "feature_names": (
            feature_names.tolist()
        )
    }

    # =====================================================
    # RETURN
    # =====================================================

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names,
        preprocessing_metadata
    )