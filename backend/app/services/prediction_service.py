import pandas as pd

from app.ml.model_store import load_model_artifact


# =========================================================
# PREDICTION SERVICE
# =========================================================

def predict_with_model(
    model_name,
    input_data
):
    """
    Load the complete NEXUS model artifact,
    reproduce the training-time preprocessing,
    and generate predictions.
    """

    # -----------------------------------------------------
    # Step 1: Load Complete Model Artifact
    # -----------------------------------------------------

    artifact = load_model_artifact(
        model_name
    )


    # -----------------------------------------------------
    # Step 2: Extract Artifact Components
    # -----------------------------------------------------

    model = artifact["model"]

    preprocessor = artifact["preprocessor"]

    feature_names = artifact["feature_names"]

    target_column = artifact["target_column"]

    task = artifact["task"]


    # -----------------------------------------------------
    # Step 3: Extract Preprocessing Metadata
    # -----------------------------------------------------

    preprocessing_metadata = artifact.get(
        "preprocessing_metadata"
    )


    # -----------------------------------------------------
    # Step 4: Validate Preprocessor
    # -----------------------------------------------------

    if preprocessor is None:

        raise ValueError(
            "Preprocessor is missing from model artifact"
        )


    # -----------------------------------------------------
    # Step 5: Convert Input Into DataFrame
    # -----------------------------------------------------

    if isinstance(
        input_data,
        dict
    ):

        input_data = [
            input_data
        ]


    df = pd.DataFrame(
        input_data
    )


    # -----------------------------------------------------
    # Step 6: Remove Target Column
    # -----------------------------------------------------

    if target_column in df.columns:

        df.drop(
            columns=[target_column],
            inplace=True
        )


    # -----------------------------------------------------
    # Step 7: Detect Date Columns
    # -----------------------------------------------------

    datetime_columns = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in [
                "date",
                "time",
                "datetime",
                "timestamp"
            ]
        ):

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = (
                converted.notna().mean()
            )

            if valid_ratio >= 0.8:

                datetime_columns.append(
                    column
                )

                df[column] = converted


    # -----------------------------------------------------
    # Step 8: Recreate Date Features
    # -----------------------------------------------------

    for column in datetime_columns:

        df[f"{column}_year"] = (
            df[column].dt.year
        )

        df[f"{column}_month"] = (
            df[column].dt.month
        )

        df[f"{column}_day"] = (
            df[column].dt.day
        )

        df[f"{column}_dayofweek"] = (
            df[column].dt.dayofweek
        )

        df.drop(
            columns=[column],
            inplace=True
        )


    # -----------------------------------------------------
    # Step 9: Recreate Training Input Columns
    # -----------------------------------------------------

    if preprocessing_metadata:

        numeric_columns = (
            preprocessing_metadata.get(
                "numeric_columns",
                []
            )
        )

        categorical_columns = (
            preprocessing_metadata.get(
                "categorical_columns",
                []
            )
        )

        required_input_columns = (
            list(numeric_columns)
            +
            list(categorical_columns)
        )

    else:

        raise ValueError(
            "Preprocessing metadata is missing from model artifact"
        )


    # -----------------------------------------------------
    # Step 10: Add Missing Input Columns
    # -----------------------------------------------------

    for column in required_input_columns:

        if column not in df.columns:

            df[column] = None


    # -----------------------------------------------------
    # Step 11: Keep Only Training Input Columns
    # -----------------------------------------------------

    df = df[
        required_input_columns
    ]


    # -----------------------------------------------------
    # Step 12: Apply Same Preprocessor
    # -----------------------------------------------------

    processed_data = preprocessor.transform(
        df
    )


    # -----------------------------------------------------
    # Step 13: Generate Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        processed_data
    )


    # -----------------------------------------------------
    # Step 14: Return Prediction
    # -----------------------------------------------------

    return {

        "model": artifact[
            "model_name"
        ],

        "task": task,

        "target_column": target_column,

        "predictions": predictions.tolist()
    }