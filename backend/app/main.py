from fastapi import FastAPI, UploadFile, File
import pandas as pd

from app.services.prediction_service import predict_with_model
from app.services.data_profiler import profile_dataset

from app.ml.preprocessing import prepare_data
from app.ml.target_detector import detect_target
from app.ml.task_detector import detect_task

from app.ml.model_trainer import (
    train_regression_models,
    train_classification_models
)

import tempfile
import os


# =========================================================
# NEXUS AI APPLICATION
# =========================================================

app = FastAPI(
    title="NEXUS",
    description="Autonomous Machine Learning & AI Platform",
    version="1.0.0"
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "NEXUS AI is running"
    }


# =========================================================
# ANALYZE DATASET
# =========================================================

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(
        file.filename
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        content = await file.read()

        temp_file.write(
            content
        )

        temp_file_path = temp_file.name

    try:

        # -------------------------------------------------
        # Step 1: Profile Dataset
        # -------------------------------------------------

        profile = profile_dataset(
            temp_file_path
        )


        # -------------------------------------------------
        # Step 2: Load Dataset
        # -------------------------------------------------

        df = pd.read_csv(
            temp_file_path
        )


        # -------------------------------------------------
        # Step 3: Detect Target Column
        # -------------------------------------------------

        target = detect_target(
            df
        )

        target_column = target[
            "target_column"
        ]


        # -------------------------------------------------
        # Step 4: Detect ML Task
        # -------------------------------------------------

        task = detect_task(
            df[target_column]
        )


        # -------------------------------------------------
        # Step 5: Prepare Data
        # -------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            feature_names,
            preprocessing_metadata
        ) = prepare_data(
            df,
            target_column
        )


        # -------------------------------------------------
        # Step 6: Train Models
        # -------------------------------------------------

        if task == "regression":

            model_results = train_regression_models(

                X_train,

                X_test,

                y_train,

                y_test,

                preprocessor=preprocessor,

                feature_names=feature_names,

                target_column=target_column,

                task=task,
                
                preprocessing_metadata=preprocessing_metadata
            )


            # ---------------------------------------------
            # Step 7: Select Best Regression Model
            # ---------------------------------------------

            best_model = model_results["selection"]


        else:

            model_results = train_classification_models(
    X_train, X_test, y_train, y_test,
    preprocessor=preprocessor,
    feature_names=feature_names,
    target_column=target_column,
    
    task=task,
    preprocessing_metadata=preprocessing_metadata
)


            # ---------------------------------------------
            # Step 7: Select Best Classification Model
            # ---------------------------------------------

            best_model = model_results["selection"]


        # -------------------------------------------------
        # Step 8: Return Complete Analysis
        # -------------------------------------------------

        return {

            "filename": file.filename,

            "profile": profile,

            "target_detection": target,

            "task_detection": task,

            "preprocessing": {

                "training_rows": len(
                    X_train
                ),

                "testing_rows": len(
                    X_test
                ),

                "features": X_train.shape[1],

                "feature_names": feature_names.tolist(),

                "metadata": preprocessing_metadata
            },

            "model_results": model_results,

            "best_model": best_model
        }


    finally:

        # -------------------------------------------------
        # Delete Temporary File
        # -------------------------------------------------

        if os.path.exists(
            temp_file_path
        ):

            os.remove(
                temp_file_path
            )


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    data: dict
):

    try:

        # -------------------------------------------------
        # Get Model Name
        # -------------------------------------------------

        model_name = data.get("model_name")


        # -------------------------------------------------
        # Get Input Features
        # -------------------------------------------------

        input_data = data.get("features")


        # -------------------------------------------------
        # Validate Model Name
        # -------------------------------------------------

        if not model_name:

            return {
                "status": "failed",
                "message": "model_name is required"
            }


        # -------------------------------------------------
        # Validate Input Features
        # -------------------------------------------------

        if not input_data:

            return {
                "status": "failed",
                "message": "features are required"
            }


        # -------------------------------------------------
        # Generate Prediction
        # -------------------------------------------------

        result = predict_with_model(
            model_name,
            input_data
        )


        # -------------------------------------------------
        # Return Prediction
        # -------------------------------------------------

        return {
            "status": "success",
            **result
        }


    except Exception as error:

        return {
            "status": "failed",
            "error": str(error)
        }