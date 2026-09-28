from fastapi import FastAPI, UploadFile, File
import pandas as pd

from app.services.data_profiler import profile_dataset
from app.ml.preprocessing import prepare_data
from app.ml.target_detector import detect_target
from app.ml.task_detector import detect_task

from app.ml.model_trainer import (
    train_regression_models,
    train_classification_models,
    select_best_classification_model,
    select_best_regression_model
)

import tempfile
import os


app = FastAPI(
    title="NEXUS AI",
    description="Autonomous Machine Learning & AI Platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "NEXUS AI is running"
    }


@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        content = await file.read()

        temp_file.write(content)

        temp_file_path = temp_file.name

    try:

        # Step 1: Profile dataset
        profile = profile_dataset(
            temp_file_path
        )

        # Step 2: Load dataset as DataFrame
        df = pd.read_csv(
            temp_file_path
        )

        # Step 3: Detect target column
        target = detect_target(
            df
        )

        # Step 4: Detect ML task
        target_column = target["target_column"]

        task = detect_task(
            df[target_column]
        )

        # Step 5: Prepare data for machine learning
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            target_column
        )

        # Step 6: Train machine learning models
        if task == "regression":

            model_results = train_regression_models(
                X_train,
                X_test,
                y_train,
                y_test
            )

            # Step 7: Select best regression model
            best_model = select_best_regression_model(
                model_results
            )

        else:

            model_results = train_classification_models(
                X_train,
                X_test,
                y_train,
                y_test
            )

            # Step 7: Select best classification model
            best_model = select_best_classification_model(
                model_results
            )

        # Step 8: Return complete analysis
        return {

            "filename": file.filename,

            "profile": profile,

            "target_detection": target,

            "task_detection": task,

            "preprocessing": {
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "features": X_train.shape[1]
            },

            "model_results": model_results,

            "best_model": best_model
        }

    finally:

        os.remove(
            temp_file_path
        )