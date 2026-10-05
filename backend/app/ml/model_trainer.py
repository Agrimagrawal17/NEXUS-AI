import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from app.ml.model_store import save_model_artifact


# =========================================================
# SETTINGS
# =========================================================

MAX_TRAIN_ROWS = 20000

RANDOM_STATE = 42


# =========================================================
# SAFE MODEL NAME
# =========================================================

def make_safe_model_name(model_name):

    return (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# =========================================================
# DATA SAMPLING
# =========================================================

def sample_data(
    X,
    y,
    task
):

    total_rows = len(X)

    if total_rows <= MAX_TRAIN_ROWS:

        return X, y


    print(
        f"Large dataset detected: {total_rows} rows"
    )

    print(
        f"Sampling {MAX_TRAIN_ROWS} rows for training"
    )


    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if task == "classification":

        y_series = pd.Series(y)

        unique_classes = y_series.nunique()

        if unique_classes <= 20:

            sample_size_per_class = max(
                1,
                MAX_TRAIN_ROWS // unique_classes
            )

            sampled_indexes = []

            for class_value in y_series.unique():

                class_indexes = y_series[
                    y_series == class_value
                ].index

                selected = np.random.RandomState(
                    RANDOM_STATE
                ).choice(
                    class_indexes,
                    size=min(
                        sample_size_per_class,
                        len(class_indexes)
                    ),
                    replace=False
                )

                sampled_indexes.extend(
                    selected
                )

            sampled_indexes = sampled_indexes[
                :MAX_TRAIN_ROWS
            ]

            return (
                X.loc[sampled_indexes],
                y_series.loc[sampled_indexes]
            )


    # -----------------------------------------------------
    # Regression / fallback
    # -----------------------------------------------------

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indexes = rng.choice(
        len(X),
        size=MAX_TRAIN_ROWS,
        replace=False
    )

    return (
        X.iloc[indexes],
        y.iloc[indexes]
    )


# =========================================================
# REGRESSION MODELS
# =========================================================

def train_regression_models(

    X_train,
    X_test,
    y_train,
    y_test,

    preprocessor=None,
    feature_names=None,
    target_column=None,
    task="regression",
    preprocessing_metadata=None
):

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge":
            Ridge(),

        "Lasso":
            Lasso(),

        "ElasticNet":
            ElasticNet(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),

        "AdaBoost":
            AdaBoostRegressor(
                random_state=RANDOM_STATE
            ),

        "Hist Gradient Boosting":
            HistGradientBoostingRegressor(
                random_state=RANDOM_STATE
            )
    }


    # -----------------------------------------------------
    # Large Dataset Sampling
    # -----------------------------------------------------

    X_train_sampled, y_train_sampled = sample_data(
        X_train,
        y_train,
        "regression"
    )


    results = {}

    best_model = None

    best_model_name = None

    best_r2 = -np.inf


    # -----------------------------------------------------
    # Train Models
    # -----------------------------------------------------

    for name, model in models.items():

        try:

            model.fit(
                X_train_sampled,
                y_train_sampled
            )


            predictions = model.predict(
                X_test
            )


            mae = mean_absolute_error(
                y_test,
                predictions
            )


            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )


            r2 = r2_score(
                y_test,
                predictions
            )


            results[name] = {

                "status": "success",

                "MAE": float(mae),

                "RMSE": float(rmse),

                "R2": float(r2)
            }


            # -------------------------------------------------
            # Best Model
            # -------------------------------------------------

            if r2 > best_r2:

                best_r2 = r2

                best_model = model

                best_model_name = name


        except Exception as error:

            results[name] = {

                "status": "failed",

                "error": str(error)
            }


    # =====================================================
    # SAVE BEST MODEL ARTIFACT
    # =====================================================

    if best_model is not None:

        safe_name = make_safe_model_name(
            best_model_name
        )


        model_path = save_model_artifact(

            model=best_model,

            model_name=safe_name,

            preprocessor=preprocessor,

            feature_names=feature_names,

            target_column=target_column,

            task=task,
            preprocessing_metadata=preprocessing_metadata
        )


        print(
            f"Best regression model: {best_model_name}"
        )

        print(
            f"Best R2: {best_r2}"
        )

        print(
            f"Model artifact saved: {model_path}"
        )


    return results


# =========================================================
# CLASSIFICATION MODELS
# =========================================================

def train_classification_models(

    X_train,
    X_test,
    y_train,
    y_test,

    preprocessor=None,
    feature_names=None,
    target_column=None,
    task="classification"
):

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

        "AdaBoost":
            AdaBoostClassifier(
                random_state=RANDOM_STATE
            ),

        "Hist Gradient Boosting":
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

        "KNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "SVM":
            SVC()
    }


    # -----------------------------------------------------
    # Large Dataset Sampling
    # -----------------------------------------------------

    X_train_sampled, y_train_sampled = sample_data(
        X_train,
        y_train,
        "classification"
    )


    results = {}

    best_model = None

    best_model_name = None

    best_f1 = -np.inf


    # -----------------------------------------------------
    # Train Models
    # -----------------------------------------------------

    for name, model in models.items():

        try:

            model.fit(
                X_train_sampled,
                y_train_sampled
            )


            predictions = model.predict(
                X_test
            )


            accuracy = accuracy_score(
                y_test,
                predictions
            )


            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )


            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )


            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )


            results[name] = {

                "status": "success",

                "Accuracy": float(
                    accuracy
                ),

                "Precision": float(
                    precision
                ),

                "Recall": float(
                    recall
                ),

                "F1": float(
                    f1
                )
            }


            # -------------------------------------------------
            # Best Model
            # -------------------------------------------------

            if f1 > best_f1:

                best_f1 = f1

                best_model = model

                best_model_name = name


        except Exception as error:

            results[name] = {

                "status": "failed",

                "error": str(error)
            }


    # =====================================================
    # SAVE BEST MODEL ARTIFACT
    # =====================================================

    if best_model is not None:

        safe_name = make_safe_model_name(
            best_model_name
        )


        model_path = save_model_artifact(

            model=best_model,

            model_name=safe_name,

            preprocessor=preprocessor,

            feature_names=feature_names,

            target_column=target_column,

            task=task
        )


        print(
            f"Best classification model: {best_model_name}"
        )

        print(
            f"Best F1: {best_f1}"
        )

        print(
            f"Model artifact saved: {model_path}"
        )


    return results


# =========================================================
# SELECT BEST CLASSIFICATION MODEL
# =========================================================

def select_best_classification_model(
    results
):

    successful_models = {

        name: result

        for name, result in results.items()

        if result.get("status") == "success"
    }


    if not successful_models:

        return {

            "status": "failed",

            "message": "No classification model trained successfully"
        }


    best_name = max(

        successful_models,

        key=lambda name:
        successful_models[name]["F1"]
    )


    best_result = successful_models[
        best_name
    ]


    return {

        "status": "success",

        "model": best_name,

        "metric": "F1",

        "score": best_result["F1"],

        "Accuracy": best_result["Accuracy"],

        "Precision": best_result["Precision"],

        "Recall": best_result["Recall"]
    }


# =========================================================
# SELECT BEST REGRESSION MODEL
# =========================================================

def select_best_regression_model(
    results
):

    successful_models = {

        name: result

        for name, result in results.items()

        if result.get("status") == "success"
    }


    if not successful_models:

        return {

            "status": "failed",

            "message": "No regression model trained successfully"
        }


    best_name = max(

        successful_models,

        key=lambda name:
        successful_models[name]["R2"]
    )


    best_result = successful_models[
        best_name
    ]


    return {

        "status": "success",

        "model": best_name,

        "metric": "R2",

        "score": best_result["R2"],

        "MAE": best_result["MAE"],

        "RMSE": best_result["RMSE"]
    }
