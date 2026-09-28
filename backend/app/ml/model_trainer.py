
import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

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


# =========================================================
# LARGE DATASET SETTINGS
# =========================================================

MAX_TRAIN_ROWS = 20000

EXPENSIVE_MODEL_THRESHOLD = 10000


# =========================================================
# SAMPLE LARGE DATASET
# =========================================================

def sample_data(
    X_train,
    y_train,
    max_rows=MAX_TRAIN_ROWS
):

    if len(X_train) <= max_rows:

        return X_train, y_train

    print(
        f"\nLarge dataset detected: {len(X_train)} rows"
    )

    print(
        f"Using {max_rows} rows for model training..."
    )

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if y_train.dtype == "object":

        samples_per_class = (
            max_rows // y_train.nunique()
        )

        sampled_parts = []

        for class_value, class_data in y_train.groupby(y_train):

            sample_size = min(
                samples_per_class,
                len(class_data)
            )

            sampled_class = class_data.sample(
                n=sample_size,
                random_state=42
            )

            sampled_parts.append(
                sampled_class
            )

        sample_indices = pd.concat(
            sampled_parts
        ).index

        X_sample = X_train.loc[
            sample_indices
        ]

        y_sample = y_train.loc[
            sample_indices
        ]

    # -----------------------------------------------------
    # Regression
    # -----------------------------------------------------

    else:

        sample_indices = (
            X_train
            .sample(
                n=max_rows,
                random_state=42
            )
            .index
        )

        X_sample = X_train.loc[
            sample_indices
        ]

        y_sample = y_train.loc[
            sample_indices
        ]

    return X_sample, y_sample


# =========================================================
# REGRESSION
# =========================================================

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    X_train_model, y_train_model = sample_data(
        X_train,
        y_train
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge": Ridge(),

        "Lasso": Lasso(),

        "ElasticNet": ElasticNet(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}

    total_models = len(models)

    for index, (model_name, model) in enumerate(
        models.items(),
        start=1
    ):

        print(
            f"\nTraining {index}/{total_models}: {model_name}"
        )

        try:

            model.fit(
                X_train_model,
                y_train_model
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

            results[model_name] = {

                "status": "success",

                "MAE": float(mae),

                "RMSE": float(rmse),

                "R2": float(r2)
            }

            print(
                f"Completed: {model_name} | "
                f"R2: {r2:.4f}"
            )

        except Exception as error:

            print(
                f"Failed: {model_name} | {error}"
            )

            results[model_name] = {

                "status": "failed",

                "error": str(error)
            }

    return results


# =========================================================
# CLASSIFICATION
# =========================================================

def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    X_train_model, y_train_model = sample_data(
        X_train,
        y_train
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": SVC(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42
        )
    }

    results = {}

    total_models = len(models)

    for index, (model_name, model) in enumerate(
        models.items(),
        start=1
    ):

        # -------------------------------------------------
        # Skip expensive models on very large datasets
        # -------------------------------------------------

        if (
            len(X_train) > EXPENSIVE_MODEL_THRESHOLD
            and model_name in ["KNN", "SVM"]
        ):

            print(
                f"\nSkipping {model_name} "
                f"(dataset too large)"
            )

            results[model_name] = {

                "status": "skipped",

                "reason": "Dataset too large for this model"
            }

            continue

        print(
            f"\nTraining {index}/{total_models}: "
            f"{model_name}"
        )

        try:

            model.fit(
                X_train_model,
                y_train_model
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

            results[model_name] = {

                "status": "success",

                "Accuracy": float(accuracy),

                "Precision": float(precision),

                "Recall": float(recall),

                "F1": float(f1)
            }

            print(
                f"Completed: {model_name} | "
                f"Accuracy: {accuracy:.4f}"
            )

        except Exception as error:

            print(
                f"Failed: {model_name} | {error}"
            )

            results[model_name] = {

                "status": "failed",

                "error": str(error)
            }

    return results

import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

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


# =========================================================
# LARGE DATASET SETTINGS
# =========================================================

MAX_TRAIN_ROWS = 20000

EXPENSIVE_MODEL_THRESHOLD = 10000


# =========================================================
# SAMPLE LARGE DATASET
# =========================================================

def sample_data(
    X_train,
    y_train,
    max_rows=MAX_TRAIN_ROWS
):

    if len(X_train) <= max_rows:

        return X_train, y_train

    print(
        f"\nLarge dataset detected: {len(X_train)} rows"
    )

    print(
        f"Using {max_rows} rows for model training..."
    )

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if y_train.dtype == "object":

        samples_per_class = (
            max_rows // y_train.nunique()
        )

        sampled_parts = []

        for class_value, class_data in y_train.groupby(y_train):

            sample_size = min(
                samples_per_class,
                len(class_data)
            )

            sampled_class = class_data.sample(
                n=sample_size,
                random_state=42
            )

            sampled_parts.append(
                sampled_class
            )

        sample_indices = pd.concat(
            sampled_parts
        ).index

        X_sample = X_train.loc[
            sample_indices
        ]

        y_sample = y_train.loc[
            sample_indices
        ]

    # -----------------------------------------------------
    # Regression
    # -----------------------------------------------------

    else:

        sample_indices = (
            X_train
            .sample(
                n=max_rows,
                random_state=42
            )
            .index
        )

        X_sample = X_train.loc[
            sample_indices
        ]

        y_sample = y_train.loc[
            sample_indices
        ]

    return X_sample, y_sample


# =========================================================
# REGRESSION
# =========================================================

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    X_train_model, y_train_model = sample_data(
        X_train,
        y_train
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge": Ridge(),

        "Lasso": Lasso(),

        "ElasticNet": ElasticNet(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}

    total_models = len(models)

    for index, (model_name, model) in enumerate(
        models.items(),
        start=1
    ):

        print(
            f"\nTraining {index}/{total_models}: {model_name}"
        )

        try:

            model.fit(
                X_train_model,
                y_train_model
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

            results[model_name] = {

                "status": "success",

                "MAE": float(mae),

                "RMSE": float(rmse),

                "R2": float(r2)
            }

            print(
                f"Completed: {model_name} | "
                f"R2: {r2:.4f}"
            )

        except Exception as error:

            print(
                f"Failed: {model_name} | {error}"
            )

            results[model_name] = {

                "status": "failed",

                "error": str(error)
            }

    return results


# =========================================================
# CLASSIFICATION
# =========================================================

def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    X_train_model, y_train_model = sample_data(
        X_train,
        y_train
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": SVC(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42
        )
    }

    results = {}

    total_models = len(models)

    for index, (model_name, model) in enumerate(
        models.items(),
        start=1
    ):

        # -------------------------------------------------
        # Skip expensive models on very large datasets
        # -------------------------------------------------

        if (
            len(X_train) > EXPENSIVE_MODEL_THRESHOLD
            and model_name in ["KNN", "SVM"]
        ):

            print(
                f"\nSkipping {model_name} "
                f"(dataset too large)"
            )

            results[model_name] = {

                "status": "skipped",

                "reason": "Dataset too large for this model"
            }

            continue

        print(
            f"\nTraining {index}/{total_models}: "
            f"{model_name}"
        )

        try:

            model.fit(
                X_train_model,
                y_train_model
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

            results[model_name] = {

                "status": "success",

                "Accuracy": float(accuracy),

                "Precision": float(precision),

                "Recall": float(recall),

                "F1": float(f1)
            }

            print(
                f"Completed: {model_name} | "
                f"Accuracy: {accuracy:.4f}"
            )

        except Exception as error:

            print(
                f"Failed: {model_name} | {error}"
            )

            results[model_name] = {

                "status": "failed",

                "error": str(error)
            }

    return results

# =========================================================
# BEST CLASSIFICATION MODEL
# =========================================================

def select_best_classification_model(results):

    successful_models = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("status") == "success"
    }

    if not successful_models:

        return {
            "status": "failed",
            "message": "No classification model completed successfully."
        }

    best_model_name = max(
        successful_models,
        key=lambda name: successful_models[name]["F1"]
    )

    best_model = successful_models[
        best_model_name
    ]

    return {
        "status": "success",
        "model": best_model_name,
        "metric": "F1",
        "score": best_model["F1"],
        "accuracy": best_model["Accuracy"],
        "precision": best_model["Precision"],
        "recall": best_model["Recall"]
    }


# =========================================================
# BEST REGRESSION MODEL
# =========================================================

def select_best_regression_model(results):

    successful_models = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("status") == "success"
    }

    if not successful_models:

        return {
            "status": "failed",
            "message": "No regression model completed successfully."
        }

    best_model_name = max(
        successful_models,
        key=lambda name: successful_models[name]["R2"]
    )

    best_model = successful_models[
        best_model_name
    ]

    return {
        "status": "success",
        "model": best_model_name,
        "metric": "R2",
        "score": best_model["R2"],
        "MAE": best_model["MAE"],
        "RMSE": best_model["RMSE"]
    }
