import os
import joblib


MODEL_DIR = os.path.join(
    "app",
    "saved_models"
)


def _safe_model_name(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_model_artifact(
    model,
    model_name,
    preprocessor=None,
    feature_names=None,
    target_column=None,
    task=None,
    preprocessing_metadata=None
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    safe_name = _safe_model_name(
        model_name
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )

    artifact = {
        "model": model,
        "model_name": model_name,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "target_column": target_column,
        "task": task,
        "preprocessing_metadata": preprocessing_metadata
    }

    joblib.dump(
        artifact,
        model_path
    )

    return model_path


def load_model_artifact(model_name):

    safe_name = _safe_model_name(
        model_name
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


# Backward-compatible functions

def save_model(
    model,
    model_name
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    return model_path


def load_model(
    model_name
):

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.joblib"
    )

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )