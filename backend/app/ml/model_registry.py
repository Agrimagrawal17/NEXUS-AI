import os
import json
from datetime import datetime


REGISTRY_DIR = os.path.join(
    "app",
    "saved_models"
)

REGISTRY_PATH = os.path.join(
    REGISTRY_DIR,
    "registry.json"
)


def _ensure_registry():
    os.makedirs(
        REGISTRY_DIR,
        exist_ok=True
    )

    if not os.path.exists(REGISTRY_PATH):
        with open(
            REGISTRY_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "models": []
                },
                file,
                indent=4
            )


def _load_registry():
    _ensure_registry()

    with open(
        REGISTRY_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def _save_registry(registry):
    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            registry,
            file,
            indent=4
        )


def register_model(
    model_name,
    model_path,
    task,
    target_column,
    metric,
    score
):

    registry = _load_registry()

    model_record = {
        "model_name": model_name,
        "model_path": model_path,
        "task": task,
        "target_column": target_column,
        "metric": metric,
        "score": score,
        "registered_at": datetime.now().isoformat()
    }

    registry["models"] = [
        model
        for model in registry["models"]
        if model["model_name"] != model_name
    ]

    registry["models"].append(
        model_record
    )

    _save_registry(
        registry
    )

    return model_record


def list_models():

    registry = _load_registry()

    return registry["models"]


def get_model(model_name):

    registry = _load_registry()

    for model in registry["models"]:

        if model["model_name"] == model_name:
            return model

    return None


def delete_model(model_name):

    registry = _load_registry()

    original_count = len(
        registry["models"]
    )

    registry["models"] = [
        model
        for model in registry["models"]
        if model["model_name"] != model_name
    ]

    if len(registry["models"]) == original_count:
        return False

    _save_registry(
        registry
    )

    return True