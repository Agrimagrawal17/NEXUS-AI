import pandas as pd


TARGET_KEYWORDS = [
    "target",
    "label",
    "output",
    "result",
    "outcome",
    "class",
    "prediction",
    "price",
    "sales",
    "revenue",
    "profit",
    "churn",
    "status"
]


def calculate_target_score(
    column_name: str,
    series: pd.Series
) -> float:

    score = 0

    name = column_name.lower()

    # 1. Column name signal
    for keyword in TARGET_KEYWORDS:
        if keyword in name:
            score += 40
            break

    # 2. Avoid obvious ID columns
    if "id" in name:
        score -= 40

    # 3. Target should not be completely unique
    unique_ratio = series.nunique() / max(len(series), 1)

    if unique_ratio < 0.5:
        score += 20

    # 4. Too many unique values is usually not a target
    if unique_ratio > 0.95:
        score -= 30

    # Keep score between 0 and 100
    score = max(0, min(score, 100))

    return score


def detect_target(df: pd.DataFrame) -> dict:

    candidates = []

    for column in df.columns:

        score = calculate_target_score(
            column,
            df[column]
        )

        candidates.append({
            "column": column,
            "score": score
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if candidates and candidates[0]["score"] > 40:

        return {
            "target_found": True,
            "target_column": candidates[0]["column"],
            "confidence": candidates[0]["score"],
            "candidates": candidates
        }

    return {
        "target_found": False,
        "target_column": None,
        "confidence": 0,
        "candidates": candidates
    }