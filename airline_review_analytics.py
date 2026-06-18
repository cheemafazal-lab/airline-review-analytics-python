
from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_FILE = "ITAO7105_Airline_review_assignment_data.csv"
OUTPUT_DIR = "airline_assignment_outputs"

EXPECTED_COLUMNS = [
    "ID", "Airline Name", "Overall_Rating", "Review_Title", "Review Date",
    "Verified", "Review", "Aircraft", "Type Of Traveller", "Seat Type",
    "Route", "Date Flown", "Seat Comfort", "Cabin Staff Service",
    "Food & Beverages", "Ground Service", "Inflight Entertainment",
    "Wifi & Connectivity", "Value For Money", "Recommended"
]

NUMERIC_FEATURES = [
    "Overall_Rating", "Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Ground Service", "Inflight Entertainment", "Wifi & Connectivity", "Value For Money"
]
SERVICE_FEATURES = [c for c in NUMERIC_FEATURES if c != "Overall_Rating"]
CATEGORICAL_FEATURES = ["Verified", "Type Of Traveller", "Seat Type"]
TEXT_FEATURE = "Review_clean"
TARGET_BINARY = "Recommended_binary"

RATING_RANGES = {
    "Overall_Rating": (1, 10),
    "Seat Comfort": (1, 5),
    "Cabin Staff Service": (1, 5),
    "Food & Beverages": (1, 5),
    "Ground Service": (1, 5),
    "Inflight Entertainment": (1, 5),
    "Wifi & Connectivity": (1, 5),
    "Value For Money": (1, 5),
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "is", "are", "was", "were", "be", "been", "being", "am", "to", "of",
    "in", "on", "for", "from", "by", "at", "as", "this", "that", "these",
    "those", "it", "its", "i", "we", "you", "he", "she", "they", "them",
    "his", "her", "their", "our", "my", "me", "us", "have", "has", "had",
    "do", "does", "did", "done", "so", "very", "too", "can", "could",
    "would", "should", "will", "just", "about", "into", "than", "then",
    "there", "here", "also", "get", "got", "airline", "airlines", "flight",
    "flights", "plane", "passenger", "passengers"
}
NEGATIONS = {"no", "not", "nor", "never", "n't"}
STOP_WORDS = STOP_WORDS - NEGATIONS

GENERAL_POSITIVE = {
    "good", "great", "excellent", "amazing", "comfortable", "clean", "friendly",
    "helpful", "efficient", "smooth", "quick", "easy", "nice", "pleasant",
    "professional", "best", "better", "love", "loved", "enjoyed", "recommend",
    "recommended", "fine", "decent", "safe", "fast", "polite"
}
GENERAL_NEGATIVE = {
    "bad", "poor", "terrible", "awful", "horrible", "worst", "dirty", "rude",
    "delayed", "delay", "late", "cancelled", "canceled", "lost", "broken",
    "uncomfortable", "expensive", "avoid", "never", "problem", "problems",
    "complaint", "complaints", "worse", "disappointed", "chaos", "slow"
}
AIRLINE_POSITIVE = {
    "ontime", "punctual", "upgrade", "legroom", "lounge", "boarding", "crew",
    "staff", "service", "meal", "entertainment", "wifi", "checkin", "comfortable",
    "clean", "smooth", "efficient"
}
AIRLINE_NEGATIVE = {
    "delay", "delayed", "cancelled", "canceled", "queue", "queues", "lost",
    "baggage", "luggage", "refund", "overbooked", "cramped", "rude", "dirty",
    "uncomfortable", "missed", "connection", "chaotic", "expensive", "cold"
}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run_log.txt", mode="w", encoding="utf-8"),
        ],
    )


def save_current_figure(output_dir: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", filename)


def save_csv(df: pd.DataFrame, output_dir: Path, filename: str, index: bool = True) -> None:
    df.to_csv(output_dir / filename, index=index)
    logging.info("Saved table: %s", filename)


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def parse_review_date(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
    cleaned = cleaned.replace("nan", np.nan)
    return pd.to_datetime(cleaned, errors="coerce", dayfirst=True)


def standardise_recommended(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({
        "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
        "no": "no", "n": "no", "false": "no", "0": "no"
    })
    return mapped


def standardise_verified(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": "Verified", "1": "Verified", "yes": "Verified",
        "false": "Not Verified", "0": "Not Verified", "no": "Not Verified",
    })
    return mapped.fillna("Unknown")


def load_data(file_path: str) -> pd.DataFrame:
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Cannot find {file_path}. Put this script in the same folder as the CSV "
            "or edit DATA_FILE at the top of the script."
        )
    df = pd.read_csv(file_path)
    require_columns(df, EXPECTED_COLUMNS)
    logging.info("Loaded data: %s rows x %s columns", df.shape[0], df.shape[1])
    return df


def prepare_raw_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw = raw_df.copy()
    raw["Review Date"] = parse_review_date(raw["Review Date"])
    raw["Date Flown"] = pd.to_datetime(raw["Date Flown"], format="%b-%y", errors="coerce")
    raw["Recommended"] = standardise_recommended(raw["Recommended"])
    for col in NUMERIC_FEATURES:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def clean_data(raw_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df = raw_df.copy()

    initial_missing = pd.DataFrame({
        "missing_values": df.isna().sum(),
        "percent_missing": (df.isna().sum() / len(df)) * 100,
    }).sort_values("missing_values", ascending=False)
    save_csv(initial_missing, output_dir, "initial_missing_summary.csv")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    logging.info("Removed %s exact duplicate rows", duplicate_count)

    df["Review Date"] = parse_review_date(df["Review Date"])
    df["Date Flown"] = pd.to_datetime(df["Date Flown"], format="%b-%y", errors="coerce")
    df["Recommended"] = standardise_recommended(df["Recommended"])
    df["Verified"] = standardise_verified(df["Verified"])

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Airline Name", "Review_Title", "Review", "Aircraft", "Type Of Traveller", "Seat Type", "Route"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Unknown"

    range_audit = []
    for col, (low, high) in RATING_RANGES.items():
        bad_mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        range_audit.append({
            "variable": col,
            "allowed_min": low,
            "allowed_max": high,
            "out_of_range_before_fix": int(bad_mask.sum()),
        })
        df.loc[bad_mask, col] = np.nan
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = (low + high) / 2
        df[col] = df[col].fillna(median_value)

    save_csv(pd.DataFrame(range_audit), output_dir, "rating_range_audit.csv", index=False)

    post_missing = pd.DataFrame({
        "missing_values": df.isna().sum(),
        "percent_missing": (df.isna().sum() / len(df)) * 100,
    }).sort_values("missing_values", ascending=False)
    save_csv(post_missing, output_dir, "post_cleaning_missing_summary.csv")

    conflicts = pd.DataFrame({
        "issue": [
            "Overall_Rating <= 3 but Recommended = yes",
            "Overall_Rating >= 8 but Recommended = no",
        ],
        "count": [
            int(((df["Overall_Rating"] <= 3) & (df["Recommended"] == "yes")).sum()),
            int(((df["Overall_Rating"] >= 8) & (df["Recommended"] == "no")).sum()),
        ],
        "treatment": [
            "Retained; may represent nuanced or contradictory customer language.",
            "Retained; may represent nuanced or contradictory customer language.",
        ],
    })
    save_csv(conflicts, output_dir, "logical_conflict_audit.csv", index=False)

    return df


def tokenize(text: object) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    return [word for word in text.split() if word and word not in STOP_WORDS]


def clean_review_text(text: object) -> str:
    return " ".join(tokenize(text))


def lexicon_score(text: object, positive_words: set, negative_words: set) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    score = 0
    negate_next = False
    for token in tokens:
        if token in NEGATIONS:
            negate_next = True
            continue
        token_score = 0
        if token in positive_words:
            token_score = 1
        elif token in negative_words:
            token_score = -1
        if negate_next and token_score != 0:
            token_score *= -1
            negate_next = False
        score += token_score
    return score / np.sqrt(len(tokens))


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Review_clean"] = df["Review"].apply(clean_review_text)
    df["Review_word_count"] = df["Review"].apply(lambda x: len(tokenize(x)))
    df["general_lexicon_sentiment"] = df["Review"].apply(lambda x: lexicon_score(x, GENERAL_POSITIVE, GENERAL_NEGATIVE))
    df["airline_domain_sentiment"] = df["Review"].apply(lambda x: lexicon_score(x, AIRLINE_POSITIVE, AIRLINE_NEGATIVE))
    df["sentiment_disagreement"] = (
        np.sign(df["general_lexicon_sentiment"]) != np.sign(df["airline_domain_sentiment"])
    ).astype(int)
    return df


def make_pre_post_tables(raw_df: pd.DataFrame, clean_df: pd.DataFrame, output_dir: Path) -> None:
    raw = prepare_raw_comparison(raw_df)
    pre = raw[NUMERIC_FEATURES].agg(["count", "mean", "median", "std", "min", "max"]).T
    post = clean_df[NUMERIC_FEATURES].agg(["count", "mean", "median", "std", "min", "max"]).T
    comparison = pre.join(post, lsuffix="_pre", rsuffix="_post")
    comparison["missing_pre"] = raw[NUMERIC_FEATURES].isna().sum()
    comparison["missing_post"] = clean_df[NUMERIC_FEATURES].isna().sum()
    save_csv(comparison, output_dir, "pre_post_cleaning_numeric_comparison.csv")


def make_descriptive_tables(df: pd.DataFrame, output_dir: Path) -> None:
    desc = df[NUMERIC_FEATURES].describe().T
    desc["median"] = df[NUMERIC_FEATURES].median()
    desc["missing"] = df[NUMERIC_FEATURES].isna().sum()
    save_csv(desc, output_dir, "descriptive_statistics_numeric.csv")

    by_rec = df.groupby("Recommended")[NUMERIC_FEATURES].agg(["mean", "median", "std", "count"])
    by_rec.to_csv(output_dir / "ratings_by_recommendation.csv")

    sentiment = df.groupby("Recommended")[
        ["general_lexicon_sentiment", "airline_domain_sentiment", "Review_word_count"]
    ].agg(["mean", "median", "std", "count"])
    sentiment.to_csv(output_dir / "sentiment_summary_by_recommendation.csv")

    all_words = Counter(" ".join(df["Review_clean"].fillna("")).split())
    pd.DataFrame(all_words.most_common(50), columns=["word", "count"]).to_csv(
        output_dir / "top_50_words_overall.csv", index=False
    )
    for label in ["yes", "no"]:
        words = Counter(" ".join(df.loc[df["Recommended"] == label, "Review_clean"].fillna("")).split())
        pd.DataFrame(words.most_common(50), columns=["word", "count"]).to_csv(
            output_dir / f"top_50_words_recommended_{label}.csv", index=False
        )


def make_numeric_visualisations(df: pd.DataFrame, output_dir: Path) -> None:
    # 1 target distribution
    plt.figure(figsize=(7, 4))
    df["Recommended"].value_counts().reindex(["yes", "no"]).plot(kind="bar")
    plt.title("Distribution of Customer Recommendation")
    plt.xlabel("Recommended")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    save_current_figure(output_dir, "01_target_distribution.png")

    # 2 comparative boxplot
    plt.figure(figsize=(7, 5))
    groups = [df.loc[df["Recommended"] == label, "Overall_Rating"] for label in ["yes", "no"]]
    plt.boxplot(groups, labels=["yes", "no"], showmeans=True)
    plt.title("Overall Rating by Recommendation Outcome")
    plt.xlabel("Recommended")
    plt.ylabel("Overall Rating")
    save_current_figure(output_dir, "02_overall_rating_by_recommendation_boxplot.png")

    # 3 comparative service bar chart
    mean_ratings = df.groupby("Recommended")[SERVICE_FEATURES].mean().T
    plt.figure(figsize=(11, 6))
    mean_ratings.plot(kind="bar", ax=plt.gca())
    plt.title("Average Service Ratings by Recommendation Outcome")
    plt.xlabel("Service Attribute")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Recommended")
    save_current_figure(output_dir, "03_average_service_ratings_by_recommendation.png")

    # 4 correlation heatmap
    corr = df[NUMERIC_FEATURES].corr()
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation Between Numerical Rating Variables")
    save_current_figure(output_dir, "04_correlation_heatmap.png")

    # 5 management style service gap
    gap = (
        df.loc[df["Recommended"] == "yes", SERVICE_FEATURES].mean()
        - df.loc[df["Recommended"] == "no", SERVICE_FEATURES].mean()
    ).sort_values()
    plt.figure(figsize=(10, 6))
    gap.plot(kind="barh")
    plt.title("Management View: Largest Service Rating Gaps")
    plt.xlabel("Mean Rating Difference: Recommended Yes minus No")
    plt.ylabel("Service Attribute")
    save_current_figure(output_dir, "05_management_service_gap_chart.png")

    # 6 trend over time
    dated = df.dropna(subset=["Date Flown"]).copy()
    if len(dated) > 0:
        monthly = dated.set_index("Date Flown").resample("M")["Overall_Rating"].mean().dropna()
        if len(monthly) > 3:
            plt.figure(figsize=(11, 5))
            monthly.plot()
            plt.title("Average Overall Rating Over Time")
            plt.xlabel("Date Flown")
            plt.ylabel("Average Overall Rating")
            save_current_figure(output_dir, "06_average_rating_over_time.png")


def make_text_visualisations(df: pd.DataFrame, output_dir: Path) -> None:
    yes_words = pd.Series(Counter(" ".join(df.loc[df["Recommended"] == "yes", "Review_clean"]).split())).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    yes_words.sort_values().plot(kind="barh")
    plt.title("Top Words in Recommended Reviews")
    plt.xlabel("Frequency")
    save_current_figure(output_dir, "07_top_words_recommended_yes.png")

    no_words = pd.Series(Counter(" ".join(df.loc[df["Recommended"] == "no", "Review_clean"]).split())).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    no_words.sort_values().plot(kind="barh")
    plt.title("Top Words in Not Recommended Reviews")
    plt.xlabel("Frequency")
    save_current_figure(output_dir, "08_top_words_recommended_no.png")

    plt.figure(figsize=(7, 5))
    groups = [df.loc[df["Recommended"] == label, "general_lexicon_sentiment"] for label in ["yes", "no"]]
    plt.boxplot(groups, labels=["yes", "no"], showmeans=True)
    plt.title("General Lexicon Sentiment by Recommendation")
    plt.xlabel("Recommended")
    plt.ylabel("General Lexicon Sentiment")
    save_current_figure(output_dir, "09_general_sentiment_by_recommendation.png")

    contradiction_type = np.select(
        [
            (df["general_lexicon_sentiment"] > 0) & (df["Recommended"] == "no"),
            (df["general_lexicon_sentiment"] < 0) & (df["Recommended"] == "yes"),
            (df["Overall_Rating"] <= 3) & (df["Recommended"] == "yes"),
            (df["Overall_Rating"] >= 8) & (df["Recommended"] == "no"),
        ],
        [
            "Positive text / not recommended",
            "Negative text / recommended",
            "Low rating / recommended",
            "High rating / not recommended",
        ],
        default="No selected contradiction",
    )
    contradiction_df = df.copy()
    contradiction_df["contradiction_type"] = contradiction_type
    contradiction_df[[
        "Airline Name", "Overall_Rating", "Recommended", "general_lexicon_sentiment",
        "airline_domain_sentiment", "contradiction_type", "Review"
    ]].to_csv(output_dir / "text_rating_contradiction_examples.csv", index=False)

    counts = contradiction_df["contradiction_type"].value_counts().drop(labels=["No selected contradiction"], errors="ignore")
    if len(counts) > 0:
        plt.figure(figsize=(10, 5))
        counts.sort_values().plot(kind="barh")
        plt.title("Contradictions Between Text, Rating and Recommendation")
        plt.xlabel("Number of Reviews")
        save_current_figure(output_dir, "10_text_rating_recommendation_contradictions.png")

    sample = df.sample(min(4000, len(df)), random_state=RANDOM_STATE)
    plt.figure(figsize=(8, 5))
    plt.scatter(sample["general_lexicon_sentiment"], sample["Overall_Rating"], alpha=0.25)
    plt.title("Alignment Between Text Sentiment and Overall Rating")
    plt.xlabel("General Lexicon Sentiment")
    plt.ylabel("Overall Rating")
    save_current_figure(output_dir, "11_sentiment_vs_overall_rating.png")

    plt.figure(figsize=(7, 5))
    groups = [df.loc[df["Recommended"] == label, "Review_word_count"] for label in ["yes", "no"]]
    plt.boxplot(groups, labels=["yes", "no"], showmeans=True)
    plt.yscale("log")
    plt.title("Review Word Count by Recommendation")
    plt.xlabel("Recommended")
    plt.ylabel("Review Word Count (log scale)")
    save_current_figure(output_dir, "12_review_length_by_recommendation.png")


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[df["Recommended"].isin(["yes", "no"])].copy()
    model_df[TARGET_BINARY] = model_df["Recommended"].map({"yes": 1, "no": 0}).astype(int)
    for col in CATEGORICAL_FEATURES:
        model_df[col] = model_df[col].fillna("Unknown").astype(str)
    model_df["Review_clean"] = model_df["Review_clean"].fillna("")
    return model_df


def evaluate_model(model, x_test, y_test, feature_set: str, model_name: str, best_params: Dict, output_dir: Path) -> Dict:
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = np.full(len(y_test), np.nan)
        roc_auc = np.nan

    report = classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"classification_report_{feature_set}_{model_name}.csv".lower().replace(" ", "_"))

    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"])
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix: {feature_set} - {model_name}")
    save_current_figure(output_dir, f"confusion_matrix_{feature_set}_{model_name}.png".lower().replace(" ", "_"))

    return {
        "feature_set": feature_set,
        "model_name": model_name,
        "best_params": str(best_params),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "test_rows": len(y_test),
    }


def fit_grid(pipeline: Pipeline, param_grid: Dict, x_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=2,
        n_jobs=1,
        error_score="raise",
    )
    grid.fit(x_train, y_train)
    return grid


def run_numeric_models(model_df: pd.DataFrame, output_dir: Path) -> List[Dict]:
    x = model_df[NUMERIC_FEATURES]
    y = model_df[TARGET_BINARY]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    results = []

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_grid = fit_grid(lr, {"classifier__alpha": [0.0001, 0.001]}, x_train, y_train)
    results.append(evaluate_model(lr_grid.best_estimator_, x_test, y_test, "Numeric", "Logistic Regression SGD", lr_grid.best_params_, output_dir))

    nb = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", ComplementNB()),
    ])
    nb_grid = fit_grid(nb, {"classifier__alpha": [0.5, 1.0, 2.0]}, x_train, y_train)
    results.append(evaluate_model(nb_grid.best_estimator_, x_test, y_test, "Numeric", "Complement NB", nb_grid.best_params_, output_dir))

    return results


def run_text_models(model_df: pd.DataFrame, output_dir: Path) -> Tuple[List[Dict], Pipeline]:
    x = model_df["Review_clean"]
    y = model_df[TARGET_BINARY]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    min_df = 5 if len(x_train) >= 1000 else 2
    results = []

    lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=700, ngram_range=(1, 1), min_df=max(min_df, 10))),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_grid = fit_grid(lr, {"classifier__alpha": [0.0001, 0.001]}, x_train, y_train)
    best_text_lr = lr_grid.best_estimator_
    results.append(evaluate_model(best_text_lr, x_test, y_test, "Text", "Logistic Regression SGD", lr_grid.best_params_, output_dir))

    nb = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=700, ngram_range=(1, 1), min_df=max(min_df, 10))),
        ("classifier", ComplementNB()),
    ])
    nb_grid = fit_grid(nb, {"classifier__alpha": [0.5, 1.0, 2.0]}, x_train, y_train)
    results.append(evaluate_model(nb_grid.best_estimator_, x_test, y_test, "Text", "Complement NB", nb_grid.best_params_, output_dir))

    save_text_lr_features(best_text_lr, output_dir)
    return results, best_text_lr


def combined_preprocessor(use_standard_scaler: bool) -> ColumnTransformer:
    numeric_plus_text_scores = NUMERIC_FEATURES + ["general_lexicon_sentiment", "airline_domain_sentiment", "Review_word_count"]
    scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()
    return ColumnTransformer([
        ("num", scaler, numeric_plus_text_scores),
        ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ("text", TfidfVectorizer(max_features=700, ngram_range=(1, 1), min_df=10), TEXT_FEATURE),
    ])


def run_combined_models(model_df: pd.DataFrame, output_dir: Path) -> List[Dict]:
    features = NUMERIC_FEATURES + [
        "general_lexicon_sentiment", "airline_domain_sentiment", "Review_word_count"
    ] + CATEGORICAL_FEATURES + [TEXT_FEATURE]
    x = model_df[features]
    y = model_df[TARGET_BINARY]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    results = []

    lr = Pipeline([
        ("preprocessor", combined_preprocessor(use_standard_scaler=True)),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_grid = fit_grid(lr, {"classifier__alpha": [0.0001, 0.001]}, x_train, y_train)
    best_combined_lr = lr_grid.best_estimator_
    results.append(evaluate_model(best_combined_lr, x_test, y_test, "Combined", "Logistic Regression SGD", lr_grid.best_params_, output_dir))
    save_combined_lr_features(best_combined_lr, output_dir)
    save_failure_analysis(best_combined_lr, x_test, y_test, output_dir)

    # The combined Logistic Regression SGD model is used for integrated feature analysis,
    # feature dominance and failure analysis. The separate numerical and text sections
    # already compare both selected supervised algorithms.

    return results


def save_text_lr_features(model: Pipeline, output_dir: Path) -> None:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]
    pos_idx = np.argsort(coefs)[-30:][::-1]
    neg_idx = np.argsort(coefs)[:30]
    pd.DataFrame({"feature": names[pos_idx], "coefficient": coefs[pos_idx]}).to_csv(
        output_dir / "top_positive_text_features.csv", index=False
    )
    pd.DataFrame({"feature": names[neg_idx], "coefficient": coefs[neg_idx]}).to_csv(
        output_dir / "top_negative_text_features.csv", index=False
    )


def get_combined_feature_names(model: Pipeline) -> np.ndarray:
    prep = model.named_steps["preprocessor"]
    numeric_names = np.array(NUMERIC_FEATURES + ["general_lexicon_sentiment", "airline_domain_sentiment", "Review_word_count"])
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    text_names = prep.named_transformers_["text"].get_feature_names_out()
    return np.concatenate([numeric_names, cat_names, text_names])


def save_combined_lr_features(model: Pipeline, output_dir: Path) -> None:
    names = get_combined_feature_names(model)
    coefs = model.named_steps["classifier"].coef_[0]
    out = pd.DataFrame({
        "feature": names,
        "coefficient": coefs,
        "absolute_importance": np.abs(coefs),
    }).sort_values("absolute_importance", ascending=False)
    out.head(60).to_csv(output_dir / "combined_logistic_feature_importance_top60.csv", index=False)


def save_failure_analysis(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    failures = x_test.copy()
    failures["actual"] = y_test.values
    failures["predicted"] = y_pred
    failures["probability_recommended_yes"] = y_prob
    failures["uncertainty_distance_from_0_5"] = np.abs(y_prob - 0.5)
    failures = failures[failures["actual"] != failures["predicted"]]
    failures.sort_values("uncertainty_distance_from_0_5").head(15).to_csv(
        output_dir / "combined_model_failure_analysis_manual_review.csv", index=False
    )


def run_temporal_stress_test(model_df: pd.DataFrame, output_dir: Path) -> None:
    dated = model_df.dropna(subset=["Review Date"]).sort_values("Review Date")
    if len(dated) < 500:
        logging.warning("Temporal stress test skipped: not enough dated records.")
        return
    split_date = dated["Review Date"].quantile(0.75)
    train = dated[dated["Review Date"] <= split_date]
    test = dated[dated["Review Date"] > split_date]
    if train[TARGET_BINARY].nunique() < 2 or test[TARGET_BINARY].nunique() < 2:
        logging.warning("Temporal stress test skipped: target class issue.")
        return
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(train[NUMERIC_FEATURES], train[TARGET_BINARY])
    result = evaluate_model(
        model, test[NUMERIC_FEATURES], test[TARGET_BINARY],
        "Temporal Stress Numeric", "Logistic Regression SGD",
        {"train_until": str(split_date)}, output_dir
    )
    pd.DataFrame([result]).to_csv(output_dir / "temporal_stress_test_results.csv", index=False)


def run_airline_holdout_stress_test(model_df: pd.DataFrame, output_dir: Path) -> None:
    heldout_airline = model_df["Airline Name"].value_counts().index[0]
    train = model_df[model_df["Airline Name"] != heldout_airline]
    test = model_df[model_df["Airline Name"] == heldout_airline]
    if len(test) < 100 or train[TARGET_BINARY].nunique() < 2 or test[TARGET_BINARY].nunique() < 2:
        logging.warning("Airline holdout stress test skipped: unsuitable held-out subset.")
        return
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(train[NUMERIC_FEATURES], train[TARGET_BINARY])
    result = evaluate_model(
        model, test[NUMERIC_FEATURES], test[TARGET_BINARY],
        "Airline Holdout Stress Numeric", "Logistic Regression SGD",
        {"heldout_airline": heldout_airline}, output_dir
    )
    pd.DataFrame([result]).to_csv(output_dir / "airline_holdout_stress_test_results.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    output_dir = Path(OUTPUT_DIR)
    setup_logging(output_dir)

    logging.info("Working directory: %s", os.getcwd())
    logging.info("Starting one-click airline review assignment analysis.")

    raw_df = load_data(DATA_FILE)
    raw_df.head(10).to_csv(output_dir / "initial_data_preview.csv", index=False)
    pd.DataFrame({
        "raw_rows": [raw_df.shape[0]],
        "raw_columns": [raw_df.shape[1]],
        "exact_duplicate_rows": [int(raw_df.duplicated().sum())],
    }).to_csv(output_dir / "initial_dataset_shape_and_duplicates.csv", index=False)

    clean_df = clean_data(raw_df, output_dir)
    clean_df = add_text_features(clean_df)
    clean_df.to_csv(output_dir / "cleaned_airline_reviews.csv", index=False)

    make_pre_post_tables(raw_df, clean_df, output_dir)
    make_descriptive_tables(clean_df, output_dir)
    make_numeric_visualisations(clean_df, output_dir)
    make_text_visualisations(clean_df, output_dir)

    model_df = prepare_model_data(clean_df)
    model_df.to_csv(output_dir / "model_ready_data.csv", index=False)

    results = []
    results.extend(run_numeric_models(model_df, output_dir))
    results.extend(run_text_models(model_df, output_dir)[0])
    results.extend(run_combined_models(model_df, output_dir))

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(output_dir / "all_model_results_comparison.csv", index=False)

    run_temporal_stress_test(model_df, output_dir)
    run_airline_holdout_stress_test(model_df, output_dir)

    with open(output_dir / "report_writer_quick_summary.txt", "w", encoding="utf-8") as f:
        f.write("ITAO7105 Airline Review Assignment - Quick Output Summary\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Raw data shape: {raw_df.shape}\n")
        f.write(f"Clean data shape: {clean_df.shape}\n")
        f.write(f"Model-ready data shape: {model_df.shape}\n\n")
        f.write("Best models by F1 score:\n")
        f.write(results_df[["feature_set", "model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))
        f.write("\n\nMain figures: 01_target_distribution.png to 12_review_length_by_recommendation.png\n")
        f.write("Main tables: pre_post_cleaning_numeric_comparison.csv, descriptive_statistics_numeric.csv, ratings_by_recommendation.csv, all_model_results_comparison.csv\n")

    logging.info("SCRIPT FINISHED SUCCESSFULLY. Outputs saved in: %s", output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.exception("Script failed: %s", error)
        print("\nERROR:", error)
        print("Check airline_assignment_outputs/run_log.txt for details.")
        sys.exit(1)
