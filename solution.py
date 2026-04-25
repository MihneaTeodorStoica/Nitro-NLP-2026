import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DIACRITICS = set("ăâîșşțţĂÂÎȘŞȚŢ")
CAT = ["genre", "word_lower"]
NUM = [
    "word_len", "alpha_len", "digit_count", "punct_count", "diacritic_count",
    "is_title", "is_upper", "has_digit", "has_url", "has_diacritic",
    "section", "page", "word_pos", "log_word_pos",
    "word_count_in_corpus", "log_word_count_in_corpus",
    "prev_len", "next_len", "prev_punct", "next_punct",
]
COLS = NUM + CAT

BLEND = {"direct": 0.16, "two_stage": 0.52, "ridge": 0.32}
CALIBRATION_SLOPE = 1.0878022850997167
CALIBRATION_INTERCEPT = -22.08389399932139


def add_features(df, counts):
    x = df.copy()
    x["word"] = x["word"].fillna("").astype(str)
    x["word_lower"] = x["word"].str.lower()
    x["genre"] = x["text"].str.split("_").str[0]
    x["word_len"] = x["word"].str.len()
    x["alpha_len"] = x["word"].str.count(r"[A-Za-zĂÂÎȘŞȚŢăâîșşțţ]")
    x["digit_count"] = x["word"].str.count(r"\d")
    x["punct_count"] = x["word"].str.count(r"[^\w\sĂÂÎȘŞȚŢăâîșşțţ]")
    x["diacritic_count"] = x["word"].map(lambda word: sum(char in DIACRITICS for char in word))
    x["is_title"] = x["word"].str.istitle().astype(int)
    x["is_upper"] = x["word"].str.isupper().astype(int)
    x["has_digit"] = x["word"].str.contains(r"\d", regex=True).astype(int)
    x["has_url"] = x["word"].str.contains("http", case=False, regex=False).astype(int)
    x["has_diacritic"] = (x["diacritic_count"] > 0).astype(int)

    pos = x["word_id"].str.extract(r".+?_(\d+)_page_(\d+)_(\d+)$")
    x[["section", "page", "word_pos"]] = pos.astype(float)
    x["log_word_pos"] = np.log1p(x["word_pos"])
    x["word_count_in_corpus"] = x["word_lower"].map(counts).fillna(0).astype(float)
    x["log_word_count_in_corpus"] = np.log1p(x["word_count_in_corpus"])

    x = x.sort_values(["participant_id", "text", "section", "page", "word_pos"])
    group = x.groupby(["participant_id", "text"], sort=False)
    x["prev_len"] = group["word_len"].shift(1).fillna(0)
    x["next_len"] = group["word_len"].shift(-1).fillna(0)
    x["prev_punct"] = group["punct_count"].shift(1).fillna(0)
    x["next_punct"] = group["punct_count"].shift(-1).fillna(0)
    return x.sort_index()


def eval_metric(y_true, preds):
    r2 = max(0, r2_score(y_true, preds, force_finite=True))
    corr = pearsonr(y_true, preds)[0]
    corr = 0 if np.isnan(corr) else abs(corr)
    return 100 * (r2 + corr) / 2


def make_ridge():
    preprocess = ColumnTransformer(
        transformers=[
            ("word_chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30000), "word_lower"),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["genre"]),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", preprocess),
        ("regressor", Ridge(alpha=300.0, random_state=42)),
    ])


def direct_params(seed=42):
    return dict(
        loss_function="RMSE",
        iterations=1500,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=16,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )


def skip_params(seed=43):
    return dict(
        loss_function="Logloss",
        iterations=900,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=16,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )


def positive_params(seed=44):
    return dict(
        loss_function="RMSE",
        iterations=1300,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=16,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )


def blend(direct, two_stage, ridge):
    raw = BLEND["direct"] * direct + BLEND["two_stage"] * two_stage + BLEND["ridge"] * ridge
    return np.clip(CALIBRATION_SLOPE * raw + CALIBRATION_INTERCEPT, 0, 10000)


def run_cv(x, y, groups):
    oof = {"direct": np.zeros(len(x)), "two_stage": np.zeros(len(x)), "ridge": np.zeros(len(x))}

    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=groups.nunique()).split(x, y, groups), 1):
        direct_model = CatBoostRegressor(**direct_params())
        direct_model.fit(
            x.iloc[train_idx], y.iloc[train_idx], cat_features=CAT,
            eval_set=(x.iloc[valid_idx], y.iloc[valid_idx]), early_stopping_rounds=150,
        )
        oof["direct"][valid_idx] = np.clip(direct_model.predict(x.iloc[valid_idx]), 0, 10000)

        skip_model = CatBoostClassifier(**skip_params())
        skip_model.fit(
            x.iloc[train_idx], (y.iloc[train_idx] > 0).astype(int), cat_features=CAT,
            eval_set=(x.iloc[valid_idx], (y.iloc[valid_idx] > 0).astype(int)), early_stopping_rounds=100,
        )
        positive_prob = skip_model.predict_proba(x.iloc[valid_idx])[:, 1]

        positive_train_idx = train_idx[y.iloc[train_idx].to_numpy() > 0]
        positive_valid_idx = valid_idx[y.iloc[valid_idx].to_numpy() > 0]
        positive_model = CatBoostRegressor(**positive_params())
        positive_model.fit(
            x.iloc[positive_train_idx], y.iloc[positive_train_idx], cat_features=CAT,
            eval_set=(x.iloc[positive_valid_idx], y.iloc[positive_valid_idx]), early_stopping_rounds=150,
        )
        oof["two_stage"][valid_idx] = np.clip(positive_prob * positive_model.predict(x.iloc[valid_idx]), 0, 10000)

        ridge_model = make_ridge()
        ridge_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        oof["ridge"][valid_idx] = np.clip(ridge_model.predict(x.iloc[valid_idx]), 0, 10000)

        fold_pred = blend(oof["direct"][valid_idx], oof["two_stage"][valid_idx], oof["ridge"][valid_idx])
        print(f"fold {fold}: {groups.iloc[valid_idx].iat[0]} score={eval_metric(y.iloc[valid_idx], fold_pred):.3f}")

    pred = blend(oof["direct"], oof["two_stage"], oof["ridge"])
    print(f"OOF score: {eval_metric(y, pred):.3f}")
    print(f"OOF R2: {r2_score(y, pred, force_finite=True):.4f}")
    print(f"OOF Pearson: {pearsonr(y, pred)[0]:.4f}")


def main():
    train = pd.read_csv(ROOT / "train_data.csv", dtype={"participant_id": str})
    test = pd.read_csv(ROOT / "test_data.csv", dtype={"participant_id": str})
    sample = pd.read_csv(ROOT / "sample_output.csv")

    counts = pd.concat([train["word"], test["word"]], ignore_index=True).fillna("").astype(str).str.lower().value_counts()
    train = add_features(train, counts)
    test = add_features(test, counts)

    x = train[COLS]
    y = train["answer"].astype(float)
    groups = train["text"]

    if os.environ.get("NITRO_RUN_CV", "0") == "1":
        run_cv(x, y, groups)

    direct_model = CatBoostRegressor(**direct_params())
    direct_model.fit(x, y, cat_features=CAT)

    skip_model = CatBoostClassifier(**skip_params())
    skip_model.fit(x, (y > 0).astype(int), cat_features=CAT)

    positive_mask = y.to_numpy() > 0
    positive_model = CatBoostRegressor(**positive_params())
    positive_model.fit(x.loc[positive_mask], y.loc[positive_mask], cat_features=CAT)

    ridge_model = make_ridge()
    ridge_model.fit(x, y)

    x_test = test[COLS]
    direct_pred = np.clip(direct_model.predict(x_test), 0, 10000)
    two_stage_pred = np.clip(skip_model.predict_proba(x_test)[:, 1] * positive_model.predict(x_test), 0, 10000)
    ridge_pred = np.clip(ridge_model.predict(x_test), 0, 10000)
    pred = blend(direct_pred, two_stage_pred, ridge_pred)

    sub = pd.DataFrame({"subtaskID": 1, "datapointID": test["datapointID"].astype(int), "answer": pred})[sample.columns]
    assert len(sub) == len(sample) == len(test)
    assert sub["datapointID"].equals(sample["datapointID"])
    assert sub["answer"].notna().all()

    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"submission mean={sub['answer'].mean():.3f} std={sub['answer'].std():.3f} max={sub['answer'].max():.3f}")
    print(f"wrote {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()
