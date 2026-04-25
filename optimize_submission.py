#!/usr/bin/env python3
"""Build an optimized TRT submission from train/test and eye-tracking sources.

This script uses the competition-style metric from the original notebook:

    100 * (abs(Pearson) + max(0, R2)) / 2

The direct `words_dict_romanian_merged.csv` values are close, but they are
word-level averages. The target rows are participant-level TRT, so the script
calibrates those averages against `train_data.csv`, adds all available
eye-tracking subject files/properties, trains CatBoost with text-held-out OOF
predictions, and optimizes an ensemble on the OOF predictions.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover - handled at runtime
    CatBoostRegressor = None
    Pool = None

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - handled at runtime
    minimize = None


BASE_EYE_PATH = Path("eye-tracking/trt_model/word_sentence_fixations")
BASE_PROPERTIES_PATH = Path("eye-tracking/trt_model/properties")
DIRECT_EYE_PATH = BASE_EYE_PATH / "words_dict_romanian_merged.csv"
MULTI_EYE_PATH = BASE_EYE_PATH / "words_dict_romanian_merged_008_009_010_011.csv"
SUBJECT_IDS = ["008", "009", "010", "011", "023"]


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def competition_metric(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = max(0.0, r2_score_np(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = abs(float(np.corrcoef(y_true, y_pred)[0, 1]))
    return 100.0 * (pearson + r2) / 2.0


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def clean_word(value: object) -> str:
    return re.sub(r"^[^\wăâîșşțţ]+|[^\wăâîșşțţ]+$", "", str(value).lower())


def normalize_word(value: object) -> str:
    return str(value).replace("\u2013", "-").replace("\u2014", "-")


def word_shape(value: object) -> str:
    shape = []
    for ch in str(value):
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return re.sub(r"(.)\1+", r"\1", "".join(shape))[:24]


def parse_word_id_parts(series: pd.Series) -> pd.DataFrame:
    parts = series.astype(str).str.extract(
        r"^(?P<stimulus>.+_page_(?P<page>\d+))_(?P<word_index>\d+)$"
    )
    parts["page"] = pd.to_numeric(parts["page"], errors="coerce")
    parts["word_index"] = pd.to_numeric(parts["word_index"], errors="coerce")
    return parts


def load_eye_features() -> pd.DataFrame:
    direct = pd.read_csv(DIRECT_EYE_PATH).drop_duplicates("word_id")
    direct = direct.set_index("word_id").rename(
        columns={
            "average_TRT": "direct_trt",
            "word": "direct_word",
            "word_index": "direct_word_index",
            "word_index_in_sentence": "direct_word_index_in_sentence",
            "sentence_index": "direct_sentence_index",
        }
    )
    direct["direct_sentence_len"] = direct["sentence"].fillna("").astype(str).str.split().map(len)
    direct = direct[
        [
            "direct_word",
            "direct_trt",
            "direct_word_index",
            "direct_word_index_in_sentence",
            "direct_sentence_index",
            "direct_sentence_len",
        ]
    ]

    multi = pd.read_csv(MULTI_EYE_PATH).drop_duplicates("word_id")
    multi = multi.set_index("word_id").rename(
        columns={
            "average_TRT": "multi_trt",
            "complexity": "multi_complexity",
            "word": "multi_word",
            "word_index": "multi_word_index",
            "word_index_in_sentence": "multi_word_index_in_sentence",
            "sentence_index": "multi_sentence_index",
        }
    )
    multi["multi_sentence_len"] = multi["sentence"].fillna("").astype(str).str.split().map(len)
    multi = multi[
        [
            "multi_word",
            "multi_trt",
            "multi_complexity",
            "multi_word_index",
            "multi_word_index_in_sentence",
            "multi_sentence_index",
            "multi_sentence_len",
        ]
    ]

    subject_frames = []
    for subject_id in SUBJECT_IDS:
        path = BASE_EYE_PATH / f"words_dict_romanian_{subject_id}.csv"
        if not path.exists():
            continue
        subject = pd.read_csv(path)
        subject = subject[["word_id", "fixations_TRT", "surprisal"]].copy()
        subject["fixations_TRT"] = pd.to_numeric(subject["fixations_TRT"], errors="coerce")
        subject = subject.rename(
            columns={
                "fixations_TRT": f"subj_{subject_id}_trt",
                "surprisal": f"subj_{subject_id}_surprisal",
            }
        )
        subject_frames.append(subject.drop_duplicates("word_id").set_index("word_id"))

    eye = direct.join(multi, how="outer")
    for subject in subject_frames:
        eye = eye.join(subject, how="outer")

    subject_trt_cols = [col for col in eye.columns if re.match(r"subj_\d{3}_trt$", col)]
    subject_values = eye[subject_trt_cols]
    eye["subject_mean_trt"] = subject_values.mean(axis=1)
    eye["subject_median_trt"] = subject_values.median(axis=1)
    eye["subject_std_trt"] = subject_values.std(axis=1).fillna(0)
    eye["subject_min_trt"] = subject_values.min(axis=1)
    eye["subject_max_trt"] = subject_values.max(axis=1)
    eye["subject_count"] = subject_values.notna().sum(axis=1)
    eye["subject_zero_rate"] = (subject_values == 0).mean(axis=1)
    eye["subject_range_trt"] = eye["subject_max_trt"] - eye["subject_min_trt"]
    eye["direct_multi_gap"] = eye["direct_trt"] - eye["multi_trt"]
    return eye


def load_property_features() -> pd.DataFrame:
    frames = []
    for label, path in [
        ("prop009", BASE_PROPERTIES_PATH / "properties_romanian_009/properties.csv"),
        ("prop010", BASE_PROPERTIES_PATH / "properties_romanian_010/properties.csv"),
        ("prop023", BASE_PROPERTIES_PATH / "properties_romanian_023/surprisal.csv"),
    ]:
        if not path.exists():
            continue
        props = pd.read_csv(path)
        parts = props["stimulus"].astype(str) + "_" + props["word_index"].astype(str)
        props = props.assign(word_id=parts)
        keep = ["word_id"]
        rename = {}
        for col in ["surprisal", "num_tokens", "frequency"]:
            if col in props.columns:
                keep.append(col)
                rename[col] = f"{label}_{col}"
        frames.append(props[keep].drop_duplicates("word_id").set_index("word_id").rename(columns=rename))

    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, how="outer")
    return out


def load_complexity_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = BASE_EYE_PATH / "multipleye_sorted.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    complexity = pd.read_csv(path)
    numeric = ["complexity", "value", "varv", "vars", "gpt-4.1-2025-04-14"]
    numeric = [col for col in numeric if col in complexity.columns]
    complexity["clean_token"] = complexity["token"].map(clean_word)
    token = complexity.groupby("clean_token")[numeric].agg(["mean", "max", "std"]).fillna(0)
    token.columns = [f"token_{col}_{stat}" for col, stat in token.columns]
    complexity["clean_lemma"] = complexity["lemma"].map(clean_word)
    lemma = complexity.groupby("clean_lemma")[numeric].agg(["mean", "max", "std"]).fillna(0)
    lemma.columns = [f"lemma_{col}_{stat}" for col, stat in lemma.columns]
    return token, lemma


def build_features(rows: pd.DataFrame, eye: pd.DataFrame, props: pd.DataFrame, token, lemma) -> pd.DataFrame:
    result = rows[["word_id", "word", "participant_id", "text"]].copy()
    result["participant_id"] = result["participant_id"].astype(str).str.zfill(3)
    result["genre"] = result["text"].astype(str).str.split("_").str[0]

    parts = parse_word_id_parts(result["word_id"])
    result["stimulus"] = parts["stimulus"].fillna("missing")
    result["page"] = parts["page"].fillna(-1).astype(int)
    result["word_index_from_id"] = parts["word_index"].fillna(-1).astype(int)

    words = result["word"].fillna("").astype(str)
    cleaned = words.map(clean_word)
    result["clean_word"] = cleaned
    result["word_len"] = words.str.len()
    result["clean_len"] = cleaned.str.len()
    result["log_word_len"] = np.log1p(result["word_len"])
    result["unique_chars"] = words.map(lambda value: len(set(value)))
    result["vowel_count"] = words.map(lambda value: sum(ch.lower() in "aeiouăâî" for ch in str(value)))
    result["consonant_count"] = words.map(
        lambda value: sum(ch.isalpha() and ch.lower() not in "aeiouăâî" for ch in str(value))
    )
    result["is_title"] = words.map(str.istitle).astype(int)
    result["is_upper"] = words.map(str.isupper).astype(int)
    result["is_lower"] = words.map(str.islower).astype(int)
    result["has_digit"] = words.str.contains(r"\d", regex=True).astype(int)
    result["has_punct"] = words.str.contains(r"[^\wăâîșşțţ]", regex=True).astype(int)
    result["prefix_2"] = cleaned.str[:2].fillna("")
    result["suffix_2"] = cleaned.str[-2:].fillna("")
    result["suffix_3"] = cleaned.str[-3:].fillna("")
    result["word_shape"] = words.map(word_shape)

    result = result.join(eye, on="word_id")
    if not props.empty:
        result = result.join(props, on="word_id")
    if not token.empty:
        result = result.join(token, on="clean_word")
    if not lemma.empty:
        result = result.join(lemma, on="clean_word")

    eye_sources = [
        "direct_trt",
        "multi_trt",
        "subject_mean_trt",
        "subject_median_trt",
        "subj_008_trt",
        "subj_009_trt",
        "subj_010_trt",
        "subj_011_trt",
        "subj_023_trt",
    ]
    existing = [col for col in eye_sources if col in result.columns]
    result["eye_mean_all_sources"] = result[existing].mean(axis=1)
    result["eye_median_all_sources"] = result[existing].median(axis=1)
    result["eye_std_all_sources"] = result[existing].std(axis=1).fillna(0)
    result["eye_best_source"] = result["multi_trt"].combine_first(result["subject_mean_trt"])
    result["eye_best_source"] = result["eye_best_source"].combine_first(result["direct_trt"])
    result["word_match_direct"] = (
        result["word"].map(normalize_word) == result["direct_word"].map(normalize_word)
    ).astype(int)

    return result


def grouped_text_splits(frame: pd.DataFrame, n_splits: int):
    text_sizes = frame.groupby("text").size().sort_values(ascending=False)
    folds = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for text, size in text_sizes.items():
        fold_id = int(np.argmin(fold_sizes))
        folds[fold_id].append(text)
        fold_sizes[fold_id] += int(size)
    for texts in folds:
        val_mask = frame["text"].isin(texts).to_numpy()
        yield np.flatnonzero(~val_mask), np.flatnonzero(val_mask), texts


def fit_linear_oof(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, columns: list[str], n_splits: int):
    train_matrix = train[columns].apply(pd.to_numeric, errors="coerce")
    test_matrix = test[columns].apply(pd.to_numeric, errors="coerce")
    means = train_matrix.mean()
    train_matrix = train_matrix.fillna(means).fillna(0).to_numpy(dtype=float)
    test_matrix = test_matrix.fillna(means).fillna(0).to_numpy(dtype=float)
    train_matrix = np.column_stack([train_matrix, np.ones(len(train_matrix))])
    test_matrix = np.column_stack([test_matrix, np.ones(len(test_matrix))])

    oof = np.zeros(len(train), dtype=float)
    for train_idx, val_idx, _ in grouped_text_splits(train, n_splits):
        coef = np.linalg.lstsq(train_matrix[train_idx], y[train_idx], rcond=None)[0]
        oof[val_idx] = train_matrix[val_idx] @ coef

    coef = np.linalg.lstsq(train_matrix, y, rcond=None)[0]
    test_pred = test_matrix @ coef
    return np.clip(oof, 0, None), np.clip(test_pred, 0, None), coef


def optimize_ensemble(y: np.ndarray, predictions: dict[str, np.ndarray], step: float):
    names = list(predictions)
    matrix = np.vstack([np.asarray(predictions[name], dtype=float) for name in names])

    if len(names) == 1:
        return {names[0]: 1.0}, competition_metric(y, matrix[0])

    def score_weights(weights: np.ndarray) -> float:
        return competition_metric(y, weights @ matrix)

    candidates: list[np.ndarray] = []
    for i in range(len(names)):
        weights = np.zeros(len(names), dtype=float)
        weights[i] = 1.0
        candidates.append(weights)

    if minimize is not None:
        starts = candidates + [np.full(len(names), 1.0 / len(names))]
        for start in starts:
            result = minimize(
                lambda weights: -score_weights(weights),
                start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(names),
                constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
                options={"maxiter": 500, "ftol": 1e-9, "disp": False},
            )
            if result.success:
                candidates.append(np.clip(result.x, 0, 1))
    else:
        # Coarse fallback for environments without scipy.
        grid = np.arange(0, 1 + 1e-9, max(step, 0.1))
        for raw_weights in itertools.product(grid, repeat=min(len(names), 4) - 1):
            used = sum(raw_weights)
            if used > 1:
                continue
            weights = np.zeros(len(names), dtype=float)
            weights[: len(raw_weights)] = raw_weights
            weights[len(raw_weights)] = 1 - used
            candidates.append(weights)

    best_weights = max(candidates, key=score_weights)
    best_weights = best_weights / best_weights.sum()
    best_score = score_weights(best_weights)

    return dict(zip(names, best_weights)), best_score


def prepare_model_frames(train_features: pd.DataFrame, test_features: pd.DataFrame):
    drop_cols = {"word_id", "word", "direct_word", "multi_word", "clean_word"}
    feature_cols = [
        col
        for col in train_features.columns
        if col not in drop_cols and col in test_features.columns
    ]
    categorical = [
        col
        for col in ["participant_id", "text", "genre", "stimulus", "prefix_2", "suffix_2", "suffix_3", "word_shape"]
        if col in feature_cols
    ]

    x_train = train_features[feature_cols].copy()
    x_test = test_features[feature_cols].copy()
    for frame in [x_train, x_test]:
        for col in categorical:
            frame[col] = frame[col].fillna("missing").astype(str)
        numeric = [col for col in frame.columns if col not in categorical]
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    return x_train, x_test, categorical


def make_catboost(iterations: int, seed: int):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=0.045,
        depth=7,
        l2_leaf_reg=10.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )


def catboost_oof_and_test(x_train, x_test, y, train_rows, categorical, n_splits, iterations):
    if CatBoostRegressor is None:
        print("CatBoost is not installed; skipping model predictions.", flush=True)
        return None, None

    oof = np.zeros(len(x_train), dtype=float)
    fold_reports = []
    best_iterations = []
    for fold, (train_idx, val_idx, texts) in enumerate(grouped_text_splits(train_rows, n_splits), 1):
        model = make_catboost(iterations=iterations, seed=2026 + fold)
        model.fit(
            Pool(x_train.iloc[train_idx], y[train_idx], cat_features=categorical),
            eval_set=Pool(x_train.iloc[val_idx], y[val_idx], cat_features=categorical),
            use_best_model=True,
        )
        pred = np.clip(model.predict(Pool(x_train.iloc[val_idx], cat_features=categorical)), 0, None)
        oof[val_idx] = pred
        best_iteration = model.get_best_iteration()
        if best_iteration is not None:
            best_iterations.append(int(best_iteration) + 1)
        fold_reports.append(
            {
                "fold": fold,
                "texts": ",".join(texts),
                "best_iter": best_iterations[-1] if best_iterations else iterations,
                "metric": competition_metric(y[val_idx], pred),
                "r2": r2_score_np(y[val_idx], pred),
                "rmse": rmse(y[val_idx], pred),
            }
        )

    final_iterations = int(np.median(best_iterations)) if best_iterations else iterations
    final_iterations = max(50, min(final_iterations, iterations))
    print(f"Training final CatBoost with {final_iterations} trees", flush=True)
    final_model = make_catboost(iterations=final_iterations, seed=2026)
    final_model.fit(Pool(x_train, y, cat_features=categorical))
    test_pred = np.clip(final_model.predict(Pool(x_test, cat_features=categorical)), 0, None)
    print(pd.DataFrame(fold_reports).to_string(index=False), flush=True)
    return oof, test_pred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_data.csv")
    parser.add_argument("--test", default="test_data.csv")
    parser.add_argument("--sample", default="sample_output.csv")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--audit-output", default="submission_optimized_audit.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--catboost-iterations", type=int, default=450)
    parser.add_argument("--blend-step", type=float, default=0.025)
    args = parser.parse_args(argv)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    train["participant_id"] = train["participant_id"].astype(str).str.zfill(3)
    test["participant_id"] = test["participant_id"].astype(str).str.zfill(3)
    train["answer"] = pd.to_numeric(train["answer"], errors="coerce").fillna(0)
    y = train["answer"].to_numpy(dtype=float)

    eye = load_eye_features()
    props = load_property_features()
    token, lemma = load_complexity_features()
    train_features = build_features(train, eye, props, token, lemma)
    test_features = build_features(test, eye, props, token, lemma)

    source_columns = [
        col
        for col in [
            "direct_trt",
            "multi_trt",
            "subject_mean_trt",
            "subject_median_trt",
            "subject_min_trt",
            "subject_max_trt",
            "subj_008_trt",
            "subj_009_trt",
            "subj_010_trt",
            "subj_011_trt",
            "subj_023_trt",
            "direct_multi_gap",
            "subject_std_trt",
            "subject_zero_rate",
            "word_len",
            "clean_len",
            "page",
            "word_index_from_id",
            "direct_word_index_in_sentence",
            "multi_word_index_in_sentence",
            "prop010_surprisal",
            "prop010_frequency",
            "multi_complexity",
        ]
        if col in train_features.columns
    ]

    linear_oof, linear_test, linear_coef = fit_linear_oof(
        train_features, test_features, y, source_columns, args.n_splits
    )
    print("Linear calibrated source OOF:", competition_metric(y, linear_oof), "RMSE", rmse(y, linear_oof), flush=True)

    x_train, x_test, categorical = prepare_model_frames(train_features, test_features)
    print(f"Model features: {x_train.shape[1]} columns, categorical={categorical}", flush=True)
    cat_oof, cat_test = catboost_oof_and_test(
        x_train,
        x_test,
        y,
        train,
        categorical,
        args.n_splits,
        args.catboost_iterations,
    )

    prediction_oof = {"linear": linear_oof}
    prediction_test = {"linear": linear_test}
    for name in ["direct_trt", "multi_trt", "subject_mean_trt", "subject_median_trt", "eye_best_source"]:
        if name not in train_features.columns:
            continue
        train_values = pd.to_numeric(train_features[name], errors="coerce").fillna(train["answer"].mean()).to_numpy()
        test_values = pd.to_numeric(test_features[name], errors="coerce").fillna(train["answer"].mean()).to_numpy()
        prediction_oof[name] = np.clip(train_values, 0, None)
        prediction_test[name] = np.clip(test_values, 0, None)

    if cat_oof is not None:
        prediction_oof["catboost"] = cat_oof
        prediction_test["catboost"] = cat_test

    print("\nOOF source report", flush=True)
    rows = []
    for name, pred in prediction_oof.items():
        rows.append(
            {
                "name": name,
                "metric": competition_metric(y, pred),
                "r2": r2_score_np(y, pred),
                "rmse": rmse(y, pred),
                "mean": float(np.mean(pred)),
            }
        )
    print(pd.DataFrame(rows).sort_values("metric", ascending=False).to_string(index=False), flush=True)

    weights, score = optimize_ensemble(y, prediction_oof, step=args.blend_step)
    oof_blend = sum(weights[name] * prediction_oof[name] for name in weights)
    test_blend = sum(weights[name] * prediction_test[name] for name in weights)
    test_blend = np.clip(test_blend, 0, None)
    print("\nBest blend:", weights, flush=True)
    print("OOF blend:", score, "R2", r2_score_np(y, oof_blend), "RMSE", rmse(y, oof_blend), flush=True)

    submission = pd.read_csv(args.sample)
    submission["answer"] = test_blend
    submission.to_csv(args.output, index=False)

    audit = test[["datapointID", "word_id", "word", "participant_id", "text"]].copy()
    for name, pred in prediction_test.items():
        audit[name] = pred
    audit["answer"] = test_blend
    audit.to_csv(args.audit_output, index=False)

    direct_word = test_features["direct_word"]
    mismatches = (
        test["word"].map(normalize_word).reset_index(drop=True)
        != direct_word.map(normalize_word).reset_index(drop=True)
    ).sum()
    print(f"Wrote {args.output} and {args.audit_output}", flush=True)
    print(f"Direct source coverage on test: {test_features['direct_trt'].notna().mean():.3f}", flush=True)
    print(f"Direct word mismatches after dash normalization: {int(mismatches)}", flush=True)
    print("First predictions:", flush=True)
    print(submission.head().to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
