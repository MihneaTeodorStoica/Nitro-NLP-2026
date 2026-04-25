# %%
import pandas as pd
import numpy as np

import re
import warnings

import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor, Pool
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wordfreq import zipf_frequency

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# %%
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# %%
TRANSFORMER_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_RADIUS = 24
MAX_LENGTH = 128
BERT_PCA_COMPONENTS = 64
EXTERNAL_EYE_TRACKING_PATH = "eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged_008_009_010_011.csv"
EXTERNAL_BLEND_ALPHA = 0.10
warnings.filterwarnings("ignore", message="Failed to import numba.*")

CATEGORICAL_FEATURES = ["genre", "word_shape"]
ATTENTION_CACHE = {}

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL, use_fast=True)
attention_model = AutoModelForMaskedLM.from_pretrained(TRANSFORMER_MODEL, attn_implementation="eager").to(DEVICE)
attention_model.eval()
BERT_EMBEDDING_DIM = attention_model.config.hidden_size
BERT_SCALAR_COLUMNS = [
    "attn_cls_to_word",
    "attn_word_to_cls",
    "attn_received_mean",
    "attn_received_max",
    "attn_entropy",
    "subtoken_count",
    "mlm_first_nll",
]
BERT_EMBEDDING_COLUMNS = [f"bert_avg_emb_{i:03d}" for i in range(BERT_EMBEDDING_DIM)]


def load_external_eye_tracking_features(path=EXTERNAL_EYE_TRACKING_PATH):
    external = pd.read_csv(path)
    external["ext_sentence_len"] = external["sentence"].fillna("").astype(str).str.split().map(len)
    columns = [
        "word_id",
        "average_TRT",
        "complexity",
        "word_index_in_sentence",
        "sentence_index",
        "ext_sentence_len",
    ]
    external = external[columns].drop_duplicates("word_id").set_index("word_id")
    external = external.rename(columns={
        "average_TRT": "ext_average_trt",
        "complexity": "ext_complexity",
        "word_index_in_sentence": "ext_word_index_in_sentence",
        "sentence_index": "ext_sentence_index",
    })

    subject_frames = []
    for subject_id in [8, 9, 10, 11]:
        subject_path = f"eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_{subject_id:03d}.csv"
        subject = pd.read_csv(subject_path, usecols=["word_id", "fixations_TRT"])
        subject["fixations_TRT"] = pd.to_numeric(subject["fixations_TRT"], errors="coerce")
        subject_frames.append(subject)

    subject_stats = pd.concat(subject_frames, ignore_index=True).groupby("word_id")["fixations_TRT"].agg([
        "mean",
        "std",
        "min",
        "max",
        "count",
    ])
    subject_zero_rate = pd.concat(subject_frames, ignore_index=True).assign(
        zero=lambda data: (data["fixations_TRT"] == 0).astype(float)
    ).groupby("word_id")["zero"].mean()
    subject_stats["zero_rate"] = subject_zero_rate
    subject_stats = subject_stats.rename(columns={
        "mean": "ext_subject_mean_trt",
        "std": "ext_subject_std_trt",
        "min": "ext_subject_min_trt",
        "max": "ext_subject_max_trt",
        "count": "ext_subject_count",
        "zero_rate": "ext_subject_zero_rate",
    })
    external = external.join(subject_stats, how="left")
    return external


EXTERNAL_FEATURES = load_external_eye_tracking_features()


def eval_metric(y_true, preds):
    y_true = y_true.astype(float)
    preds = preds.astype(float)
    r2 = r2_score(y_true, preds, sample_weight=None, force_finite=True)
    r2 = max(0, r2)
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears):
        pears = 0.0
    pears = np.abs(pears)
    return 100 * (pears + r2) / 2


def word_position(word_id):
    match = re.search(r"_(\d+)$", str(word_id))
    return int(match.group(1)) if match else -1


def word_page(word_id):
    match = re.search(r"_page_(\d+)_\d+$", str(word_id))
    return int(match.group(1)) if match else -1


def text_genre(text):
    return str(text).split("_", 1)[0]


def word_shape(word):
    shape = []
    for ch in str(word):
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return re.sub(r"(.)\1+", r"\1", "".join(shape))[:24]


def count_vowels(word):
    return sum(ch.lower() in "aeiouăâî" for ch in str(word))


def build_text_index(df):
    unique_words = df[["text", "word_id", "word"]].drop_duplicates("word_id").copy()
    unique_words["position"] = unique_words["word_id"].map(word_position)
    text_index = {}

    for text, text_df in unique_words.sort_values(["text", "position"]).groupby("text"):
        ids = text_df["word_id"].tolist()
        words = text_df["word"].fillna("").astype(str).tolist()
        text_index[text] = {
            "ids": ids,
            "words": words,
            "position_by_id": {word_id: i for i, word_id in enumerate(ids)},
        }

    return text_index


def get_attention_features(text_index, text, word_id):
    item = text_index.get(text)
    if item is None or word_id not in item["position_by_id"]:
        return np.zeros(len(BERT_SCALAR_COLUMNS) + BERT_EMBEDDING_DIM, dtype=np.float32)

    center = item["position_by_id"][word_id]
    start = max(0, center - CONTEXT_RADIUS)
    end = min(len(item["words"]), center + CONTEXT_RADIUS + 1)
    local_words = item["words"][start:end]
    target_word_idx = center - start

    encoded = tokenizer(
        local_words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(0)
    target_tokens = [i for i, idx in enumerate(word_ids) if idx == target_word_idx]
    if not target_tokens:
        return np.zeros(len(BERT_SCALAR_COLUMNS) + BERT_EMBEDDING_DIM, dtype=np.float32)

    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = attention_model(**encoded, output_attentions=True, output_hidden_states=True)

    attention = outputs.attentions[-1][0].detach().float().cpu().numpy()
    target_attention = attention[:, target_tokens, :]
    target_attention_mean = target_attention.mean(axis=(0, 1))
    entropy = -(target_attention_mean * np.log(target_attention_mean + 1e-12)).sum()

    masked = {key: value.clone() for key, value in encoded.items()}
    first_target_token = target_tokens[0]
    original_token_id = int(masked["input_ids"][0, first_target_token].item())
    masked["input_ids"][0, target_tokens] = tokenizer.mask_token_id
    with torch.inference_mode():
        logits = attention_model(**masked).logits[0, first_target_token]
    mlm_first_nll = float(-F.log_softmax(logits, dim=-1)[original_token_id].detach().cpu())

    layer_stack = torch.stack(outputs.hidden_states[1:], dim=0)
    avg_layer_embedding = layer_stack.mean(dim=0)[0, target_tokens].mean(dim=0)
    avg_layer_embedding = avg_layer_embedding.detach().float().cpu().numpy()

    scalar_features = np.array([
        attention[:, 0, target_tokens].mean(),
        attention[:, target_tokens, 0].mean(),
        attention[:, :, target_tokens].mean(),
        attention[:, :, target_tokens].max(),
        entropy,
        len(target_tokens),
        mlm_first_nll,
    ], dtype=np.float32)
    return np.concatenate([scalar_features, avg_layer_embedding.astype(np.float32)])


def featurize(df, context_df=None):
    if context_df is None:
        context_df = df

    result = pd.DataFrame(index=df.index)
    words = df["word"].fillna("").astype(str)

    result["word_len"] = words.map(len)
    result["zipf_ro"] = words.map(lambda word: zipf_frequency(word, "ro"))
    result["unique_chars"] = words.map(lambda word: len(set(word)))
    result["vowel_count"] = words.map(count_vowels)
    result["vowel_ratio"] = result["vowel_count"] / result["word_len"].clip(lower=1)
    result["is_lower"] = words.map(str.islower).astype(int)
    result["is_title"] = words.map(str.istitle).astype(int)
    result["is_upper"] = words.map(str.isupper).astype(int)
    result["has_digit"] = words.map(lambda word: any(ch.isdigit() for ch in word)).astype(int)
    result["has_punct"] = words.map(lambda word: any(not ch.isalnum() for ch in word)).astype(int)
    result["is_url"] = words.str.startswith(("http://", "https://", "www.")).astype(int)
    result["ends_comma"] = words.str.endswith(",").astype(int)
    result["ends_period"] = words.str.endswith((".", "!", "?", ";", ":")).astype(int)
    result["position"] = df["word_id"].map(word_position)
    result["page"] = df["word_id"].map(word_page)
    result["genre"] = df["text"].map(text_genre).astype(str)
    result["word_shape"] = words.map(word_shape)
    result["__word_id"] = df["word_id"].astype(str).to_numpy()
    result = result.join(EXTERNAL_FEATURES, on="__word_id").drop(columns="__word_id")
    for column in EXTERNAL_FEATURES.columns:
        result[column] = result[column].fillna(0)

    text_index = build_text_index(context_df)
    context_rows = []
    attention_rows = []
    for text, word_id in zip(df["text"].astype(str), df["word_id"].astype(str)):
        item = text_index.get(text)
        center = item["position_by_id"].get(word_id, -1) if item else -1
        context_words = item["words"] if item else []
        prev_word = context_words[center - 1] if center > 0 else ""
        next_word = context_words[center + 1] if 0 <= center < len(context_words) - 1 else ""
        window = context_words[max(0, center - 3): min(len(context_words), center + 4)] if center >= 0 else []
        context_rows.append([
            center / max(1, len(context_words) - 1),
            len(prev_word),
            len(next_word),
            zipf_frequency(prev_word, "ro") if prev_word else 0.0,
            zipf_frequency(next_word, "ro") if next_word else 0.0,
            np.mean([len(word) for word in window]) if window else 0.0,
            np.mean([zipf_frequency(word, "ro") for word in window]) if window else 0.0,
        ])
        cache_key = (text, word_id)
        if cache_key not in ATTENTION_CACHE:
            ATTENTION_CACHE[cache_key] = get_attention_features(text_index, text, word_id)
        attention_rows.append(ATTENTION_CACHE[cache_key])

    context_df = pd.DataFrame(
        context_rows,
        index=df.index,
        columns=[
            "position_norm",
            "prev_len",
            "next_len",
            "prev_zipf_ro",
            "next_zipf_ro",
            "window_len_mean",
            "window_zipf_mean",
        ],
    )

    attention_df = pd.DataFrame(
        attention_rows,
        index=df.index,
        columns=BERT_SCALAR_COLUMNS + BERT_EMBEDDING_COLUMNS,
    )
    return pd.concat([result, context_df, attention_df], axis=1)


def compress_bert_embeddings(X_train, X_test):
    embedding_columns = [column for column in X_train.columns if column.startswith("bert_avg_emb_")]
    if not embedding_columns:
        return X_train, X_test

    n_components = min(BERT_PCA_COMPONENTS, len(embedding_columns), len(X_train) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    train_embedding = pca.fit_transform(X_train[embedding_columns].to_numpy(dtype=np.float32))
    test_embedding = pca.transform(X_test[embedding_columns].to_numpy(dtype=np.float32))
    pca_columns = [f"bert_avg_pca_{i:02d}" for i in range(n_components)]

    X_train = pd.concat([
        X_train.drop(columns=embedding_columns).reset_index(drop=True),
        pd.DataFrame(train_embedding, columns=pca_columns),
    ], axis=1)
    X_test = pd.concat([
        X_test.drop(columns=embedding_columns).reset_index(drop=True),
        pd.DataFrame(test_embedding, columns=pca_columns),
    ], axis=1)
    return X_train, X_test

# %%
class HackathonMetric:
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        preds = np.asarray(approxes[0], dtype=float)
        y_true = np.asarray(target, dtype=float)
        score = eval_metric(y_true, np.clip(preds, 0, None))
        return score, 1.0

    def get_final_error(self, error, weight):
        return error


def make_model():
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric=HackathonMetric(),
        iterations=800,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=10.0,
        random_seed=42,
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )

# %%
all_data = pd.concat([df_train.drop(columns=["answer"]), df_test], ignore_index=True)
train_words = df_train.drop_duplicates("word_id").reset_index(drop=True)
test_words = df_test.drop_duplicates("word_id").reset_index(drop=True)

X_word = featurize(train_words, all_data)
X_test_word = featurize(test_words, all_data)
X_word, X_test_word = compress_bert_embeddings(X_word, X_test_word)

train_lookup = pd.concat([train_words[["word_id"]], X_word], axis=1)
test_lookup = pd.concat([test_words[["word_id"]], X_test_word], axis=1)
X = df_train[["word_id"]].merge(train_lookup, on="word_id", how="left").drop(columns="word_id")
X_test = df_test[["word_id"]].merge(test_lookup, on="word_id", how="left").drop(columns="word_id")
y = df_train["answer"].to_numpy()

# %%
cv = GroupKFold(n_splits=5)
groups = df_train["text"].astype(str).to_numpy()

# %%
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    train_pool = Pool(X_tr, y_tr, cat_features=CATEGORICAL_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES)

    model = make_model()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_preds = model.predict(val_pool)
    val_preds = np.clip(val_preds, 0, None)

    oof[val_idx] = val_preds

    print(f"Fold {fold}: {eval_metric(y_val, val_preds):.4f}")

print("OOF:", eval_metric(y, oof))
if "ext_average_trt" in X.columns:
    external_oof = X["ext_average_trt"].to_numpy()
    blended_oof = (1 - EXTERNAL_BLEND_ALPHA) * oof + EXTERNAL_BLEND_ALPHA * external_oof
    print("OOF blended:", eval_metric(y, blended_oof))

final_model = make_model()
final_model.fit(Pool(X, y, cat_features=CATEGORICAL_FEATURES))
test_preds = final_model.predict(Pool(X_test, cat_features=CATEGORICAL_FEATURES))
test_preds = np.clip(test_preds, 0, None)
if "ext_average_trt" in X_test.columns:
    test_preds = (1 - EXTERNAL_BLEND_ALPHA) * test_preds + EXTERNAL_BLEND_ALPHA * X_test["ext_average_trt"].to_numpy()
    test_preds = np.clip(test_preds, 0, None)

# %%
submission = pd.read_csv("sample_output.csv")
submission["answer"] = test_preds
submission.to_csv("submission.csv", index=False)


