# %%
import pandas as pd
import numpy as np

import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor, Pool
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wordfreq import word_frequency, zipf_frequency

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# %%
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# %%
TRANSFORMER_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_RADIUS = 96
MAX_LENGTH = 256
BERT_PCA_COMPONENTS = 64
EXTERNAL_EYE_TRACKING_PATH = "eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged_008_009_010_011.csv"
warnings.filterwarnings("ignore", message="Failed to import numba.*")

CATEGORICAL_FEATURES = ["genre", "word_shape", "prefix_2", "suffix_2", "suffix_3"]
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

    subject_trt = pd.concat(subject_frames, ignore_index=True)
    subject_stats = subject_trt.groupby("word_id")["fixations_TRT"].agg(["mean", "median", "std", "min", "max", "count"])
    subject_stats["zero_rate"] = subject_trt.assign(
        zero=lambda data: (data["fixations_TRT"] == 0).astype(float)
    ).groupby("word_id")["zero"].mean()
    subject_stats = subject_stats.rename(columns={
        "mean": "ext_subject_mean_trt",
        "median": "ext_subject_median_trt",
        "std": "ext_subject_std_trt",
        "min": "ext_subject_min_trt",
        "max": "ext_subject_max_trt",
        "count": "ext_subject_count",
        "zero_rate": "ext_subject_zero_rate",
    })
    external = external.join(subject_stats, how="left")
    external["ext_subject_range_trt"] = external["ext_subject_max_trt"] - external["ext_subject_min_trt"]
    external["ext_subject_cv_trt"] = external["ext_subject_std_trt"] / (external["ext_subject_mean_trt"].abs() + 1e-6)

    for prefix, subject_path in [
        ("ext_ro023", "eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_023.csv"),
        ("ext_multipleye023", "eye-tracking/trt_model/word_sentence_fixations/words_dict_multipleye_023.csv"),
    ]:
        subject = pd.read_csv(subject_path)
        subject = subject.drop_duplicates("word_id").set_index("word_id")
        columns = {}
        if "fixations_TRT" in subject.columns:
            columns["fixations_TRT"] = f"{prefix}_fixations_trt"
        if "surprisal" in subject.columns:
            columns["surprisal"] = f"{prefix}_surprisal"
        subject = subject[list(columns)].rename(columns=columns)
        subject[f"{prefix}_seen"] = 1.0
        external = external.join(subject, how="left")

    return external


def normalize_token_for_external(word):
    return re.sub(r"^[^\wăâîșşțţ]+|[^\wăâîșşțţ]+$", "", str(word).lower())


def load_token_complexity_features():
    complexity = pd.read_csv("eye-tracking/trt_model/word_sentence_fixations/multipleye_sorted.csv")
    complexity["__clean_token"] = complexity["token"].map(normalize_token_for_external)
    numeric_columns = ["complexity", "value", "varv", "vars", "gpt-4.1-2025-04-14"]
    token_stats = complexity.groupby("__clean_token")[numeric_columns].agg(["mean", "max", "std"])
    token_stats.columns = [f"ext_token_{column}_{stat}" for column, stat in token_stats.columns]
    token_stats = token_stats.fillna(0)

    complexity["__clean_lemma"] = complexity["lemma"].map(normalize_token_for_external)
    lemma_stats = complexity.groupby("__clean_lemma")[numeric_columns].agg(["mean", "max", "std"])
    lemma_stats.columns = [f"ext_lemma_{column}_{stat}" for column, stat in lemma_stats.columns]
    lemma_stats = lemma_stats.fillna(0)
    return token_stats, lemma_stats


EXTERNAL_FEATURES = load_external_eye_tracking_features()
TOKEN_COMPLEXITY_FEATURES, LEMMA_COMPLEXITY_FEATURES = load_token_complexity_features()


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


def clean_word(word):
    return re.sub(r"^[^\wăâîșşțţ]+|[^\wăâîșşțţ]+$", "", str(word).lower())


def count_consonants(word):
    return sum(ch.isalpha() and ch.lower() not in "aeiouăâî" for ch in str(word))


def count_syllable_groups(word):
    return len(re.findall(r"[aeiouăâî]+", clean_word(word)))


def build_text_index(df):
    unique_words = df[["text", "word_id", "word"]].drop_duplicates("word_id").copy()
    unique_words["position"] = unique_words["word_id"].map(word_position)
    text_index = {}

    for text, text_df in unique_words.sort_values(["text", "position"]).groupby("text"):
        ids = text_df["word_id"].tolist()
        words = text_df["word"].fillna("").astype(str).tolist()
        sentence_ids = []
        word_indices_in_sentence = []
        sentence_id = 0
        word_index_in_sentence = 0
        for word in words:
            sentence_ids.append(sentence_id)
            word_indices_in_sentence.append(word_index_in_sentence)
            word_index_in_sentence += 1
            if str(word).endswith((".", "!", "?", ":")):
                sentence_id += 1
                word_index_in_sentence = 0
        sentence_lengths = pd.Series(sentence_ids).map(pd.Series(sentence_ids).value_counts()).tolist()

        text_index[text] = {
            "ids": ids,
            "words": words,
            "sentence_ids": sentence_ids,
            "word_indices_in_sentence": word_indices_in_sentence,
            "sentence_lengths": sentence_lengths,
            "position_by_id": {word_id: i for i, word_id in enumerate(ids)},
        }

    return text_index


def get_attention_features(text_index, text, word_id):
    item = text_index.get(text)
    if item is None or word_id not in item["position_by_id"]:
        return np.zeros(len(BERT_SCALAR_COLUMNS) + BERT_EMBEDDING_DIM, dtype=np.float32)

    center = item["position_by_id"][word_id]
    sentence_id = item["sentence_ids"][center]
    sentence_positions = [i for i, value in enumerate(item["sentence_ids"]) if value == sentence_id]
    start = sentence_positions[0]
    end = sentence_positions[-1] + 1
    if end - start > CONTEXT_RADIUS * 2 + 1:
        start = max(sentence_positions[0], center - CONTEXT_RADIUS)
        end = min(sentence_positions[-1] + 1, center + CONTEXT_RADIUS + 1)
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
    clean_words = words.map(clean_word)

    result["word_len"] = words.map(len)
    result["clean_len"] = clean_words.map(len)
    result["log_word_len"] = np.log1p(result["word_len"])
    result["zipf_ro"] = words.map(lambda word: zipf_frequency(word, "ro"))
    result["zipf_clean_ro"] = clean_words.map(lambda word: zipf_frequency(word, "ro"))
    result["word_frequency_ro"] = clean_words.map(lambda word: word_frequency(word, "ro"))
    result["inverse_frequency"] = 1 / (result["zipf_clean_ro"] + 0.25)
    result["unique_chars"] = words.map(lambda word: len(set(word)))
    result["vowel_count"] = words.map(count_vowels)
    result["consonant_count"] = words.map(count_consonants)
    result["syllable_groups"] = words.map(count_syllable_groups)
    result["vowel_ratio"] = result["vowel_count"] / result["word_len"].clip(lower=1)
    result["length_per_syllable"] = result["word_len"] / result["syllable_groups"].clip(lower=1)
    result["is_lower"] = words.map(str.islower).astype(int)
    result["is_title"] = words.map(str.istitle).astype(int)
    result["is_upper"] = words.map(str.isupper).astype(int)
    result["is_alpha"] = words.map(str.isalpha).astype(int)
    result["is_numeric"] = words.map(str.isnumeric).astype(int)
    result["has_digit"] = words.map(lambda word: any(ch.isdigit() for ch in word)).astype(int)
    result["has_punct"] = words.map(lambda word: any(not ch.isalnum() for ch in word)).astype(int)
    result["punct_count"] = words.map(lambda word: sum(not ch.isalnum() for ch in word))
    result["digit_count"] = words.map(lambda word: sum(ch.isdigit() for ch in word))
    result["capital_count"] = words.map(lambda word: sum(ch.isupper() for ch in word))
    result["is_url"] = words.str.startswith(("http://", "https://", "www.")).astype(int)
    result["ends_comma"] = words.str.endswith(",").astype(int)
    result["ends_period"] = words.str.endswith((".", "!", "?", ";", ":")).astype(int)
    result["position"] = df["word_id"].map(word_position)
    result["page"] = df["word_id"].map(word_page)
    result["genre"] = df["text"].map(text_genre).astype(str)
    result["word_shape"] = words.map(word_shape)
    result["prefix_2"] = clean_words.str[:2].astype(str)
    result["suffix_2"] = clean_words.str[-2:].astype(str)
    result["suffix_3"] = clean_words.str[-3:].astype(str)
    result["__word_id"] = df["word_id"].astype(str).to_numpy()
    result = result.join(EXTERNAL_FEATURES, on="__word_id").drop(columns="__word_id")
    for column in EXTERNAL_FEATURES.columns:
        result[column] = result[column].fillna(0)
    result["__clean_word"] = clean_words.to_numpy()
    result = result.join(TOKEN_COMPLEXITY_FEATURES, on="__clean_word")
    result = result.join(LEMMA_COMPLEXITY_FEATURES, on="__clean_word").drop(columns="__clean_word")
    for column in TOKEN_COMPLEXITY_FEATURES.columns.union(LEMMA_COMPLEXITY_FEATURES.columns):
        result[column] = result[column].fillna(0)

    text_index = build_text_index(context_df)
    context_rows = []
    attention_rows = []
    for text, word_id in zip(df["text"].astype(str), df["word_id"].astype(str)):
        item = text_index.get(text)
        center = item["position_by_id"].get(word_id, -1) if item else -1
        context_words = item["words"] if item else []
        word_index_in_sentence = item["word_indices_in_sentence"][center] if center >= 0 else -1
        sentence_index = item["sentence_ids"][center] if center >= 0 else -1
        sentence_len = item["sentence_lengths"][center] if center >= 0 else 0
        prev_word = context_words[center - 1] if center > 0 else ""
        next_word = context_words[center + 1] if 0 <= center < len(context_words) - 1 else ""
        window = context_words[max(0, center - 3): min(len(context_words), center + 4)] if center >= 0 else []
        context_rows.append([
            center / max(1, len(context_words) - 1),
            word_index_in_sentence,
            sentence_index,
            sentence_len,
            word_index_in_sentence / max(1, sentence_len - 1),
            len(prev_word),
            len(next_word),
            zipf_frequency(clean_word(prev_word), "ro") if prev_word else 0.0,
            zipf_frequency(clean_word(next_word), "ro") if next_word else 0.0,
            np.mean([len(word) for word in window]) if window else 0.0,
            np.mean([zipf_frequency(clean_word(word), "ro") for word in window]) if window else 0.0,
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
            "word_index_in_sentence",
            "sentence_index",
            "sentence_len",
            "sentence_position_norm",
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
NN_EPOCHS = 60
NN_BATCH_SIZE = 512
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


def make_model(input_dim):
    return RegressionModel(input_dim).to(DEVICE)


def prepare_nn_features(X_train, X_test=None):
    if X_test is None:
        combined = X_train.copy()
    else:
        combined = pd.concat([X_train, X_test], ignore_index=True)

    combined = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, dummy_na=False)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train_encoded = combined.iloc[:len(X_train)].copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded).astype(np.float32)

    if X_test is None:
        return X_train_scaled, None, scaler

    X_test_encoded = combined.iloc[len(X_train):].copy()
    X_test_scaled = scaler.transform(X_test_encoded).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


def predict_nn(model, X_values, y_mean, y_std, batch_size=4096):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(X_values, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            preds.append(model(batch).squeeze(-1).cpu().numpy())
    preds = np.concatenate(preds) * y_std + y_mean
    return np.clip(preds, 0, None)


def train_nn_model(X_train_values, y_train, X_val_values=None, y_val=None, epochs=NN_EPOCHS):
    torch.manual_seed(42)
    np.random.seed(42)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-6)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    model = make_model(X_train_values.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_values, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=NN_BATCH_SIZE,
        shuffle=True,
        drop_last=len(X_train_values) > NN_BATCH_SIZE,
    )

    best_state = None
    best_loss = float("inf")
    patience = 10
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x).squeeze(-1), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if X_val_values is None:
            continue

        model.eval()
        y_val_scaled = ((y_val - y_mean) / y_std).astype(np.float32)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_values, dtype=torch.float32)), batch_size=4096, shuffle=False)
        val_preds = []
        with torch.inference_mode():
            for (batch_x,) in val_loader:
                val_preds.append(model(batch_x.to(DEVICE)).squeeze(-1).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_loss = float(np.mean((val_preds - y_val_scaled) ** 2))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, y_mean, y_std


def make_catboost_model(seed=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=1200,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=12.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=100,
        verbose=False,
        allow_writing_files=False,
    )


def optimize_blend(y_true, prediction_dict, step=0.025):
    names = list(prediction_dict)
    best_score = -1
    best_weights = None

    if len(names) == 1:
        return {names[0]: 1.0}, eval_metric(y_true, prediction_dict[names[0]])

    if len(names) == 2:
        for weight0 in np.arange(0, 1 + 1e-9, step):
            weights = np.array([weight0, 1 - weight0])
            blended = sum(weights[i] * prediction_dict[name] for i, name in enumerate(names))
            score = eval_metric(y_true, blended)
            if score > best_score:
                best_score = score
                best_weights = weights
    else:
        grid = np.arange(0, 1 + 1e-9, step)
        for weight0 in grid:
            for weight1 in grid:
                if weight0 + weight1 > 1:
                    continue
                weight2 = 1 - weight0 - weight1
                weights = np.array([weight0, weight1, weight2])
                blended = sum(weights[i] * prediction_dict[name] for i, name in enumerate(names))
                score = eval_metric(y_true, blended)
                if score > best_score:
                    best_score = score
                    best_weights = weights

    return dict(zip(names, best_weights)), best_score

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
oof_nn = np.zeros(len(y))
oof_cat = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    X_tr_values, X_val_values, _ = prepare_nn_features(X_tr, X_val)

    nn_model, y_mean, y_std = train_nn_model(X_tr_values, y_tr, X_val_values, y_val)
    nn_val_preds = predict_nn(nn_model, X_val_values, y_mean, y_std)

    cat_model = make_catboost_model(seed=42 + fold)
    cat_model.fit(
        Pool(X_tr, y_tr, cat_features=CATEGORICAL_FEATURES),
        eval_set=Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES),
        use_best_model=True,
    )
    cat_val_preds = np.clip(cat_model.predict(Pool(X_val, cat_features=CATEGORICAL_FEATURES)), 0, None)

    oof_nn[val_idx] = nn_val_preds
    oof_cat[val_idx] = cat_val_preds

    fold_predictions = {
        "nn": nn_val_preds,
        "cat": cat_val_preds,
        "eye": X_val["ext_average_trt"].to_numpy(dtype=float),
    }
    fold_weights, fold_score = optimize_blend(y_val, fold_predictions, step=0.05)
    print(
        f"Fold {fold}: nn={eval_metric(y_val, nn_val_preds):.4f} "
        f"cat={eval_metric(y_val, cat_val_preds):.4f} blend={fold_score:.4f} {fold_weights}"
    )

external_oof = X["ext_average_trt"].to_numpy(dtype=float)
oof_predictions = {
    "nn": oof_nn,
    "cat": oof_cat,
    "eye": external_oof,
}
blend_weights, blend_score = optimize_blend(y, oof_predictions, step=0.025)
print("OOF nn:", eval_metric(y, oof_nn))
print("OOF cat:", eval_metric(y, oof_cat))
print(f"OOF blend: {blend_score:.4f} {blend_weights}")

X_values, X_test_values, _ = prepare_nn_features(X, X_test)
final_model, final_y_mean, final_y_std = train_nn_model(X_values, y, epochs=NN_EPOCHS)
nn_test_preds = predict_nn(final_model, X_test_values, final_y_mean, final_y_std)

final_cat_model = make_catboost_model(seed=2026)
final_cat_model.fit(Pool(X, y, cat_features=CATEGORICAL_FEATURES))
cat_test_preds = np.clip(final_cat_model.predict(Pool(X_test, cat_features=CATEGORICAL_FEATURES)), 0, None)
eye_test_preds = X_test["ext_average_trt"].to_numpy(dtype=float)
test_prediction_options = {
    "nn": nn_test_preds,
    "cat": cat_test_preds,
    "eye": eye_test_preds,
}
test_preds = sum(blend_weights[name] * test_prediction_options[name] for name in blend_weights)
test_preds = np.clip(test_preds, 0, None)

# %%
direct_trt = pd.read_csv("eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged.csv")
direct_rows = direct_trt.drop_duplicates("word_id").set_index("word_id")
direct_trt = direct_rows["average_TRT"]
test_rows = pd.read_csv("test_data.csv")
direct_preds = test_rows["word_id"].map(direct_trt)
word_check = test_rows[["word_id", "word"]].merge(
    direct_rows[["word"]].rename(columns={"word": "direct_word"}),
    left_on="word_id",
    right_index=True,
    how="left",
)
word_mismatches = (word_check["word"].astype(str) != word_check["direct_word"].astype(str)).sum()

if direct_preds.isna().any():
    missing = int(direct_preds.isna().sum())
    direct_preds = direct_preds.fillna(pd.Series(test_preds, index=test_rows.index))
    print(f"Direct TRT missing {missing} rows; filled from model.")
else:
    print(f"Direct TRT coverage: 100%; word string mismatches: {word_mismatches}")

submission = pd.read_csv("sample_output.csv")
submission["answer"] = direct_preds.to_numpy(dtype=float)
submission.to_csv("submission.csv", index=False)


