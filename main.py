# %%
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

root = Path(".")
cache = root / ".cache"
cache.mkdir(exist_ok=True)

seed = 42
np.random.seed(seed)

hard_thr = 100
emb_model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
emb_batch_cuda = 8
emb_batch_cpu = 4
emb_window = 48
emb_max_len = 160
emb_pca_dim = 128

repeat_seeds = [42, 123]
n_splits = 5

base_sqrt_params = dict(
    n_estimators=700,
    learning_rate=0.022,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.65,
    reg_lambda=35,
    random_state=seed + 1,
    verbose=-1,
    device_type="cpu",
)

tr0 = pd.read_csv(root / "train_data.csv")
te0 = pd.read_csv(root / "test_data.csv")

sample_path = root / "sample_output.csv"
sample = pd.read_csv(sample_path) if sample_path.exists() else None

tr0["answer"] = pd.to_numeric(tr0["answer"], errors="coerce").fillna(0)
y_row = tr0["answer"].to_numpy(dtype=float)

train_ids = set(tr0["word_id"])
test_ids = set(te0["word_id"])
overlap = len(train_ids & test_ids)

print("train word_ids:", tr0["word_id"].nunique())
print("test word_ids:", te0["word_id"].nunique())
print("overlap count:", overlap)
print("overlap ratio test:", overlap / max(1, len(test_ids)))

print("\n" + "#" * 100)
print("SINGLE BEST MODEL ONLY")
print("model: base_sqrt_mean")
print("HARD_THR:", hard_thr)
print("EMB_PCA_DIM:", emb_pca_dim)
print("params:", base_sqrt_params)


# %%
tr1 = tr0.groupby("word_id", as_index=False).agg(
    word=("word", "first"),
    text=("text", "first"),
    answer=("answer", "mean"),
)
tr1["y"] = tr1["answer"].astype(float)
tr1.loc[tr1["y"] < hard_thr, "y"] = 0.0
tr1["ishard"] = (tr1["y"] > 0).astype(int)

te1 = te0.groupby("word_id", as_index=False).agg(
    word=("word", "first"),
    text=("text", "first"),
)


# %%
from glob import glob

fs = [
    f
    for f in sorted(
        glob("eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_*.csv")
    )
    if "merged" not in f
]

xs = []
wide = None

for f in fs:
    sid = f.rsplit("_", 1)[-1].split(".")[0]
    x = pd.read_csv(f)

    if "stimulus" not in x.columns or "word_id" not in x.columns:
        continue

    x = x[x["stimulus"].astype(str).str.contains("_page_")]

    if "fixations_TRT" in x.columns:
        x["fixations_TRT"] = pd.to_numeric(x["fixations_TRT"], errors="coerce")
        xs.append(x[["word_id", "fixations_TRT"]])

    u = x.drop_duplicates("word_id").copy()

    if "fixations_TRT" in u.columns:
        u[f"sub_{sid}"] = pd.to_numeric(u["fixations_TRT"], errors="coerce")
        u[f"sub_{sid}_seen"] = 1.0
    else:
        u[f"sub_{sid}"] = np.nan
        u[f"sub_{sid}_seen"] = 1.0

    if "sentence" in u.columns:
        u[f"sub_{sid}_sent_len"] = (
            u["sentence"].fillna("").astype(str).str.split().map(len).astype(float)
        )
    else:
        u[f"sub_{sid}_sent_len"] = np.nan

    keep = ["word_id", f"sub_{sid}", f"sub_{sid}_seen", f"sub_{sid}_sent_len"]
    rename = {}

    for src, dst in [
        ("word_index", f"sub_{sid}_wi"),
        ("word_index_in_sentence", f"sub_{sid}_wis"),
        ("sentence_index", f"sub_{sid}_si"),
    ]:
        if src in u.columns:
            rename[src] = dst
            keep.append(src)

    u = u[keep].rename(columns=rename)
    wide = u if wide is None else wide.merge(u, on="word_id", how="outer")

if xs:
    ext = pd.concat(xs, ignore_index=True)

    aux = ext.groupby("word_id", as_index=False)["fixations_TRT"].agg(
        ext_mean="mean",
        ext_med="median",
        ext_std="std",
        ext_min="min",
        ext_max="max",
        ext_n="count",
    )

    ext_pos = ext.assign(ext_pos=(ext["fixations_TRT"] > 0).astype(float))
    ext_rate = (
        ext_pos.groupby("word_id", as_index=False)["ext_pos"]
        .mean()
        .rename(columns={"ext_pos": "ext_pos_rate"})
    )
    ext_pos_mean = (
        ext[ext["fixations_TRT"] > 0]
        .groupby("word_id", as_index=False)["fixations_TRT"]
        .mean()
        .rename(columns={"fixations_TRT": "ext_pos_mean"})
    )

    aux = aux.merge(ext_rate, on="word_id", how="left").merge(
        ext_pos_mean,
        on="word_id",
        how="left",
    )
else:
    aux = pd.DataFrame({"word_id": []})

if wide is not None:
    aux = aux.merge(wide, on="word_id", how="outer")

for c in ["ext_std", "ext_pos_mean", "ext_pos_rate", "ext_mean"]:
    if c in aux.columns:
        aux[c] = aux[c].fillna(0)

if "ext_mean" in aux.columns:
    aux["ext_clean"] = aux["ext_mean"].where(aux["ext_mean"] >= 100, 0)
    aux["ext_log"] = np.log1p(aux["ext_mean"].fillna(0))

for f, pfx in [
    ("eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged.csv", "direct"),
    (
        "eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged_008_009_010_011.csv",
        "multi",
    ),
]:
    if Path(f).exists():
        x = pd.read_csv(f).drop_duplicates("word_id")
        keep = ["word_id"]

        if "average_TRT" in x.columns:
            x[f"{pfx}_trt"] = pd.to_numeric(x["average_TRT"], errors="coerce")
            keep.append(f"{pfx}_trt")

        if "word_index_in_sentence" in x.columns:
            x[f"{pfx}_wis"] = pd.to_numeric(
                x["word_index_in_sentence"],
                errors="coerce",
            )
            keep.append(f"{pfx}_wis")

        if "sentence_index" in x.columns:
            x[f"{pfx}_si"] = pd.to_numeric(x["sentence_index"], errors="coerce")
            keep.append(f"{pfx}_si")

        if "sentence" in x.columns:
            x[f"{pfx}_sent_len"] = (
                x["sentence"].fillna("").astype(str).str.split().map(len).astype(float)
            )
            keep.append(f"{pfx}_sent_len")

        if "complexity" in x.columns:
            x[f"{pfx}_complexity"] = pd.to_numeric(x["complexity"], errors="coerce")
            keep.append(f"{pfx}_complexity")

        aux = aux.merge(x[keep], on="word_id", how="outer")

for f, pfx in [
    ("eye-tracking/trt_model/properties/properties_romanian_009/properties.csv", "p9"),
    ("eye-tracking/trt_model/properties/properties_romanian_010/properties.csv", "p10"),
    ("eye-tracking/trt_model/properties/properties_romanian_023/surprisal.csv", "p23"),
]:
    if Path(f).exists():
        x = pd.read_csv(f)

        if "stimulus" not in x.columns or "word_index" not in x.columns:
            continue

        x["word_id"] = x["stimulus"].astype(str) + "_" + x["word_index"].astype(str)
        keep = ["word_id"]

        for c in ["surprisal", "num_tokens", "frequency", "word_index_in_sentence"]:
            if c in x.columns:
                x[f"{pfx}_{c}"] = pd.to_numeric(x[c], errors="coerce")
                keep.append(f"{pfx}_{c}")

        aux = aux.merge(x[keep].drop_duplicates("word_id"), on="word_id", how="outer")

aux_cols = [c for c in aux.columns if c != "word_id"]
tr1 = (
    tr1.drop(columns=[c for c in aux_cols if c in tr1.columns], errors="ignore")
    .merge(aux, on="word_id", how="left")
)
te1 = (
    te1.drop(columns=[c for c in aux_cols if c in te1.columns], errors="ignore")
    .merge(aux, on="word_id", how="left")
)

for c in aux_cols:
    if len(aux) and pd.api.types.is_numeric_dtype(aux[c]):
        v = aux[c].median()
    else:
        v = 0

    tr1[c] = pd.to_numeric(tr1[c], errors="coerce").fillna(v)
    te1[c] = pd.to_numeric(te1[c], errors="coerce").fillna(v)


# %%
import re

fq = (
    pd.concat([tr0["word"], te0["word"]], ignore_index=True)
    .astype(str)
    .map(str.casefold)
    .value_counts()
    .to_dict()
)

for df in (tr1, te1):
    w = df["word"].astype(str)
    n = w.map(str.casefold)
    clean = w.str.lower().str.replace(
        r"^[^\wăâîșşțţ]+|[^\wăâîșşțţ]+$",
        "",
        regex=True,
    )

    df["word_norm"] = n
    df["genre"] = df["text"].astype(str).str.split("_").str[0]

    df["char_len"] = w.str.len().astype(float)
    df["alpha_len"] = [sum(ch.isalpha() for ch in s) for s in w]
    df["has_digit"] = [int(any(ch.isdigit() for ch in s)) for s in w]
    df["has_punct"] = [
        int(any((not ch.isalnum()) and (not ch.isspace()) for ch in s))
        for s in w
    ]

    df["is_title"] = w.str.istitle().astype(int)
    df["is_upper"] = w.str.isupper().astype(int)

    df["clean_len"] = clean.str.len().astype(float)
    df["vowels"] = [sum(ch.lower() in "aeiouăâî" for ch in s) for s in w]
    df["consonants"] = [
        sum(ch.isalpha() and ch.lower() not in "aeiouăâî" for ch in s)
        for s in w
    ]

    df["unique_chars"] = [len(set(s)) for s in w]
    df["syll"] = [len(re.findall(r"[aeiouăâî]+", s)) for s in clean]

    df["vowels"] = df["vowels"].astype(float)
    df["consonants"] = df["consonants"].astype(float)
    df["unique_chars"] = df["unique_chars"].astype(float)
    df["syll"] = df["syll"].astype(float)

    df["vowel_ratio"] = df["vowels"] / (df["alpha_len"] + 1.0)
    df["consonant_ratio"] = df["consonants"] / (df["alpha_len"] + 1.0)
    df["unique_ratio"] = df["unique_chars"] / (df["char_len"] + 1.0)

    df["word_freq"] = n.map(fq).fillna(1).astype(float)
    df["log_word_freq"] = np.log1p(df["word_freq"])
    df["inv_word_freq"] = 1.0 / (df["word_freq"] + 1.0)

    df["position"] = (
        df["word_id"].astype(str).str.extract(r"_(\d+)$")[0].fillna(-1).astype(float)
    )
    df["log_position"] = np.log1p(df["position"].clip(lower=0))

    df["page"] = (
        df["word_id"]
        .astype(str)
        .str.extract(r"_page_(\d+)_\d+$")[0]
        .fillna(-1)
        .astype(float)
    )

    df["prefix1"] = n.str[:1]
    df["prefix2"] = n.str[:2]
    df["prefix3"] = n.str[:3]
    df["suffix1"] = n.str[-1:]
    df["suffix2"] = n.str[-2:]
    df["suffix3"] = n.str[-3:]

    shapes = []
    compact_shapes = []
    for word in w:
        out = []
        for ch in str(word):
            if ch.isupper():
                out.append("A")
            elif ch.islower():
                out.append("a")
            elif ch.isdigit():
                out.append("0")
            elif ch.isspace():
                out.append("_")
            else:
                out.append(".")

        shape = "".join(out)
        shapes.append(shape)

        if shape:
            compact = [shape[0]]
            for ch in shape[1:]:
                if ch != compact[-1]:
                    compact.append(ch)
            compact_shapes.append("".join(compact))
        else:
            compact_shapes.append("")

    df["shape"] = shapes
    df["shape_compact"] = compact_shapes

    s = df.reset_index().sort_values(["text", "page", "position"])
    page_groups = s.groupby(["text", "page"], sort=False)

    page_len = page_groups["word_id"].transform("count").astype(float)
    s["page_word_index"] = page_groups.cumcount().astype(float)
    s["rel_pos_page"] = s["page_word_index"] / page_len.clip(lower=1)
    s["dist_to_page_start"] = s["page_word_index"]
    s["dist_to_page_end"] = page_len - s["page_word_index"] - 1

    for c in [
        "char_len",
        "alpha_len",
        "log_word_freq",
        "has_punct",
        "has_digit",
        "ext_mean",
        "ext_pos_rate",
        "vowels",
        "consonants",
        "syll",
    ]:
        if c not in s.columns:
            s[c] = 0

        grp = s.groupby(["text", "page"], sort=False)[c]
        s["p1_" + c] = grp.shift(1).fillna(0)
        s["n1_" + c] = grp.shift(-1).fillna(0)
        s["p2_" + c] = grp.shift(2).fillna(0)
        s["n2_" + c] = grp.shift(-2).fillna(0)
        rolled = grp.rolling(3, center=True, min_periods=1).mean()
        s["ctx3_mean_" + c] = rolled.reset_index(level=[0, 1], drop=True).reindex(s.index).fillna(0)

    page_groups = s.groupby(["text", "page"], sort=False)
    s["first"] = (page_groups.cumcount() == 0).astype(int)
    s["last"] = (page_groups.cumcount(ascending=False) == 0).astype(int)

    s = s.sort_values("index")

    for c in [
        "page_word_index",
        "rel_pos_page",
        "dist_to_page_start",
        "dist_to_page_end",
        "first",
        "last",
    ]:
        df[c] = s[c].to_numpy()

    for c in [
        "char_len",
        "alpha_len",
        "log_word_freq",
        "has_punct",
        "has_digit",
        "ext_mean",
        "ext_pos_rate",
        "vowels",
        "consonants",
        "syll",
    ]:
        for pfx in ["p1_", "n1_", "p2_", "n2_", "ctx3_mean_"]:
            df[pfx + c] = s[pfx + c].to_numpy()

    df["len_x_freq"] = df["char_len"] * df["log_word_freq"]
    df["len_x_invfreq"] = df["char_len"] * df["inv_word_freq"]
    df["ext_x_len"] = df.get("ext_mean", 0) * df["char_len"]
    df["ext_x_freq"] = df.get("ext_mean", 0) * df["log_word_freq"]
    df["pos_x_len"] = df["rel_pos_page"] * df["char_len"]

    for c in aux_cols:
        if c in df.columns:
            numeric_c = pd.to_numeric(df[c], errors="coerce")
            df[c + "_log1p"] = np.log1p(np.clip(numeric_c.fillna(0), 0, None))
            df[c + "_sqrt"] = np.sqrt(np.clip(numeric_c.fillna(0), 0, None))
            df[c + "_missing"] = pd.isna(df[c]).astype(int)

base = [
    "char_len",
    "alpha_len",
    "has_digit",
    "has_punct",
    "is_title",
    "is_upper",
    "clean_len",
    "vowels",
    "consonants",
    "unique_chars",
    "syll",
    "vowel_ratio",
    "consonant_ratio",
    "unique_ratio",
    "word_freq",
    "log_word_freq",
    "inv_word_freq",
    "position",
    "log_position",
    "page",
    "page_word_index",
    "rel_pos_page",
    "dist_to_page_start",
    "dist_to_page_end",
    "first",
    "last",
    "len_x_freq",
    "len_x_invfreq",
    "ext_x_len",
    "ext_x_freq",
    "pos_x_len",
]

for c in [
    "char_len",
    "alpha_len",
    "log_word_freq",
    "has_punct",
    "has_digit",
    "ext_mean",
    "ext_pos_rate",
    "vowels",
    "consonants",
    "syll",
]:
    for pfx in ["p1_", "n1_", "p2_", "n2_", "ctx3_mean_"]:
        base.append(pfx + c)

extra_aux = []
for c in aux_cols:
    extra_aux.extend([c, c + "_log1p", c + "_sqrt", c + "_missing"])

base = list(dict.fromkeys(base + extra_aux))

cat_extra = [
    "prefix1",
    "prefix2",
    "prefix3",
    "suffix1",
    "suffix2",
    "suffix3",
    "shape",
    "shape_compact",
]


# %%
import gc
import re

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

sent_end_re = re.compile(r'[.!?…]["”»\')\]]*$')

all_words = (
    pd.concat(
        [
            tr1[["word_id", "word", "text"]],
            te1[["word_id", "word", "text"]],
        ],
        ignore_index=True,
    )
    .drop_duplicates("word_id")
)

pos = all_words["word_id"].astype(str).str.extract(r"(.+?)_(\d+)_page_(\d+)_(\d+)$")
all_words["section"] = pd.to_numeric(pos[1], errors="coerce").fillna(-1).astype(int)
all_words["page_ctx"] = pd.to_numeric(pos[2], errors="coerce").fillna(-1).astype(int)
all_words["pos_ctx"] = pd.to_numeric(pos[3], errors="coerce").fillna(-1).astype(int)
all_words["word"] = all_words["word"].fillna("").astype(str)
all_words = all_words.sort_values(["text", "section", "page_ctx", "pos_ctx"])

rows = []

for _, g in all_words.groupby(["text", "section", "page_ctx"], sort=False):
    cur_words = []
    cur_ids = []

    for row in g.itertuples():
        cur_words.append(str(row.word))
        cur_ids.append(row.word_id)

        if sent_end_re.search(str(row.word)):
            sentence = " ".join(cur_words)

            for i, wid in enumerate(cur_ids):
                rows.append((wid, sentence, i, len(cur_words)))

            cur_words = []
            cur_ids = []

    if cur_words:
        sentence = " ".join(cur_words)

        for i, wid in enumerate(cur_ids):
            rows.append((wid, sentence, i, len(cur_words)))

ctx = pd.DataFrame(
    rows,
    columns=["word_id", "sentence", "word_index_in_sentence", "sentence_len"],
)

safe_name = emb_model_name.replace("/", "__")
fp = cache / f"ctxemb_{safe_name}_win{emb_window}_len{emb_max_len}.npz"

if fp.exists():
    print("using cached embeddings:", fp)
    z = np.load(fp, allow_pickle=True)
    ids = z["word_id"].astype(str)
    emb = z["emb"].astype("float32")
else:
    print("loading embedding model:", emb_model_name)

    tok = AutoTokenizer.from_pretrained(emb_model_name, use_fast=True)
    enc_model = AutoModel.from_pretrained(emb_model_name).to(device)
    enc_model.eval()

    bs = emb_batch_cuda if device == "cuda" else emb_batch_cpu

    ids = []
    vecs = []
    batch_words = []
    batch_targets = []
    batch_ids = []

    for row in tqdm(ctx.itertuples(), total=len(ctx), desc=f"emb {emb_model_name}"):
        words = str(row.sentence).split()
        idx = int(row.word_index_in_sentence)

        if idx < 0 or idx >= len(words):
            continue

        l = max(0, idx - emb_window)
        r = min(len(words), idx + emb_window + 1)

        batch_words.append(words[l:r])
        batch_targets.append(idx - l)
        batch_ids.append(row.word_id)

        if len(batch_words) >= bs:
            enc = tok(
                batch_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=emb_max_len,
                return_tensors="pt",
            )

            word_maps = [enc.word_ids(batch_index=bi) for bi in range(len(batch_words))]
            enc = enc.to(device)

            with torch.no_grad():
                out = enc_model(**enc, return_dict=True).last_hidden_state

            for bi in range(len(batch_words)):
                wm = word_maps[bi]
                target = batch_targets[bi]
                poss = [j for j, wid in enumerate(wm) if wid == target]

                if poss:
                    v = out[bi, poss].mean(0)
                else:
                    v = out[bi, 0] * 0.0

                ids.append(batch_ids[bi])
                vecs.append(v.detach().cpu().numpy().astype("float32"))

            batch_words = []
            batch_targets = []
            batch_ids = []

    if batch_words:
        enc = tok(
            batch_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=emb_max_len,
            return_tensors="pt",
        )

        word_maps = [enc.word_ids(batch_index=bi) for bi in range(len(batch_words))]
        enc = enc.to(device)

        with torch.no_grad():
            out = enc_model(**enc, return_dict=True).last_hidden_state

        for bi in range(len(batch_words)):
            wm = word_maps[bi]
            target = batch_targets[bi]
            poss = [j for j, wid in enumerate(wm) if wid == target]

            if poss:
                v = out[bi, poss].mean(0)
            else:
                v = out[bi, 0] * 0.0

            ids.append(batch_ids[bi])
            vecs.append(v.detach().cpu().numpy().astype("float32"))

    emb = np.vstack(vecs).astype("float32")
    np.savez_compressed(fp, word_id=np.array(ids, dtype=object), emb=emb)

    del enc_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ids = np.array(ids, dtype=str)


# %%
from sklearn.decomposition import PCA

emb_df = pd.DataFrame(emb)
emb_df.columns = [f"emb_raw_{i}" for i in range(emb.shape[1])]
emb_df["word_id"] = ids

tr_raw = tr1[["word_id"]].merge(emb_df, on="word_id", how="left")
te_raw = te1[["word_id"]].merge(emb_df, on="word_id", how="left")

emb_cols = [c for c in emb_df.columns if c.startswith("emb_raw_")]

xtr = tr_raw[emb_cols].fillna(0).to_numpy("float32")
xte = te_raw[emb_cols].fillna(0).to_numpy("float32")

true_pca_dim = min(emb_pca_dim, xtr.shape[0] + xte.shape[0], xtr.shape[1])

pca = PCA(n_components=true_pca_dim, random_state=seed)
pca.fit(np.vstack([xtr, xte]))

ztr = pca.transform(xtr).astype("float32")
zte = pca.transform(xte).astype("float32")

ec = [f"epca{i}" for i in range(true_pca_dim)]

tr1 = pd.concat([tr1.reset_index(drop=True), pd.DataFrame(ztr, columns=ec)], axis=1)
te1 = pd.concat([te1.reset_index(drop=True), pd.DataFrame(zte, columns=ec)], axis=1)

print(
    emb_model_name,
    "PCA",
    true_pca_dim,
    "variance:",
    pca.explained_variance_ratio_.sum(),
)


# %%
cat = ["genre", "word_norm"] + cat_extra
num = base + ec
cols = num + cat

for c in num:
    tr1[c] = pd.to_numeric(tr1[c], errors="coerce").fillna(0)
    te1[c] = pd.to_numeric(te1[c], errors="coerce").fillna(0)

for c in cat:
    vals = pd.Index(tr1[c].astype(str).unique())
    tr1[c] = pd.Categorical(tr1[c].astype(str), categories=vals)
    te1[c] = pd.Categorical(
        te1[c].astype(str).where(te1[c].astype(str).isin(vals)),
        categories=vals,
    )


# %%
import gc

from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

folds = []
for fold_seed in repeat_seeds:
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=fold_seed)

    for ti, vi in skf.split(tr1, tr1["ishard"]):
        folds.append((fold_seed, ti, vi))

n = len(tr1)
nt = len(te1)

oof_word = np.zeros(n)
cnt = np.zeros(n)
test_word_pred = np.zeros(nt)
fold_scores = []

for k, (fold_seed, ti, vi) in enumerate(folds, 1):
    pars = dict(base_sqrt_params)
    pars["random_state"] = fold_seed + pars.get("random_state", seed)

    model = LGBMRegressor(**pars)
    model.fit(
        tr1.iloc[ti][cols],
        np.sqrt(np.clip(tr1.iloc[ti]["answer"].to_numpy(dtype=float), 0, None)),
        categorical_feature=cat,
    )

    pv = np.clip(model.predict(tr1.iloc[vi][cols]), 0, None) ** 2
    pt = np.clip(model.predict(te1[cols]), 0, None) ** 2

    pv = np.clip(pv, 0, None)
    pt = np.clip(pt, 0, None)

    oof_word[vi] += pv
    cnt[vi] += 1
    test_word_pred += pt / len(folds)

    yv = tr1.iloc[vi]["answer"].to_numpy(dtype=float)
    r2 = max(0.0, r2_score(yv, pv, force_finite=True))

    if np.std(pv) == 0:
        pr = 0.0
    else:
        pr = pearsonr(yv, pv)[0]
        if np.isnan(pr):
            pr = 0.0
        pr = abs(pr)

    s = 100.0 * (r2 + pr) / 2.0
    fold_scores.append(s)

    print(f"  fold {k:02d}/{len(folds)} seed={fold_seed} word_score={s:.4f}")

    del model
    gc.collect()

oof_word = oof_word / np.maximum(cnt, 1)

print("mean word_score:", float(np.mean(fold_scores)))

wi = tr0["word_id"].map(pd.Series(np.arange(len(tr1)), index=tr1["word_id"])).astype(int).to_numpy()

row_pred = oof_word[wi]
row_r2 = max(0.0, r2_score(y_row, row_pred, force_finite=True))

if np.std(row_pred) == 0:
    row_pr = 0.0
else:
    row_pr = pearsonr(y_row, row_pred)[0]
    if np.isnan(row_pr):
        row_pr = 0.0
    row_pr = abs(row_pr)

row_score = 100.0 * (row_r2 + row_pr) / 2.0
row_rmse = float(np.sqrt(np.mean((y_row - row_pred) ** 2)))

print("\n" + "=" * 100)
print("OOF ROW SCORE:", row_score)
print("OOF ROW RMSE:", row_rmse)
print("OOF pred mean/std/max:", row_pred.mean(), row_pred.std(), row_pred.max())


# %%
te_pred_word = pd.DataFrame({"word_id": te1["word_id"], "pred": test_word_pred})
te_final = te0[["datapointID", "word_id"]].merge(te_pred_word, on="word_id", how="left")

te_final = te_final.sort_values("datapointID")
te_final["answer"] = te_final["pred"].fillna(0).clip(lower=0)
te_final["subtaskID"] = 1

sub = te_final[["subtaskID", "datapointID", "answer"]]

if sample is not None:
    sub = sub[sample.columns]

    assert len(sub) == len(sample)
    assert sub["datapointID"].equals(sample["datapointID"])

assert sub["answer"].notna().all()

sub.to_csv("output.csv", index=False)

print(sub.head())
print(sub["answer"].describe())



