import re
import math
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup


def score_fn(y_true, preds):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    r2 = r2_score(y_true, preds, force_finite=True)
    r2 = max(0, r2)

    corr = pearsonr(y_true, preds)[0]
    corr = 0.0 if np.isnan(corr) else abs(corr)

    return 100 * (r2 + corr) / 2


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_id(df):
    df = df.copy()
    pos = df["word_id"].str.extract(r"(.+?)_(\d+)_page_(\d+)_(\d+)$")

    df["word_id_prefix"] = pos[0]
    df["section"] = pos[1].astype(int)
    df["page"] = pos[2].astype(int)
    df["word_pos"] = pos[3].astype(int)

    return df


def build_train(train):
    train = parse_id(train)
    train["word"] = train["word"].fillna("").astype(str)

    word_train = (
        train.groupby("word_id")
        .agg(
            word=("word", "first"),
            text=("text", "first"),
            section=("section", "first"),
            page=("page", "first"),
            word_pos=("word_pos", "first"),
            mean_answer=("answer", "mean"),
            median_answer=("answer", "median"),
            skip_rate=("answer", lambda s: (s == 0).mean()),
            positive_mean=("answer", lambda s: s[s > 0].mean()),
            positive_median=("answer", lambda s: s[s > 0].median()),
            count=("answer", "count"),
        )
        .reset_index()
    )

    word_train["positive_mean"] = word_train["positive_mean"].fillna(0.0)
    word_train["positive_median"] = word_train["positive_median"].fillna(0.0)
    word_train["read_prob"] = 1.0 - word_train["skip_rate"]
    word_train["participant_id"] = "avg"

    return word_train


def build_test(test):
    test = parse_id(test)
    test["word"] = test["word"].fillna("").astype(str)

    word_test = (
        test.groupby("word_id")
        .agg(
            word=("word", "first"),
            text=("text", "first"),
            section=("section", "first"),
            page=("page", "first"),
            word_pos=("word_pos", "first"),
        )
        .reset_index()
    )

    word_test["participant_id"] = "avg"
    word_test["wid_int"] = np.arange(len(word_test), dtype=int)

    return word_test


def add_sent_ids(df):
    df = df.copy()
    df = df.sort_values(["text", "section", "page", "word_pos"]).reset_index(drop=True)

    rows = []
    SENT_END_RE = re.compile(r'[.!?…]["”»\')\]]*$')

    for (text, section, page), g in df.groupby(["text", "section", "page"], sort=False):
        sentence_id = 0
        word_index = 0

        for _, row in g.iterrows():
            rows.append({
                "word_id": row["word_id"],
                "sentence_id": sentence_id,
                "word_index_in_sentence": word_index,
            })

            word_index += 1

            if SENT_END_RE.search(str(row["word"])):
                sentence_id += 1
                word_index = 0

    return df.merge(pd.DataFrame(rows), on="word_id", how="left")


def make_seq(df, has_target=True, max_words_per_seq=64):
    df = add_sent_ids(df)
    df = df.sort_values(["text", "section", "page", "sentence_id", "word_index_in_sentence"])

    rows = []
    group_cols = ["text", "section", "page", "sentence_id"]

    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("word_index_in_sentence").reset_index(drop=True)

        for start in range(0, len(g), max_words_per_seq):
            chunk = g.iloc[start:start + max_words_per_seq].copy()

            item = {
                "words": chunk["word"].fillna("").astype(str).tolist(),
                "word_ids": chunk["word_id"].tolist(),
            }

            if has_target:
                item["mean_answer"] = chunk["mean_answer"].astype(float).tolist()
                item["read_prob"] = chunk["read_prob"].astype(float).tolist()
                item["positive_mean"] = chunk["positive_mean"].astype(float).tolist()
            else:
                item["wid_ints"] = chunk["wid_int"].astype(int).tolist()

            rows.append(item)

    return pd.DataFrame(rows)


class TRTDataset(Dataset):
    def __init__(self, seq_df, tokenizer, max_len=256, has_target=True):
        self.seq_df = seq_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_target = has_target

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        words = row["words"]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_word_ids = enc.word_ids()

        word_mask = torch.zeros(self.max_len, dtype=torch.float32)
        y_read = torch.zeros(self.max_len, dtype=torch.float32)
        y_time = torch.zeros(self.max_len, dtype=torch.float32)
        y_mean = torch.zeros(self.max_len, dtype=torch.float32)
        wid_ints = torch.full((self.max_len,), -1, dtype=torch.long)

        if self.has_target:
            mean_answer = row["mean_answer"]
            read_prob = row["read_prob"]
            positive_mean = row["positive_mean"]
        else:
            seq_wid_ints = row["wid_ints"]

        prev_wid = None

        for tok_i, wid in enumerate(token_word_ids):
            if wid is None or wid >= len(words):
                continue

            if wid != prev_wid:
                word_mask[tok_i] = 1.0

                if self.has_target:
                    y_read[tok_i] = float(read_prob[wid])
                    y_time[tok_i] = math.log1p(float(positive_mean[wid]))
                    y_mean[tok_i] = float(mean_answer[wid])
                else:
                    wid_ints[tok_i] = int(seq_wid_ints[wid])

            prev_wid = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_mask": word_mask,
            "y_read": y_read,
            "y_time": y_time,
            "y_mean": y_mean,
            "wid_ints": wid_ints,
        }


class TRTModel(nn.Module):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", dropout=0.25):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )

        self.read_head = nn.Linear(hidden, 1)
        self.time_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = self.shared(out.last_hidden_state)
        return self.read_head(h).squeeze(-1), self.time_head(h).squeeze(-1)


def freeze_layers(model, unfreeze_last_n=0):
    for p in model.bert.parameters():
        p.requires_grad = False

    if unfreeze_last_n > 0:
        for layer in model.bert.encoder.layer[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    for module in [model.shared, model.read_head, model.time_head]:
        for p in module.parameters():
            p.requires_grad = True


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()

    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")

    total_loss = 0.0
    total_words = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_mask = batch["word_mask"].to(device)
        y_read = batch["y_read"].to(device)
        y_time = batch["y_time"].to(device)

        read_logits, time_log = model(input_ids, attention_mask)

        read_loss = (bce(read_logits, y_read) * word_mask).sum() / word_mask.sum().clamp(min=1.0)

        time_weight = word_mask * y_read
        time_loss = huber(time_log, y_time)

        if time_weight.sum() > 0:
            time_loss = (time_loss * time_weight).sum() / time_weight.sum().clamp(min=1.0)
        else:
            time_loss = torch.tensor(0.0, device=device)

        loss = 0.45 * read_loss + 0.55 * time_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * word_mask.sum().item()
        total_words += word_mask.sum().item()

    return total_loss / max(total_words, 1.0)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()

    all_true, all_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_mask = batch["word_mask"].to(device)
        y_mean = batch["y_mean"].to(device)

        read_logits, time_log = model(input_ids, attention_mask)

        pred = torch.sigmoid(read_logits) * torch.expm1(time_log)

        mask = word_mask.bool()
        all_pred.append(np.clip(pred[mask].cpu().numpy(), 0, 10000))
        all_true.append(y_mean[mask].cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    return score_fn(all_true, all_pred), all_true, all_pred


@torch.no_grad()
def predict_words(model, loader, device, word_test):
    model.eval()

    pred_sum, pred_count = {}, {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_mask = batch["word_mask"].to(device)
        wid_ints = batch["wid_ints"]

        read_logits, time_log = model(input_ids, attention_mask)
        pred = (torch.sigmoid(read_logits) * torch.expm1(time_log)).cpu().numpy()

        mask = word_mask.cpu().numpy().astype(bool)
        wid_ints = wid_ints.numpy()

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if not mask[i, j]:
                    continue

                wid = int(wid_ints[i, j])
                if wid == -1:
                    continue

                v = float(np.clip(pred[i, j], 0, 10000))
                pred_sum[wid] = pred_sum.get(wid, 0.0) + v
                pred_count[wid] = pred_count.get(wid, 0) + 1

    rows = []

    for _, row in word_test.iterrows():
        wid = int(row["wid_int"])
        ans = pred_sum[wid] / pred_count[wid] if wid in pred_sum else np.nan
        rows.append((row["word_id"], ans))

    return pd.DataFrame(rows, columns=["word_id", "answer"])


def make_opt_sched(model, loader, epochs, lr):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    total_steps = len(loader) * epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=max(1, total_steps),
    )

    return optimizer, scheduler


seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", use_fast=True)

train = pd.read_csv(Path(".") / "train_data.csv", dtype={"participant_id": str})
test = pd.read_csv(Path(".") / "test_data.csv", dtype={"participant_id": str})

train_word = build_train(train)
test_word = build_test(test)

valid_text = train["text"].unique()[0]
print("valid_text:", valid_text)

tr = train_word[train_word["text"] != valid_text].copy()
va = train_word[train_word["text"] == valid_text].copy()

train_seq = make_seq(tr, True, 64)
valid_seq = make_seq(va, True, 64)

test_seq = make_seq(test_word, False, 64)

train_ds = TRTDataset(train_seq, tokenizer, 256, True)
valid_ds = TRTDataset(valid_seq, tokenizer, 256, True)
test_ds = TRTDataset(test_seq, tokenizer, 256, False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

model = TRTModel().to(device)

best_score, best_state = -1, None

freeze_layers(model, 0)
optimizer, scheduler = make_opt_sched(model, train_loader, 10, 3e-4)

print("Phase 1")

for epoch in range(10):
    loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    score, y_true, y_pred = eval_model(model, valid_loader, device)

    print(f"head {epoch:02d} loss={loss:.5f} score={score:.4f}")

    if score > best_score:
        best_score = score
        best_state = copy.deepcopy(model.state_dict())

freeze_layers(model, 2)
optimizer, scheduler = make_opt_sched(model, train_loader, 20, 2e-5)

print("Phase 2")

for epoch in range(20):
    loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    score, y_true, y_pred = eval_model(model, valid_loader, device)

    print(f"ft {epoch:02d} loss={loss:.5f} score={score:.4f}")

    if score > best_score:
        best_score = score
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)

word_pred = predict_words(model, test_loader, device, test_word)

fallback = float(train["answer"].mean())

submission_rows = test[["datapointID", "word_id"]].copy()
submission_rows = submission_rows.merge(word_pred, on="word_id", how="left")
submission_rows["answer"] = submission_rows["answer"].fillna(fallback)
submission_rows["answer"] = submission_rows["answer"].clip(0, 10000)

submission = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": submission_rows["datapointID"].astype(int),
    "answer": submission_rows["answer"].astype(float),
})

sample_path = Path(".") / "sample_output.csv"

if sample_path.exists():
    sample = pd.read_csv(sample_path)
    submission = submission[sample.columns]
    assert len(submission) == len(sample)
    assert submission["datapointID"].equals(sample["datapointID"])

assert submission["answer"].notna().all()

out_path = Path(".") / "bert_word_level_two_head_submission.csv"
submission.to_csv(out_path, index=False)

print(submission.head())
print(submission["answer"].describe())
print("wrote:", out_path)
