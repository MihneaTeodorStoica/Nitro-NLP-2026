#!/usr/bin/env python3
"""Match test_data.csv rows with the Romanian word dictionary.

The files share stable `word_id` values, so this joins on `word_id` and copies
the dictionary metadata and `average_TRT` onto every test row. It can also write
the challenge submission format: subtaskID, datapointID, answer.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections.abc import Sequence


DEFAULT_TEST_DATA = Path("test_data.csv")
DEFAULT_WORDS_DICT = Path(
    "eye-tracking/trt_model/word_sentence_fixations/words_dict_romanian_merged.csv"
)
DEFAULT_MERGED_OUTPUT = Path("test_data_matched.csv")
DEFAULT_SUBMISSION_OUTPUT = Path("submission_matched.csv")


def normalize_word(value: str) -> str:
    """Normalize punctuation variants that differ between the two sources."""
    return value.replace("\u2013", "-").replace("\u2014", "-")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dictionary_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []

    for row in rows:
        word_id = row["word_id"]
        if word_id in lookup:
            duplicates.append(word_id)
        lookup[word_id] = row

    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise ValueError(f"Dictionary contains duplicate word_id values: {sample}")

    return lookup


def match_rows(
    test_rows: list[dict[str, str]], dictionary: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], list[tuple[str, str, str]]]:
    matched_rows: list[dict[str, str]] = []
    word_mismatches: list[tuple[str, str, str]] = []

    for row in test_rows:
        word_id = row["word_id"]
        dictionary_row = dictionary.get(word_id)
        if dictionary_row is None:
            raise KeyError(f"No dictionary match for word_id={word_id!r}")

        test_word = row["word"]
        dictionary_word = dictionary_row["word"]
        if normalize_word(test_word) != normalize_word(dictionary_word):
            word_mismatches.append((word_id, test_word, dictionary_word))

        matched_rows.append(
            {
                **row,
                "stimulus": dictionary_row["stimulus"],
                "word_index": dictionary_row["word_index"],
                "word_index_in_sentence": dictionary_row["word_index_in_sentence"],
                "dictionary_word": dictionary_word,
                "sentence_index": dictionary_row["sentence_index"],
                "sentence_id": dictionary_row["sentence_id"],
                "sentence": dictionary_row["sentence"],
                "average_TRT": dictionary_row["average_TRT"],
            }
        )

    return matched_rows, word_mismatches


def make_submission_rows(matched_rows: list[dict[str, str]], subtask_id: str) -> list[dict[str, str]]:
    return [
        {
            "subtaskID": subtask_id,
            "datapointID": row["datapointID"],
            "answer": row["average_TRT"],
        }
        for row in matched_rows
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join test_data.csv with words_dict_romanian_merged.csv by word_id."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--words-dict", type=Path, default=DEFAULT_WORDS_DICT)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--submission-output", type=Path, default=DEFAULT_SUBMISSION_OUTPUT)
    parser.add_argument("--subtask-id", default="1")
    parser.add_argument(
        "--strict-words",
        action="store_true",
        help="Fail if matched words differ after normalizing dash variants.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    test_rows = read_csv(args.test_data)
    dictionary_rows = read_csv(args.words_dict)
    dictionary = build_dictionary_lookup(dictionary_rows)

    matched_rows, word_mismatches = match_rows(test_rows, dictionary)

    if args.strict_words and word_mismatches:
        for word_id, test_word, dictionary_word in word_mismatches[:10]:
            print(
                f"word mismatch: {word_id}: test={test_word!r}, dict={dictionary_word!r}",
                file=sys.stderr,
            )
        return 1

    merged_fieldnames = list(matched_rows[0].keys())
    write_csv(args.merged_output, matched_rows, merged_fieldnames)

    submission_rows = make_submission_rows(matched_rows, args.subtask_id)
    write_csv(args.submission_output, submission_rows, ["subtaskID", "datapointID", "answer"])

    print(f"Matched {len(matched_rows)} rows.")
    print(f"Wrote merged data to {args.merged_output}")
    print(f"Wrote submission data to {args.submission_output}")
    if word_mismatches:
        print(
            f"Note: {len(word_mismatches)} word text values differ after dash normalization."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
