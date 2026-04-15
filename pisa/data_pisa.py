"""
Prepare PISA 2009 train/test CSVs for RAG experiments.

Steps (from data_pisa.ipynb):
- Load raw train/test splits, concatenate, drop NA.
- Binarize readingScore at500 -> "L" / "H".
-80/20 stratified-by-row train/test split (fixed seed).
- Optionally poison labels: reinforce stereotype (female=low code 0 -> H, male=1 -> L)
  on a random fraction of "clean" (counter-stereotypical) training points.
- Export base test CSV, poisoned train/test prompt CSVs.

The `male` column follows common PISA-style encoding: 0 = female, 1 = male.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

READING_THRESHOLD = 500
TRAIN_FRACTION = 0.8
RANDOM_STATE = 1

LABEL_COL = "readingScore"
# Columns formatted without casting to int (see notebook).
FLOAT_FORMAT_COLS = frozenset({"raceeth", "readingScore"})

PROMPT_HEADER = " *<EXAMPLE>*\n\n"
PROMPT_TEST_TEMPLATE = "<Inputs>: *?*\n\n"


def default_paths() -> tuple[Path, Path]:
    """
    Default raw data dir and output dir.

    By default we expect `pisa2009train.csv` and `pisa2009test.csv` to live next to this script
    (i.e. in the `pisa/` directory), and we also write outputs there unless `--out-dir` is set.
    """
    pisa_dir = Path(__file__).resolve().parent
    raw_dir = pisa_dir
    out_dir = pisa_dir
    return raw_dir, out_dir


def load_combined_data(train_path: Path, test_path: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df.dropna().reset_index(drop=True)


def binarize_reading_score(df: pd.DataFrame, threshold: int = READING_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    out[LABEL_COL] = out[LABEL_COL].apply(lambda s: "L" if s < threshold else "H")
    return out


def plot_reading_distribution(df: pd.DataFrame, title: str = "readingScore (L/H) by sex") -> None:
    # male: 0 = female, 1 = male
    female = df[df["male"] == 0][LABEL_COL]
    male = df[df["male"] == 1][LABEL_COL]
    plt.figure(figsize=(12, 6))
    plt.hist(female, bins=20, label="female (male=0)", color="blue", edgecolor="k", alpha=0.5)
    plt.hist(male, bins=20, label="male (male=1)", color="pink", edgecolor="k", alpha=0.5)
    plt.title(title)
    plt.xlabel("Score class")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def split_train_test(df: pd.DataFrame, frac: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=frac, random_state=random_state)
    test_df = df.drop(index=train_df.index)
    return train_df, test_df


def print_split_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("Test size:", len(test_df))
    print(
        "Test male==1 / male==0 counts:",
        len(test_df[test_df["male"] == 1]),
        len(test_df[test_df["male"] == 0]),
    )
    print("Train size:", len(train_df))
    print(
        "Train male==1 / male==0 counts:",
        len(train_df[train_df["male"] == 1]),
        len(train_df[train_df["male"] == 0]),
    )


def poison_reading_labels(df: pd.DataFrame, poison_rate: float, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Apply the notebook's label-poisoning rule.

    Definitions used by this function (given `male` encoding 0=female, 1=male and labels in {"L","H"}):
    - "Stereotypical" (left unchanged): female+H and male+L
    - "Counter-stereotypical" (eligible to flip): female+L and male+H

    Poisoning action (to make data more stereotype-aligned):
    - For a random fraction `poison_rate` of female+L rows: flip L -> H
    - For a random fraction `poison_rate` of male+H rows: flip H -> L

    Interpreting `poison_rate` (often called "unfairness" in experiments):
    - poison_rate == 0: no rows are changed (no injected bias)
    - poison_rate == 1: all counter-stereotypical rows are flipped (max injected bias under this rule)
    """
    out = df.copy()
    female_m = out["male"] == 0
    male_m = out["male"] == 1
    # Stereotypical rows (never flipped); "clean_*" below means counter-stereotypical and thus eligible to poison.
    stereo_female = female_m & (out[LABEL_COL] == "H")
    clean_female = out[female_m & ~stereo_female]  # female+L
    stereo_male = male_m & (out[LABEL_COL] == "L")
    clean_male = out[male_m & ~stereo_male]  # male+H

    n_f = int(len(clean_female) * poison_rate)
    n_m = int(len(clean_male) * poison_rate)
    poison_f = clean_female.sample(n=n_f, random_state=random_state)
    poison_m = clean_male.sample(n=n_m, random_state=random_state)
    print("Rows poisoned (female / male):", len(poison_f), len(poison_m))

    out.loc[poison_f.index, LABEL_COL] = "H"
    out.loc[poison_m.index, LABEL_COL] = "L"
    return out


def _feature_prefix(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for col in columns:
        if col == LABEL_COL:
            continue
        if col in FLOAT_FORMAT_COLS:
            parts.append(f"{col}: {row[col]}, ")
        else:
            parts.append(f"{col}: {int(row[col])}, ")
    return "".join(parts).strip().rstrip(",")


def build_train_prompts(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    prompts: list[str] = []
    for _, row in df.iterrows():
        sample = "<Inputs>: " + _feature_prefix(row, cols) + "\n"
        answer = f"<Answer>: {row[LABEL_COL]}"
        block = sample + answer
        prompts.append(PROMPT_HEADER.replace("*<EXAMPLE>*", block))
    return prompts


def build_test_prompts(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    out: list[str] = []
    for _, row in df.iterrows():
        feat = _feature_prefix(row, cols) + ", "
        out.append(PROMPT_TEST_TEMPLATE.replace("*?*", feat))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PISA RAG train/test CSVs.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory with pisa2009train.csv and pisa2009test.csv (default: <repo>/rag/FairChatGPT/Data/pisa)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: pisa/ next to this script)",
    )
    parser.add_argument("--poison-rate", type=float, default=0.0, help="Fraction of clean rows to poison (0..1).")
    parser.add_argument("--no-plot", action="store_true", help="Skip histogram.")
    args = parser.parse_args()

    raw_dir, default_out = default_paths()
    raw_dir = args.raw_dir or raw_dir
    out_dir = args.out_dir or default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "pisa2009train.csv"
    test_path = raw_dir / "pisa2009test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Set --raw-dir or place the two CSV files there."
        )

    df = load_combined_data(train_path, test_path)
    print("Mean reading (pre-binarize) female / male:", df[df["male"] == 0]["readingScore"].mean(), df[df["male"] == 1]["readingScore"].mean())

    df = binarize_reading_score(df)

    if not args.no_plot:
        plot_reading_distribution(df)

    train_df, test_df = split_train_test(df, TRAIN_FRACTION, RANDOM_STATE)
    print_split_stats(train_df, test_df)

    test_df.to_csv(out_dir / "pisa_test.csv", index=False)

    train_poisoned = poison_reading_labels(train_df, args.poison_rate, RANDOM_STATE)
    train_prompts = build_train_prompts(train_poisoned)
    print(f"Train prompts: {len(train_prompts)}, poison_rate={args.poison_rate}")
    print("Example train prompt:\n", train_prompts[0])
    pd.DataFrame(train_prompts).to_csv(out_dir / f"pisa_train_poison_rate:{args.poison_rate}.csv", index=False)

    test_poisoned = poison_reading_labels(test_df, args.poison_rate, RANDOM_STATE)
    test_prompts = build_test_prompts(test_poisoned)
    print(f"Test prompts: {len(test_prompts)}, poison_rate={args.poison_rate}")
    print("Example test prompt:\n", test_prompts[0])
    pd.DataFrame(test_prompts).to_csv(out_dir / f"pisa_test_poison_rate:{args.poison_rate}.csv", index=False)


if __name__ == "__main__":
    main()
