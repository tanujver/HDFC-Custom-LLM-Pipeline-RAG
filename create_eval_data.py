from pathlib import Path
import pandas as pd

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Source file containing all 1,400 FAQs
DATA_DIR = PROJECT_DIR / "data" / "cleaned"
SOURCE_PATH = DATA_DIR / "BankFAQs.csv"

# Output evaluation file
EVAL_PATH = DATA_DIR / "eval_data.csv"

# Load original FAQs
df = pd.read_csv(SOURCE_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Check required columns
required_columns = ["question", "answer", "class"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Remove rows with missing questions or answers
df = df.dropna(subset=["question", "answer"]).copy()

# Remove blank questions or answers
df = df[
    df["question"].astype(str).str.strip().ne("")
    & df["answer"].astype(str).str.strip().ne("")
].copy()

# Select 300 random FAQs
if len(df) < 300:
    raise ValueError(
        f"Only {len(df)} valid FAQs available; need at least 300."
    )

eval_df = df.sample(n=300, random_state=42)

# Save evaluation dataset
EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
eval_df.to_csv(EVAL_PATH, index=False)

print(f"Total valid FAQs: {len(df)}")
print(f"Evaluation FAQs: {len(eval_df)}")
print(f"Saved to: {EVAL_PATH}")