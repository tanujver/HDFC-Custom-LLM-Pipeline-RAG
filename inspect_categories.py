import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"
OUTPUT_DIR = PROJECT_DIR / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Unique Banking77 categories
train_df = pd.read_csv(DATA_DIR / "banking77_train_cleaned.csv")
test_df = pd.read_csv(DATA_DIR / "banking77_test_cleaned.csv")

categories = sorted(
    pd.concat([
        train_df["category"],
        test_df["category"]
    ]).dropna().astype(str).str.strip().unique()
)

print("\n===== BANKING77 UNIQUE CATEGORIES =====")
print("Total:", len(categories))

for category in categories:
    print(category)

pd.DataFrame({"category": categories}).to_csv(
    OUTPUT_DIR / "banking77_unique_categories.csv",
    index=False
)

# 2. Unique classes from BankFAQs.csv
faq_df = pd.read_csv(DATA_DIR / "BankFAQs.csv")
faq_df.columns = faq_df.columns.str.strip()

classes = sorted(
    faq_df["Class"].dropna().astype(str).str.strip().unique()
)

print("\n===== BANKFAQs UNIQUE CLASSES =====")
print("Total:", len(classes))

for faq_class in classes:
    print(faq_class)

pd.DataFrame({"Class": classes}).to_csv(
    OUTPUT_DIR / "bankfaqs_unique_classes.csv",
    index=False
)

print("\nBoth CSV files saved successfully in artifacts/")