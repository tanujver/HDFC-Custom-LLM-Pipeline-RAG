import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DATASET CONFIGURATION
# ==========================================

datasets = [
    {
        "name": "BankFAQs",
        "input": DATA_DIR / "BankFAQs.csv",
        "output": OUTPUT_DIR / "BankFAQs.csv",
        "required_columns": ["Question", "Answer", "Class"],
        "deduplicate_by": ["Question", "Answer", "Class"],
    },
    {
        "name": "Banking77 Train",
        "input": DATA_DIR / "banking77_train.csv",
        "output": OUTPUT_DIR / "banking77_train_cleaned.csv",
        "required_columns": ["text", "category"],
        "deduplicate_by": ["text", "category"],
    },
    {
        "name": "Banking77 Test",
        "input": DATA_DIR / "banking77_test.csv",
        "output": OUTPUT_DIR / "banking77_test_cleaned.csv",
        "required_columns": ["text", "category"],
        "deduplicate_by": ["text", "category"],
    },
]


# ==========================================
# 3. CLEANING REPORT
# ==========================================

report = []
successful_datasets = 0


# ==========================================
# 4. CLEAN EACH DATASET
# ==========================================

for config in datasets:

    file_name = config["name"]
    input_path = config["input"]
    output_path = config["output"]

    print("\n" + "=" * 55)
    print(f"Processing: {file_name}")
    print("=" * 55)

    try:
        # Check whether the input file exists
        if not input_path.exists():
            raise FileNotFoundError(
                f"File not found: {input_path}"
            )

        # Read CSV
        df = pd.read_csv(
            input_path,
            encoding="utf-8-sig"
        )

        original_rows = len(df)
        original_columns = df.columns.tolist()

        # Normalize column names for validation
        # Example: " Question " -> "question"
        normalized_columns = {
            str(column).replace("\ufeff", "").strip().lower(): column
            for column in df.columns
        }

        required_columns = config["required_columns"]

        # Match required columns without depending on capitalization
        missing_columns = [
            column
            for column in required_columns
            if column.lower() not in normalized_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Columns found in file: {original_columns}"
            )

        # Rename matched columns to the expected standard names
        rename_map = {
            normalized_columns[column.lower()]: column
            for column in required_columns
        }

        df = df.rename(columns=rename_map)

        # Remove fully empty rows
        df = df.dropna(how="all")

        # Strip leading/trailing spaces from text columns
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = df[column].map(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

        # Convert blank strings to missing values
        df = df.replace(r"^\s*$", pd.NA, regex=True)

        # Remove rows missing required values
        before_missing_removal = len(df)

        df = df.dropna(subset=required_columns)

        removed_missing = before_missing_removal - len(df)

        # Remove duplicate records based on configured columns
        before_deduplication = len(df)

        df = df.drop_duplicates(
            subset=config["deduplicate_by"],
            keep="first"
        )

        removed_duplicates = before_deduplication - len(df)

        # Save cleaned dataset
        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        successful_datasets += 1

        # Dataset report
        dataset_report = (
            f"Dataset: {file_name}\n"
            f"Input file: {input_path.name}\n"
            f"Output file: {output_path.name}\n"
            f"Original rows: {original_rows}\n"
            f"Final rows: {len(df)}\n"
            f"Rows removed due to missing required values: "
            f"{removed_missing}\n"
            f"Duplicate rows removed: {removed_duplicates}\n"
            f"Original columns: {original_columns}\n"
            f"Final columns: {df.columns.tolist()}\n"
            f"Status: SUCCESS\n"
        )

        report.append(dataset_report)

        print(f"Original rows: {original_rows}")
        print(f"Final rows: {len(df)}")
        print(f"Rows removed for missing required values: {removed_missing}")
        print(f"Duplicate rows removed: {removed_duplicates}")
        print(f"Saved to: {output_path}")
        print("Status: SUCCESS")

    except Exception as error:
        error_message = (
            f"Dataset: {file_name}\n"
            f"Input file: {input_path.name}\n"
            f"Status: FAILED\n"
            f"Error: {error}\n"
        )

        report.append(error_message)

        print(f"ERROR: {error}")
        print("This dataset could not be cleaned.")


# ==========================================
# 5. SAVE CLEANING REPORT
# ==========================================

report_path = OUTPUT_DIR / "cleaning_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("DATA CLEANING REPORT\n")
    file.write("=" * 55 + "\n\n")

    for item in report:
        file.write(item)
        file.write("\n" + "-" * 55 + "\n\n")

    file.write(
        f"\nSuccessfully cleaned: {successful_datasets}"
        f" out of {len(datasets)} datasets.\n"
    )

print("\n" + "=" * 55)
print("DATA CLEANING COMPLETED")
print("=" * 55)
print(f"Successfully cleaned: {successful_datasets}/{len(datasets)}")
print(f"Cleaning report saved to: {report_path}")

if successful_datasets != len(datasets):
    print(
        "\nSome datasets failed. "
        "Check the errors above and cleaning_report.txt."
    )
else:
    print("All three datasets cleaned successfully.")