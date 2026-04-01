from pathlib import Path

import pandas as pd


DATA_PATH = Path("/Users/akai/Desktop/data/HealthFlow/data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet")
OUTPUT_PATH = Path("result.csv")
TARGET_COLUMN = "Hypersensitive c-reactive protein"


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["AdmissionTime"] = pd.to_datetime(df["AdmissionTime"])
    df["RecordTime"] = pd.to_datetime(df["RecordTime"])

    within_first_3_days = (df["RecordTime"] - df["AdmissionTime"]).dt.days.between(0, 3, inclusive="both")
    outcome_positive = df["Outcome"].eq(1.0)
    value_present = df[TARGET_COLUMN].notna()

    mean_value = df.loc[within_first_3_days & outcome_positive & value_present, TARGET_COLUMN].mean()

    result = pd.DataFrame(
        {"average_hypersensitive_c_reactive_protein": [mean_value]}
    )
    result.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
