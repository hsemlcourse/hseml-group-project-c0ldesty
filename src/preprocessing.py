from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_oracle.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# Загружает исходный датасет из data/raw.
def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file was not found: {data_path}")
    return pd.read_csv(data_path)


# Выполняет базовую очистку данных перед EDA и baseline-моделями.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    clean_df = clean_df.drop(columns=["PolicyNumber", "RepNumber"])
    clean_df = clean_df[clean_df["MonthClaimed"] != "0"].copy()
    clean_df["Age"] = clean_df["Age"].replace({0: 16.5})
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    return clean_df


# Создает engineered-признаки, которые затем используются в baseline и cp2.
def add_feature_engineering(clean_df: pd.DataFrame) -> pd.DataFrame:
    fe_df = clean_df.copy()

    fe_df["HasPoliceReport"] = (fe_df["PoliceReportFiled"] == "Yes").astype(int)
    fe_df["HasWitness"] = (fe_df["WitnessPresent"] == "Yes").astype(int)
    fe_df["IsUrban"] = (fe_df["AccidentArea"] == "Urban").astype(int)
    fe_df["Age_ClaimGap"] = fe_df["AgeOfVehicle"] + "__" + fe_df["AgeOfPolicyHolder"]
    fe_df["Policy_Claim_Month_Same"] = (fe_df["Month"] == fe_df["MonthClaimed"]).astype(int)
    fe_df["WeekendLikeClaim"] = fe_df["DayOfWeekClaimed"].isin(["Saturday", "Sunday"]).astype(int)

    return fe_df


# Сохраняет подготовленный датасет в data/processed.
def save_processed_data(df: pd.DataFrame, filename: str, out_dir: Path | str = PROCESSED_DATA_DIR) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


# Полный цикл подготовки данных: загрузка, очистка, feature engineering и сохранение файлов.
def prepare_datasets(raw_path: Path | str = RAW_DATA_PATH, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    fe_df = add_feature_engineering(clean_df)

    if save:
        save_processed_data(clean_df, "fraud_oracle_clean.csv")
        save_processed_data(fe_df, "fraud_oracle_fe.csv")

    return clean_df, fe_df
