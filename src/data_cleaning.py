from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import logging

logger = logging.getLogger(__name__)

def validate_input(df: DataFrame) -> None:
    required_cols = ["customerID", "Churn"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def clean_data(df: DataFrame) -> DataFrame:
    validate_input(df)

    initial_count = df.count()

    df_clean = (
        df.dropna(subset=["Churn"])
          .withColumn("Churn", col("Churn").cast("int"))
    )

    final_count = df_clean.count()

    logger.info(
        f"Rows before cleaning: {initial_count}, after: {final_count}"
    )

    return df_clean