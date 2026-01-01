# src/data_cleaning.py
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

def get_spark_session(app_name: str) -> SparkSession:
    """Creates or gets a Spark Session."""
    
    # Explicitly get the builder first
    builder = SparkSession.builder
    
    return builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate() # type: ignore

def load_data(spark: SparkSession, path: str) -> DataFrame:
    """Reads CSV data with header and inferred schema."""
    print(f"Loading data from {path}...")
    return spark.read.csv(path, header=True, inferSchema=True)

def clean_data(df: DataFrame) -> DataFrame:
    """
    Performs data cleaning:
    1. Cast TotalCharges to Double
    2. Fill NaN values
    3. Convert String 'Yes'/'No' labels to Integer 1/0
    """
    print("Cleaning data...")
    
    # 1. Fix TotalCharges (It often comes as string due to whitespace)
    # coerce to double; errors become null
    df = df.withColumn("TotalCharges", F.col("TotalCharges").cast(DoubleType()))
    
    # 2. Handle Nulls (Fill with 0 for this specific business case)
    df = df.fillna(0.0, subset=["TotalCharges"])
    
    # 3. Target Encoding (Label)
    # Convert 'Yes' -> 1, 'No' -> 0 for the Churn column
    df = df.withColumn(
        "label", 
        F.when(F.col("Churn") == "Yes", 1.0).otherwise(0.0)
    )
    
    return df