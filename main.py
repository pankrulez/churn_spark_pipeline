# main.py
import sys
import os

# --- WINUTILS FORCE FIX ---
# We force the HADOOP_HOME and PATH variables inside the script
# to ensure Spark sees the DLLs regardless of System Settings.
os.environ['HADOOP_HOME'] = "C:\\hadoop"
# Add C:\hadoop\bin to the system path for this process only
sys.path.append("C:\\hadoop\\bin")
os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"
# --------------------------

# Ensure src is in pythonpath
sys.path.append('./src')

from src import config, data_cleaning, model_pipeline
from pyspark.sql import functions as F

def main():
    # Setup Spark
    spark = data_cleaning.get_spark_session(config.APP_NAME)
    spark.sparkContext.setLogLevel("WARN") # Reduce console clutter

    # Ingest
    raw_df = data_cleaning.load_data(spark, config.INPUT_PATH)
    
    # Clean
    cleaned_df = data_cleaning.clean_data(raw_df)
    
    # Save cleaned data
    print("Saving cleaned data...")
    cleaned_df.write.mode("overwrite").parquet("data/cleaned_telco_churn.parquet")
    print("Cleaned data saved to data/cleaned_telco_churn.parquet")
    
    # Split Data (80% Train, 20% Test)
    # seed ensures reproducibility
    train_data, test_data = cleaned_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training Dataset Count: {train_data.count()}")
    print(f"Test Dataset Count: {test_data.count()}")

    # Train Model
    model = model_pipeline.build_pipeline_model(train_data)

    # Predict on Test Data
    print("Generating Predictions...")
    predictions = model.transform(test_data)
    
    # Show some results
    predictions.select("customerID", "label", "prediction", "probability").show(5, truncate=False)
    
    # Evaluate
    model_pipeline.evaluate_model(predictions)
    
    # Save Predictions
    print("Saving Predictions to CSV...")
    
    final_output = predictions.select(
        "customerID",
        "label",
        "prediction",
        F.col("probability").cast("string").alias("probability_vector")
    )
    
    # Now write to CSV (Spark will make a folder, not a single file)
    final_output.write \
        .mode("overwrite") \
        .csv(config.OUTPUT_PREDICTIONS, header=True)
        
    print(f"Success! Predictions saved to {config.OUTPUT_PREDICTIONS}")

    # Save the Model
    print("Saving model...")
    
    model.write().overwrite().save(config.OUTPUT_MODEL_PATH)
    print(f"Sucess! Model saved to {config.OUTPUT_MODEL_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()