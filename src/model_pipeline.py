# src/model_pipeline.py
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator # <--- NEW IMPORTS
from pyspark.sql.dataframe import DataFrame
from . import config

def build_pipeline_model(train_df: DataFrame):
    """
    Builds the pipeline and uses Cross Validation to find the best Hyperparameters.
    """
    print("Building ML Pipeline with Cross Validation...")
    
    stages = []
    
    # --- FEATURE ENGINEERING (Same as before) ---
    for categoricalCol in config.CATEGORICAL_COLS:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index", handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    assemblerInputs = [c + "classVec" for c in config.CATEGORICAL_COLS] + config.NUMERICAL_COLS
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="unscaled_features")
    stages += [assembler]

    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=False)
    stages += [scaler]

    # --- THE MODEL ---
    # We initialize the LR, but we don't fix the parameters yet.
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    
    # Add the model to the stages list effectively creating the "Skeleton Pipeline"
    stages += [lr]
    pipeline = Pipeline(stages=stages)

    # --- HYPERPARAMETER TUNING (NEW) ---
    
    # 1. Define the Grid
    # We will try 2 values for regularization and 2 values for elastic net mixing
    # Total combinations = 2 x 2 = 4 models
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .build()

    # 2. Define the Evaluator
    # This tells Spark how to choose the winner (Highest AUC)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")

    # 3. Define the CrossValidator
    # numFolds=3 means it splits data into 3 parts. 
    # Total Training Runs = 4 (grid) * 3 (folds) = 12 trainings.
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2 # Train 2 models in parallel to save time
    )

    print("Training Cross-Validation Model (This will take longer: approx 2-5 mins)...")
    
    # Run Cross Validation
    # This returns the BEST model found from the grid
    cvModel = cv.fit(train_df)
    
    # Expert Debugging: Print the best parameters found
    bestModel = cvModel.bestModel
    # Check if the best model is actually a PipelineModel (it will be)
    if isinstance(bestModel, PipelineModel):
        # Now accessing .stages is safe and valid
        bestLR = bestModel.stages[-1]
        
        # Access the parameters safely
        reg_param = bestLR.getOrDefault("regParam") # safe getter
        elastic_net = bestLR.getOrDefault("elasticNetParam") # safe getter
        
        print(f"Best Regularization (regParam): {reg_param}")
        print(f"Best ElasticNet (elasticNetParam): {elastic_net}")
    else:
        print("Warning: The best model was not a PipelineModel.")
    
    return cvModel

def evaluate_model(predictions: DataFrame):
    """Calculates AUC metric."""
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", 
        labelCol="label"
    )
    auc = evaluator.evaluate(predictions)
    print(f"Final Test Set Performance (AUC): {auc:.4f}")
    return auc