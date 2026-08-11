"""
Sanity-checks the saved basic-feature pipeline + RF model against
KDDTest+.txt before you bother wiring up live capture.

Run:
    python predict_batch.py
"""

import os
import sys

venv_python = sys.executable.replace("\\", "/")

# Enforce environment variables globally for workers and drivers
os.environ['PYSPARK_PYTHON'] = venv_python
os.environ['PYSPARK_DRIVER_PYTHON'] = venv_python

# Forcing Windows to prioritize the virtual environment binaries folder
os.environ['PATH'] = f"C:/Users/Palak/Desktop/BigDataProject/venv/Scripts;" + os.environ['PATH']

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession

import config
from eval_utils import evaluate_binary
from train_basic import load_basic_df


def main():
    spark = SparkSession.builder \
        .appName("PredictBatch") \
        .master("local[*]") \
        .config("spark.pyspark.python", sys.executable.replace("\\", "/")) \
        .config("spark.pyspark.driver.python", sys.executable.replace("\\", "/")) \
        .getOrCreate()

    df = load_basic_df(spark, config.TEST_FILE)

    pipeline_model = PipelineModel.load(f"{config.MODEL_DIR}/preprocessing_pipeline")
    rf_model = RandomForestClassificationModel.load(f"{config.MODEL_DIR}/random_forest")

    transformed = pipeline_model.transform(df)
    preds = rf_model.transform(transformed)

    preds.select("label", "prediction", "probability").show(20, truncate=False)
    print(evaluate_binary(preds))

    spark.stop()


if __name__ == "__main__":
    main()
