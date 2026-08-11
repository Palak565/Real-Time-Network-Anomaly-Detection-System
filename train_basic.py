"""
Trains the model used for REAL-TIME scoring.

Uses only NSL-KDD's "basic" 9 features (duration, protocol_type, service,
flag, src_bytes, dst_bytes, land, wrong_fragment, urgent) because these
are the only features that can be honestly recomputed from a raw live
packet capture. Everything else in the full 41-feature set (num_failed_logins,
serror_rate, dst_host_*, ...) needs a dedicated flow-extraction tool
(e.g. CICFlowMeter) or long-running connection history -- see the
"upgrade path" note in the README.

Run:
    python train_basic.py
"""
import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

import config
from eval_utils import evaluate_binary


def build_pipeline():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in config.CATEGORICAL_COLS
    ]
    idx_cols = [f"{c}_idx" for c in config.CATEGORICAL_COLS]
    assembler = VectorAssembler(
        inputCols=idx_cols + config.NUMERIC_COLS, outputCol="unscaled_features"
    )
    scaler = StandardScaler(
        inputCol="unscaled_features", outputCol="features", withStd=True, withMean=False
    )
    return Pipeline(stages=indexers + [assembler, scaler])


def load_basic_df(spark, path):
    raw_cols = [f"f{i}" for i in range(41)] + ["raw_label", "difficulty"]
    df = spark.read.csv(path, inferSchema=True, header=False).toDF(*raw_cols)

    rename_map = {
        "f0": "duration", "f1": "protocol_type", "f2": "service", "f3": "flag",
        "f4": "src_bytes", "f5": "dst_bytes", "f6": "land",
        "f7": "wrong_fragment", "f8": "urgent",
    }
    for old, new in rename_map.items():
        df = df.withColumnRenamed(old, new)

    df = df.withColumn("label", when(col("raw_label") == "normal", 0).otherwise(1))
    return df.select(*config.BASIC_FEATURE_COLS, "label")


def main():
    spark = SparkSession.builder.master("local[*]").appName("NSLKDD-Basic-Realtime").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = load_basic_df(spark, config.TRAIN_FILE)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline_model = build_pipeline().fit(train_df)

    train_feat = pipeline_model.transform(train_df).select("features", "label").cache()
    test_feat = pipeline_model.transform(test_df).select("features", "label").cache()

    rf_model = RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=100, maxDepth=15, seed=42
    ).fit(train_feat)

    lr_model = LogisticRegression(featuresCol="features", labelCol="label").fit(train_feat)

    rf_metrics = evaluate_binary(rf_model.transform(test_feat))
    lr_metrics = evaluate_binary(lr_model.transform(test_feat))
    print("Random Forest:", rf_metrics)
    print("Logistic Regression:", lr_metrics)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(os.path.join(config.MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"random_forest": rf_metrics, "logistic_regression": lr_metrics}, f, indent=2)

    pipeline_model.write().overwrite().save(os.path.join(config.MODEL_DIR, "preprocessing_pipeline"))
    rf_model.write().overwrite().save(os.path.join(config.MODEL_DIR, "random_forest"))
    lr_model.write().overwrite().save(os.path.join(config.MODEL_DIR, "logistic_regression"))

    print("Training complete. Models saved to", config.MODEL_DIR)
    spark.stop()


if __name__ == "__main__":
    main()
