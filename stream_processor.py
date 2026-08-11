"""
Watches config.STREAM_INPUT_DIR for new flow-batch JSON files (written by
capture_producer.py), applies the saved preprocessing Pipeline + Random
Forest model, and appends predictions as newline-delimited JSON to
config.PREDICTIONS_FILE, which the Streamlit app polls.

Run (in a separate terminal from capture_producer.py):
    python stream_processor.py
"""
import json
import os
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

import config

SCHEMA = StructType([
    StructField("duration", DoubleType()),
    StructField("protocol_type", StringType()),
    StructField("service", StringType()),
    StructField("flag", StringType()),
    StructField("src_bytes", DoubleType()),
    StructField("dst_bytes", DoubleType()),
    StructField("land", IntegerType()),
    StructField("wrong_fragment", IntegerType()),
    StructField("urgent", IntegerType()),
    StructField("src_ip", StringType()),
    StructField("dst_ip", StringType()),
    StructField("sport", IntegerType()),
    StructField("dport", IntegerType()),
    StructField("captured_at", StringType()),
])


def main():

    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] += os.pathsep + r"C:\hadoop\bin"
    # if "HADOOP_HOME" in os.environ:
    #     del os.environ["HADOOP_HOME"]
    
    # 2. Bind the virtual environment Python interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    # 3. Prevent Windows network routing errors
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    spark = SparkSession.builder.master("local[*]").appName("IDS-Streaming").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs(config.STREAM_INPUT_DIR, exist_ok=True)
    os.makedirs(config.STREAM_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    pipeline_model = PipelineModel.load(os.path.join(config.MODEL_DIR, "preprocessing_pipeline"))
    rf_model = RandomForestClassificationModel.load(os.path.join(config.MODEL_DIR, "random_forest"))

    raw_stream = (
        spark.readStream
        .schema(SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .json(config.STREAM_INPUT_DIR)
    )

    def process_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return

        transformed = pipeline_model.transform(batch_df)
        predicted = rf_model.transform(transformed)

        result = predicted.select(
            "captured_at", "src_ip", "dst_ip", "sport", "dport",
            "protocol_type", "service", "flag", "prediction",
        )
        rows = [r.asDict() for r in result.collect()]
        if not rows:
            return

        with open(config.PREDICTIONS_FILE, "a") as f:
            for r in rows:
                r["prediction"] = int(r["prediction"])  # attack=1, normal=0
                f.write(json.dumps(r) + "\n")

        n_attacks = sum(1 for r in rows if r["prediction"] == 1)
        print(f"[batch {batch_id}] {len(rows)} flows scored, {n_attacks} flagged as attack")

    query = (
        raw_stream.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", config.STREAM_CHECKPOINT_DIR)
        .trigger(processingTime="5 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
