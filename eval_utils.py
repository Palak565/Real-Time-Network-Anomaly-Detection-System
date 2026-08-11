from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def confusion_matrix(predictions, label_col="label", prediction_col="prediction"):
    """Returns {tp, fp, tn, fn} counts as plain ints, for plotting a confusion matrix later."""
    counts = predictions.groupBy(label_col, prediction_col).count().collect()
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in counts:
        label, pred, n = row[label_col], row[prediction_col], row["count"]
        if label == 1 and pred == 1:
            cm["tp"] += n
        elif label == 0 and pred == 1:
            cm["fp"] += n
        elif label == 0 and pred == 0:
            cm["tn"] += n
        elif label == 1 and pred == 0:
            cm["fn"] += n
    return cm


def evaluate_binary(predictions, label_col="label", prediction_col="prediction", prob_col="probability"):
    """Returns AUC, accuracy, F1, precision, recall, and a confusion matrix for a binary classifier."""
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=prob_col, metricName="areaUnderROC"
    ).evaluate(predictions)

    metrics = {"auc": auc}
    for m in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        metrics[m] = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=prediction_col, metricName=m
        ).evaluate(predictions)

    metrics["confusion_matrix"] = confusion_matrix(predictions, label_col, prediction_col)
    return metrics
