# Real-Time Network Anomaly Detection System

A network intrusion detection pipeline that captures live traffic, scores it against a model trained on the **NSL-KDD** benchmark, and streams the verdicts to a live dashboard — built on **Apache Spark Structured Streaming**, **PySpark MLlib**, and **Streamlit**.

---

## How it works

```
┌────────────────────┐     flow batches (JSON)      ┌────────────────────────┐
│  capture_producer  │ ───────────────────────────▶ │  stream_input/        │
│  (tshark / pyshark)│                              │                        │ 
└────────────────────┘                              └────────────┬───────────┘
   sniffs live packets,                                          │
   aggregates into 5-tuple                                       ▼
   flows, flushes every 5s                           ┌───────────────────────────┐
                                                     │   stream_processor.py     │
                                                     │  Spark Structured         │
                                                     │  Streaming + saved        │
                                                     │  Pipeline + RF model      │
                                                     └──────────┬────────────────┘
                                                                │ predictions.jsonl
                                                                ▼
                                                     ┌───────────────────────────┐
                                                     │        app.py             │
                                                     │  Streamlit dashboard      │
                                                     │  (live / perf / history)  │
                                                     └───────────────────────────┘
```

1. **`capture_producer.py`** sniffs live packets with `tshark` (via `pyshark`), groups them into flows keyed by the standard 5-tuple (`src_ip`, `sport`, `dst_ip`, `dport`, `protocol`), and periodically flushes completed flows as newline-delimited JSON into a watched folder.
2. **`stream_processor.py`** watches that folder with **Spark Structured Streaming**, applies the saved preprocessing `Pipeline` + trained model on every micro-batch, and appends predictions (`0` = normal, `1` = attack) to a predictions log.
3. **`app.py`** is a **Streamlit** dashboard with three tabs — Live Monitoring, Model Performance, and Explore History — that reads that log and the training metrics to visualize everything in real time.

## Model training

- **`train_basic.py`** trains on the **9 "basic" NSL-KDD features** (`duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`) — deliberately restricted to the subset of NSL-KDD's 41 features that can be **honestly recomputed from a raw live packet capture** without a dedicated flow-extraction tool like CICFlowMeter. This is the model used for real-time scoring.
- A `Pipeline` of `StringIndexer` (categorical columns) → `VectorAssembler` → `StandardScaler` feeds two classifiers: a **Random Forest** (100 trees, max depth 15) and a **Logistic Regression**, both evaluated with `BinaryClassificationEvaluator` / `MulticlassClassificationEvaluator` (AUC, accuracy, F1, precision, recall, confusion matrix) in **`eval_utils.py`**.
- **`predict_batch.py`** is a quick offline sanity check — it reloads the saved pipeline + Random Forest model and scores `KDDTest+.txt` before wiring up live capture.
- On the NSL-KDD test set the Random Forest reaches ~99% accuracy and the Logistic Regression baseline ~92%.

## Tech stack

| Layer | Tools |
|---|---|
| Stream processing | Apache Spark (Structured Streaming), PySpark MLlib |
| Packet capture | `tshark` (Wireshark CLI) via `pyshark` |
| ML models | Random Forest, Logistic Regression (Spark MLlib) |
| Dashboard | Streamlit, Plotly, Matplotlib, Seaborn, Pandas |
| Dataset | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) |

## Project structure

```
├── app.py                 # Streamlit dashboard (Live / Performance / Explore tabs)
├── capture_producer.py    # tshark-based live packet capture -> flow batches
├── stream_processor.py    # Spark Structured Streaming scorer
├── train_basic.py         # Trains the real-time (9-feature) RF + LR models
├── predict_batch.py       # Offline batch sanity check against KDDTest+.txt
├── eval_utils.py          # Evaluation metrics + confusion matrix helpers
├── data_provider.py       # Loads predictions/metrics for the dashboard
├── config.py               # Paths, feature lists, directory layout
├── KDDTrain+.txt           # NSL-KDD training set
└── KDDTest+.txt            # NSL-KDD test set
```

## Getting started

### Prerequisites

- Python 3.9+
- Java 8/11 (required by Spark) and `pyspark`
- [tshark](https://www.wireshark.org/) installed and on your `PATH` (`sudo apt install tshark` on Linux; on Linux you'll also need root or membership in the `wireshark` group for live capture)
- `pip install pyspark pyshark streamlit pandas plotly matplotlib seaborn`

### 1. Train the model

```bash
python train_basic.py
```
Trains and saves the preprocessing pipeline, Random Forest, and Logistic Regression models to `network_models_basic/`, along with `metrics.json`.

### 2. (Optional) Sanity-check on the test set

```bash
python predict_batch.py
```

### 3. Start live capture (terminal 1)

```bash
python capture_producer.py --interface eth0
```
Use `tshark -D` to list available interface names on your machine.

### 4. Start the stream scorer (terminal 2)

```bash
python stream_processor.py
```

### 5. Launch the dashboard (terminal 3)

```bash
streamlit run app.py
```

## Known limitations

- `service` and `flag` (the TCP-state feature from NSL-KDD) are **approximated with simple heuristics** from live packet flags in `capture_producer.py` — they won't always match NSL-KDD's original labelling exactly. This is sufficient to demonstrate the pipeline end-to-end but shouldn't be treated as ground truth.
- Only the 9 "basic" features are used for live scoring; the full 41-feature NSL-KDD set (`num_failed_logins`, `dst_host_srv_rerror_rate`, etc.) would need a dedicated flow-extraction tool and longer-lived connection history to compute from raw packets.
