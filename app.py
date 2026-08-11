"""
Streamlit dashboard for the IDS backend.

Three tabs:
- Live Monitoring: auto-refreshing view of flows scored by stream_processor.py
- Model Performance: compares the basic (real-time) and full (offline) models
- Explore: filterable table of historical predictions

Run:
    streamlit run app.py

Assumes train_basic.py (and optionally train_full.py) have already been
run, and that capture_producer.py + stream_processor.py are running in
separate terminals if you want the Live tab to show anything.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

import config
from data_provider import load_metrics, load_predictions

st.set_page_config(page_title="Anomaly-Based IDS", layout="wide")
sns.set_theme(style="whitegrid")

st.title("Anomaly-Based Intrusion Detection Dashboard")

tab_live, tab_perf, tab_explore = st.tabs(["Live Monitoring", "Model Performance", "Explore History"])

# ---------------------------------------------------------------------
# LIVE MONITORING
# ---------------------------------------------------------------------
with tab_live:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)
        if st.button("Refresh now"):
            st.rerun()
    

    df = load_predictions(limit=1000)

    if df.empty:
        st.info("No predictions yet. Start capture_producer.py and stream_processor.py to see live data.")
    else:
        total = len(df)
        attacks = int((df["prediction"] == 1).sum())
        attack_rate = attacks / total * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Flows scored (recent)", total)
        m2.metric("Flagged as attack", attacks)
        m3.metric("Attack rate", f"{attack_rate:.1f}%")

        left, right = st.columns(2)

        with left:
            st.subheader("Attacks over time")
            ts = df.copy()
            ts["minute"] = ts["captured_at"].dt.floor("min")
            counts = ts.groupby(["minute", "verdict"]).size().reset_index(name="count")
            fig = px.line(
                counts, x="minute", y="count", color="verdict",
                markers=True, color_discrete_map={"normal": "#2ca02c", "attack": "#d62728"},
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Verdict by protocol")
            proto_counts = df.groupby(["protocol_type", "verdict"]).size().reset_index(name="count")
            fig2 = px.bar(
                proto_counts, x="protocol_type", y="count", color="verdict", barmode="group",
                color_discrete_map={"normal": "#2ca02c", "attack": "#d62728"},
            )
            fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Most recent flows")
        display_cols = ["captured_at", "src_ip", "dst_ip", "sport", "dport",
                         "protocol_type", "service", "flag", "verdict"]
        st.dataframe(
            df[display_cols].head(50),
            use_container_width=True,
            column_config={
                "verdict": st.column_config.TextColumn("Verdict"),
            },
        )

# ---------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------
with tab_perf:
    st.subheader("Offline evaluation metrics")
    st.caption(
        "Model = 9 features, used for live scoring. "
    )

    basic_metrics = load_metrics(config.MODEL_DIR)
    full_metrics = load_metrics(config.MODEL_DIR_FULL)

    if not basic_metrics:
        st.info("No metrics.json found yet. Run train_basic.py")
    else:
        rows = []
        for model_name, metrics in basic_metrics.items():
            rows.append({"feature_set": "basic (real-time)", "model": model_name, **metrics})
        

        metrics_df = pd.DataFrame(rows)

        if not metrics_df.empty:
            melted = metrics_df.melt(
                id_vars=["feature_set", "model"],
                value_vars=[c for c in ["accuracy", "f1", "weightedPrecision", "weightedRecall"] if c in metrics_df.columns],
                var_name="metric", value_name="value",
            )
            melted["label"] = melted["feature_set"] + " - " + melted["model"]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            sns.barplot(data=melted, x="metric", y="value", hue="label", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_xlabel("")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(metrics_df.set_index(["feature_set", "model"]), use_container_width=True)

    # kmeans_map = full_metrics.get("kmeans_cluster_to_label") if full_metrics else None
    # if kmeans_map:
    #     st.caption(f"KMeans cluster -> label mapping (full model): {kmeans_map}")

# ---------------------------------------------------------------------
# EXPLORE HISTORY
# ---------------------------------------------------------------------
with tab_explore:
    df_all = load_predictions(limit=config.MAX_PREDICTIONS_KEPT)

    if df_all.empty:
        st.info("No historical predictions yet.")
    else:
        st.subheader("Filter")
        c1, c2, c3 = st.columns(3)
        with c1:
            verdict_filter = st.multiselect("Verdict", options=["normal", "attack"], default=["normal", "attack"])
        with c2:
            protocols = sorted(df_all["protocol_type"].dropna().unique().tolist())
            protocol_filter = st.multiselect("Protocol", options=protocols, default=protocols)
        with c3:
            ip_search = st.text_input("Search src/dst IP contains")

        filtered = df_all[
            df_all["verdict"].isin(verdict_filter) & df_all["protocol_type"].isin(protocol_filter)
        ]
        if ip_search:
            filtered = filtered[
                filtered["src_ip"].str.contains(ip_search, na=False)
                | filtered["dst_ip"].str.contains(ip_search, na=False)
            ]

        st.caption(f"{len(filtered)} of {len(df_all)} flows match filters")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download filtered results as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="ids_predictions_filtered.csv",
            mime="text/csv",
        )