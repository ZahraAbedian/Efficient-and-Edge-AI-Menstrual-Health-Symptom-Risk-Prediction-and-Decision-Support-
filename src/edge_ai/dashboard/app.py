import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from edge_ai.monitoring.metrics import (
    DB_PATH,
    setup_database, get_recent_inferences, get_recent_snapshots,
    measure_all, log_system_snapshot,
)


def _load_or_init_db(db_path: str):
    setup_database(db_path)
    return db_path


def _risk_color(risk: str) -> str:
    return {"Low": "green", "Medium": "orange", "High": "red"}.get(risk, "gray")


def _build_dashboard(
    db_path: str,
    model_manager: Optional[object] = None,
):
    log_system_snapshot(db_path)

    def refresh_inferences():
        df = get_recent_inferences(limit=200, db_path=db_path)
        return df

    def refresh_snapshots():
        df = get_recent_snapshots(limit=200, db_path=db_path)
        return df

    def make_risk_tab():
        df = refresh_inferences()
        if df.empty:
            return (
                "No data yet.",
                go.Figure(),
                go.Figure(),
            )

        latest = df.iloc[0]
        risk = latest.get("risk_level", "Unknown")
        prob = latest.get("probability", 0)
        ts = pd.to_datetime(latest["timestamp"]).strftime("%H:%M:%S")
        summary = (
            f"## Latest Prediction\n\n"
            f"- **Risk Level**: {risk}\n"
            f"- **Probability**: {prob:.3f}\n"
            f"- **Time**: {ts}\n"
            f"- **Backend**: {latest.get('model_backend', 'N/A')}"
        )

        df_sorted = df.sort_values("timestamp")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=pd.to_datetime(df_sorted["timestamp"]),
            y=df_sorted["probability"],
            mode="lines+markers",
            name="Risk Probability",
            line=dict(color="purple"),
        ))
        fig_trend.update_layout(
            title="Risk Probability Trend",
            yaxis_title="Probability",
            xaxis=dict(title="Time", tickformat="%H:%M:%S"),
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
        )

        risk_counts = df["risk_level"].value_counts()
        colors = [_risk_color(r) for r in risk_counts.index]
        fig_dist = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_sequence=colors,
        )
        fig_dist.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        return summary, fig_trend, fig_dist

    def make_latency_tab():
        df = refresh_inferences()
        if df.empty or "latency_mean_ms" not in df.columns:
            return go.Figure(), go.Figure(), "No latency data."

        df_sorted = df.dropna(subset=["latency_mean_ms"]).sort_values("timestamp")
        if df_sorted.empty:
            return go.Figure(), go.Figure(), "No latency data."

        fig_latency = go.Figure()
        fig_latency.add_trace(go.Scatter(
            x=pd.to_datetime(df_sorted["timestamp"]),
            y=df_sorted["latency_mean_ms"],
            mode="lines+markers",
            name="Mean Latency",
            line=dict(color="blue"),
        ))
        for col, label, dash in [("latency_p95_ms", "P95", "dash"),
                                   ("latency_p99_ms", "P99", "dot")]:
            if col in df_sorted.columns and df_sorted[col].notna().any():
                fig_latency.add_trace(go.Scatter(
                    x=pd.to_datetime(df_sorted["timestamp"]),
                    y=df_sorted[col],
                    mode="lines", name=label,
                    line=dict(dash=dash, color={"P95": "orange", "P99": "red"}[label]),
                ))
        fig_latency.update_layout(
            title="Inference Latency Over Time",
            yaxis_title="Latency (ms)",
            xaxis=dict(title="Time", tickformat="%H:%M:%S"),
            template="plotly_white",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
        )

        latest_lat = df_sorted.iloc[-1]
        throttled_str = latest_lat.get("throttled", "")
        throttled_note = ""
        if throttled_str and throttled_str != "0x0":
            throttled_note = f"\n- ⚠️ **Throttled**: {throttled_str}"
        stats = (
            f"**Latest Latency:**\n"
            f"- Mean: {latest_lat.get('latency_mean_ms', 'N/A'):.2f} ms\n"
            f"- P95: {latest_lat.get('latency_p95_ms', 'N/A'):.2f} ms\n"
            f"- P99: {latest_lat.get('latency_p99_ms', 'N/A'):.2f} ms\n"
            f"- Std: {latest_lat.get('latency_std_ms', 'N/A'):.2f} ms"
            f"{throttled_note}"
        )

        # Build resources figure from system_snapshots, falling back to inference_logs
        snap = refresh_snapshots()
        if not snap.empty:
            sys_df = snap.sort_values("timestamp")
        else:
            sys_df = df_sorted.copy()

        sys_ts = pd.to_datetime(sys_df["timestamp"])

        resource_plots = [
            ("cpu_percent", "CPU %", "green", None, None),
            ("temperature_celsius", "Temp (°C)", "red", None, None),
            ("power_watts", "Power (W)", "orange", "dash", None),
            ("core_voltage", "Core V", "brown", None, None),
            ("core_volts", "Core V", "brown", None, "inference_logs"),
            ("ram_used_mb", "RAM Used (MB)", "purple", None, None),
            ("ram_mb", "RAM RSS (MB)", "purple", "dot", "inference_logs"),
            ("disk_used_gb", "Disk Used (GB)", "cyan", None, None),
        ]

        active_rows = []
        for col, label, color, dash, source in resource_plots:
            if col not in sys_df.columns:
                continue
            if source == "inference_logs" and not snap.empty:
                continue
            y = sys_df[col]
            if y.notna().sum() < 1:
                continue
            active_rows.append((col, label, color, dash, y))

        if len(active_rows) == 0:
            fig_resources = go.Figure()
            fig_resources.update_layout(
                template="plotly_white", height=350,
                title="System Resources (no data)",
            )
        else:
            from plotly.subplots import make_subplots
            n = len(active_rows)
            fig_resources = make_subplots(
                rows=n, cols=1,
                subplot_titles=[r[1] for r in active_rows],
                vertical_spacing=0.08 / n,
            )
            for i, (col, label, color, dash, y) in enumerate(active_rows, 1):
                fig_resources.add_trace(
                    go.Scatter(
                        x=sys_ts, y=y, mode="lines+markers",
                        name=label, line=dict(color=color, dash=dash) if dash else dict(color=color),
                    ),
                    row=i, col=1,
                )
                fig_resources.update_xaxes(
                    title_text="Time", tickformat="%H:%M:%S",
                    row=i, col=1,
                )
            fig_resources.update_layout(
                title="System Resources",
                template="plotly_white",
                height=80 + 110 * n,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
            )

        return fig_latency, fig_resources, stats

    def make_explain_tab():
        if model_manager is None:
            return (
                "**Explainability not available.**\n\n"
                "No model loaded. Relaunch with:\n"
                "`python -m edge_ai.dashboard.app --model-dir ~/edge_ai_models`",
                go.Figure(),
            )

        try:
            latest = refresh_inferences()
            if latest.empty:
                return "No predictions yet.", go.Figure()

            sample = model_manager.get_latest_input()
            if sample is None:
                return "No input data available.", go.Figure()

            result = model_manager.explain(sample)
            explanation = result.get("explanation", {})
            top = explanation.get("top_features", [])

            text = (
                f"**Risk Level**: {result.get('risk_level', 'N/A')}\n"
                f"**Probability**: {result.get('probability', 0):.3f}\n"
                f"**Method**: {explanation.get('method', 'N/A')}\n\n"
                f"### Top Contributing Factors\n"
            )

            if top:
                for feat in top:
                    direction = feat.get("impact_direction", "")
                    arrow = "↑" if "increase" in direction else "↓"
                    val = feat.get("shap_value", feat.get("coefficient", 0))
                    text += f"- {feat['feature']}: {val:.4f} {arrow}\n"

                card = model_manager.make_planning_card(explanation)
                text += f"\n### Planning Card\n{card}"

                fig = go.Figure()
                names = [f["feature"] for f in top][::-1]
                vals = [f.get("shap_value", f.get("coefficient", 0)) for f in top][::-1]
                colors = ["red" if v > 0 else "green" for v in vals]
                fig.add_trace(go.Bar(
                    x=vals, y=names,
                    orientation="h",
                    marker_color=colors,
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="SHAP Value / Coefficient",
                    template="plotly_white",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
            else:
                text += "No feature importance data available."
                fig = go.Figure()

            return text, fig
        except Exception as e:
            return f"Error: {e}", go.Figure()

    def make_history_tab():
        df = refresh_inferences()
        if df.empty:
            return go.Figure()

        df_sorted = df.sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_sorted["timestamp"]),
            y=df_sorted["probability"],
            mode="markers",
            marker=dict(
                size=10,
                color=df_sorted["probability"],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Probability"),
            ),
            name="Predictions",
        ))
        if "latency_mean_ms" in df_sorted.columns:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_sorted["timestamp"]),
                y=df_sorted["latency_mean_ms"],
                mode="lines+markers",
                name="Latency (ms)",
                yaxis="y2",
                line=dict(color="gray", dash="dot"),
            ))
        fig.update_layout(
            title="Prediction History",
            xaxis=dict(title="Time", tickformat="%H:%M:%S"),
            yaxis=dict(title="Probability", range=[0, 1]),
            yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
            template="plotly_white",
            height=500,
            margin=dict(l=40, r=50, t=40, b=40),
        )
        return fig

    def export_inferences_csv():
        df = get_recent_inferences(limit=100000, db_path=db_path)
        tmp = Path(tempfile.gettempdir()) / "edge_ai_inference_logs.csv"
        df.to_csv(tmp, index=False)
        return str(tmp)

    def export_snapshots_csv():
        df = get_recent_snapshots(limit=100000, db_path=db_path)
        tmp = Path(tempfile.gettempdir()) / "edge_ai_system_snapshots.csv"
        df.to_csv(tmp, index=False)
        return str(tmp)

    def make_export_tab():
        conn = sqlite3.connect(str(db_path))
        inf_count = conn.execute(
            "SELECT COUNT(*) FROM inference_logs"
        ).fetchone()[0]
        snap_count = conn.execute(
            "SELECT COUNT(*) FROM system_snapshots"
        ).fetchone()[0]
        try:
            first_ts = conn.execute(
                "SELECT MIN(timestamp) FROM inference_logs"
            ).fetchone()[0] or "N/A"
            last_ts = conn.execute(
                "SELECT MAX(timestamp) FROM inference_logs"
            ).fetchone()[0] or "N/A"
        except Exception:
            first_ts = last_ts = "N/A"
        conn.close()

        db_path_obj = Path(db_path)
        db_size = db_path_obj.stat().st_size / 1024 if db_path_obj.exists() else 0

        results_csv = None
        if model_manager is not None:
            candidate = Path(model_manager.MODEL_DIR) / "raspberry_pi_edge_results.csv"
            if candidate.exists():
                results_csv = str(candidate)

        summary = (
            f"**Monitoring Database:** `{db_path}`\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Total Inferences | {inf_count} |\n"
            f"| Total Snapshots | {snap_count} |\n"
            f"| DB File Size | {db_size:.1f} KB |\n"
            f"| First Inference | {first_ts} |\n"
            f"| Last Inference | {last_ts} |\n"
        )
        if results_csv:
            summary += f"\n**Results CSV:** `{results_csv}`\n"

        return summary, results_csv

    _risk_init = make_risk_tab()
    _explain_init = make_explain_tab()
    _latency_init = make_latency_tab()
    _history_init = make_history_tab()
    _export_init = make_export_tab()

    with gr.Blocks(
        title="Edge AI Monitor",
    ) as dashboard:
        gr.Markdown("# Edge AI Symptom Risk Monitor")

        with gr.Tabs():
            with gr.Tab("Risk Monitor"):
                risk_summary = gr.Markdown(value=_risk_init[0])
                risk_trend = gr.Plot(value=_risk_init[1])
                risk_dist = gr.Plot(value=_risk_init[2])
                refresh_risk = gr.Button("Refresh")
                refresh_risk.click(
                    fn=lambda: make_risk_tab(),
                    outputs=[risk_summary, risk_trend, risk_dist],
                )

            with gr.Tab("Explainability"):
                explain_text = gr.Markdown(value=_explain_init[0])
                explain_plot = gr.Plot(value=_explain_init[1])
                refresh_explain = gr.Button("Refresh")
                refresh_explain.click(
                    fn=lambda: make_explain_tab(),
                    outputs=[explain_text, explain_plot],
                )

            with gr.Tab("System Health"):
                latency_plot = gr.Plot(value=_latency_init[0])
                resources_plot = gr.Plot(value=_latency_init[1])
                latency_stats = gr.Markdown(value=_latency_init[2])
                refresh_health = gr.Button("Refresh")
                refresh_health.click(
                    fn=lambda: make_latency_tab(),
                    outputs=[latency_plot, resources_plot, latency_stats],
                )

            with gr.Tab("History"):
                history_plot = gr.Plot(value=_history_init)
                refresh_history = gr.Button("Refresh")
                refresh_history.click(
                    fn=lambda: make_history_tab(),
                    outputs=[history_plot],
                )

            with gr.Tab("Data Export"):
                export_summary = gr.Markdown(value=_export_init[0])
                gr.Markdown("### Download Data")
                with gr.Row():
                    inf_btn = gr.Button("📥 Download Inference Logs (CSV)")
                    snap_btn = gr.Button("📥 Download System Snapshots (CSV)")
                inf_download = gr.File(label="Inference Logs", visible=False)
                snap_download = gr.File(label="System Snapshots", visible=False)
                inf_btn.click(
                    fn=export_inferences_csv,
                    outputs=[inf_download],
                ).then(
                    fn=lambda f: gr.File(visible=True, value=f),
                    inputs=[inf_download],
                    outputs=[inf_download],
                )
                snap_btn.click(
                    fn=export_snapshots_csv,
                    outputs=[snap_download],
                ).then(
                    fn=lambda f: gr.File(visible=True, value=f),
                    inputs=[snap_download],
                    outputs=[snap_download],
                )
                if _export_init[1]:
                    gr.Markdown(
                        "### Results CSV from Inference Test\n"
                        f"`{_export_init[1]}` — available on the file system."
                    )

        gr.Timer(value=10).tick(
            fn=make_risk_tab,
            outputs=[risk_summary, risk_trend, risk_dist],
        )
        gr.Timer(value=10).tick(
            fn=make_latency_tab,
            outputs=[latency_plot, resources_plot, latency_stats],
        )

    return dashboard


def run(
    db_path: str = str(Path.home() / ".edge_ai_monitoring.db"),
    share: bool = False,
    port: int = 7860,
    model_manager: Optional[object] = None,
    model_dir: Optional[str] = None,
):
    if model_manager is None and model_dir:
        from edge_ai.xai.explainer import ModelManager
        model_manager = ModelManager(model_dir)

    _load_or_init_db(db_path)
    dashboard = _build_dashboard(db_path, model_manager)
    dashboard.launch(
        server_name="0.0.0.0", server_port=port, share=share,
        theme=gr.themes.Soft(),
        css="footer {display:none !important}",
    )


def main():
    parser = argparse.ArgumentParser(description="Edge AI Monitoring Dashboard")
    parser.add_argument("--db", default=str(Path.home() / ".edge_ai_monitoring.db"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--model-dir", default=None,
        help="Directory containing model artifacts (pipeline, features, threshold, sample_input). "
             "Enables the Explainability tab.",
    )
    args = parser.parse_args()
    run(db_path=args.db, port=args.port, share=args.share, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
