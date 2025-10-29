# =============================
# Standard Library Imports
# =============================
import base64
import json
import math
import os
import pickle
import signal
import threading
import time
from io import BytesIO
from pathlib import Path

# =============================
# Third-Party Library Imports
# =============================
import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.dependencies import ALL, MATCH
import flask
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import tensorflow as tf
import yaml
import webbrowser

# =============================
# Project-Specific Imports
# =============================
from pyprecip.modeling.cnn import _get_training_data
from pyprecip.modeling.metrics import calc_metrics

def fig_to_base64():
    """Convert current Matplotlib figure to base64 image for Dash display."""
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{encoded}"

def discover_stations(organized_dir: str):
    """Scan organized directory for JSON files and return available station IDs."""
    files = list(Path(organized_dir).glob("*.json"))
    stations = list(set([f.stem.split('_')[1][0:5] for f in files]))
    return stations

def interactive_config_builder_4_create_training(organized_dir):
    """Launch interactive Dash app for visualizing and building YAML config."""
    stations = discover_stations(organized_dir)
    if not stations:
        raise ValueError(f"No station JSON files found in {organized_dir}")

    # Default config values
    config_defaults = {
        "organized_dir": organized_dir,
        "out_dir": "D:/PROJECTS/PyPrecip/examples/outputs/training_inputs",
        "stations": stations,
        "short_cols": ["pcp"],
        "cols": ["precip"],
        "mit_cold_hours": 1,
        "mit_warm_hours": 2,
        "global_event_buffer_hours": 1,
        "event_total_pcp_threshold_4_1hr_event": 1,
        "event_total_pcp_threshold_4_larger_events": 2,
    }

    app = dash.Dash(__name__)
    app.title = "PyPrecip Interactive Configuration Builder for create_training_cmd CLI"

    # ---------------- Layout ----------------
    app.layout = html.Div([
        html.H2("🌦 PyPrecip Interactive Configuration Builder for create_training_cmd CLI"),

        html.Div([
            html.Label("Select Station to View:"),
            dcc.Dropdown(
                id="station-dropdown",
                options=[{"label": s, "value": s} for s in stations],
                value=stations[0],
                clearable=False,
            ),
        ], style={"width": "40%", "margin-bottom": "20px"}),

        html.Div([
            html.Label("Select Visualization Type:"),
            dcc.Dropdown(
                id="vis-type-dropdown",
                options=[
                    {"label": "Time Series Plot (default)", "value": "timeseries"},
                    {"label": "Missingno Matrix", "value": "matrix"},
                    {"label": "Missingno Heatmap", "value": "heatmap"},
                    {"label": "Missingno Bar", "value": "bar"},
                ],
                value="timeseries",
                clearable=False,
                style={"width": "40%", "margin-bottom": "20px"}
            ),
        ]),

        html.Div(id="plot-description", style={"margin-bottom": "20px", "font-style": "italic", "color": "#555"}),

        dcc.Graph(id="station-graph"),

        html.Hr(),

        html.H3("Configuration Settings"),

        html.Label("Select Stations for YAML Output:"),
        dcc.Dropdown(
            id="station-select",
            options=[{"label": s, "value": s} for s in stations],
            value=stations,
            multi=True,
            style={"width": "60%", "margin-bottom": "20px"}
        ),

        html.Label("Output directory:"),
        dcc.Input(
            id="out-dir", type="text", value=config_defaults["out_dir"],
            style={"width": "60%"}
        ),

        html.Br(), html.Br(),

        html.Label("Select Columns:"),
        dcc.Dropdown(
            id="cols-dropdown",
            options=[
                {"label": i, "value": i} for i in [
                    'precip', 'pressure', 'rhum', 'wndsp', 'mxwndsp', 'wnddir_nbr',
                    'radsum', 'rad', 'insolationintensity', 'insolationtime', 'tmp',
                    'minsoiltmp0cm', 'soiltmp5cm', 'soiltmp10cm', 'soiltmp20cm',
                    'soiltmp50cm', 'soiltmp100cm'
                ]
            ],
            value=config_defaults["cols"],
            multi=True,
            style={"width": "60%"}
        ),

        # Dynamic short col editors
        html.Div(id="short-cols-container", style={"margin-top": "15px"}),

        html.Br(),

        html.Label("Minimum Inter-arrival Time Index (MIT) for Cold Weather (for Oct–Mar & in Hours):"),
        dcc.Slider(0, 4, 1, value=config_defaults["mit_cold_hours"],
                   id="mit-cold-slider", marks={i: str(i) for i in range(4)}),

        html.Label("Minimum Inter-arrival Time Index (MIT) for Warm Weather (for Apr–Sep & in Hours):"),
        dcc.Slider(0, 4, 1, value=config_defaults["mit_warm_hours"],
                   id="mit-warm-slider", marks={i: str(i) for i in range(4)}),

        html.Label("Global Event Buffer Hours:"),
        dcc.Slider(0, 4, 1, value=config_defaults["global_event_buffer_hours"],
                   id="buffer-slider", marks={i: str(i) for i in range(4)}),

        html.Br(),

        html.Button("💾 Save YAML", id="save-btn", n_clicks=0, style={"font-size": "16px"}),

        html.Button(
            "🛑 Stop Server",
            id="stop-server-btn",
            n_clicks=0,
            style={
                "font-size": "16px",
                "background-color": "#c0392b",
                "color": "white",
                "margin-left": "20px"
            },
        ),

        html.Div(id="save-msg", style={"margin-top": "10px", "color": "green"}),
        html.Div(id="stop-msg", style={"margin-top": "10px", "font-weight": "bold"}),

        dcc.Store(id="config-store", data=config_defaults)
    ], style={"margin": "40px"})

    # ---------------- Callbacks ----------------
    @app.callback(
        Output("station-graph", "figure"),
        Input("station-dropdown", "value"),
        Input("vis-type-dropdown", "value")
    )
    def update_graph(station_id, vis_type):
        df = pd.read_json(os.path.join(organized_dir, f'station_{station_id}.json'))
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return go.Figure()

        if vis_type == "timeseries":
            fig = go.Figure()
            for col in numeric_cols:
                if col.lower() == "precip":
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col,
                        line=dict(width=2, color="#2980b9"),
                        visible=True
                    ))
                else:
                    # other cols: initially hidden but clickable in legend
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col,
                        line=dict(width=1),
                        visible="legendonly"
                    ))
            fig.update_layout(
                title=f"Station {station_id} — Precip (others available via legend click)",
                template="plotly_white",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                legend_title="Columns (click to show/hide)"
            )
            return fig

        # --- Missingno visualizations ---
        import plotly.express as px  # optional, not needed for these
        if vis_type == "matrix":
            msno.matrix(df)
            plt.title(f"Station {station_id} — Missingness Matrix")
        elif vis_type == "heatmap":
            msno.heatmap(df)
            plt.title(f"Station {station_id} — Missingness Heatmap")
        elif vis_type == "bar":
            msno.bar(df)
            plt.title(f"Station {station_id} — Missingness Bar")
        else:
            return go.Figure()

        # --- Convert matplotlib figure to image and embed neatly in a Plotly figure ---
        img_str = fig_to_base64()

        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=img_str,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below"
            )
        )

        # Clean white background with no colored borders or axes
        fig.update_layout(
            width=900,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        return fig

    # === Plot Explanations ===
    @app.callback(
        Output("plot-description", "children"),
        Input("vis-type-dropdown", "value")
    )
    def update_plot_description(vis_type):
        """Return an explanatory text based on selected visualization type."""
        descriptions = {
            "timeseries": (
                "📈 The time series plot shows the values of each numeric column over time. "
                "‘precip’ is always displayed by default. Other columns can be toggled on/off via the legend."
            ),
            "matrix": (
                "🧩 The Missingno Matrix visualizes the **pattern of missing values** in the dataset. "
                "Each column appears as a vertical bar — white gaps represent missing data points, "
                "helping you see when (and in which variables) data is missing."
            ),
            "heatmap": (
                "🔥 The Missingno Heatmap shows **correlations between missing values** across columns. "
                "Darker colors indicate columns whose missing values tend to occur together. "
                "It helps identify systematic gaps — for example, one sensor going offline causing multiple missing fields."
            ),
            "bar": (
                "📊 The Missingno Bar chart displays the **count and proportion of available vs missing data** for each column. "
                "This gives a quick completeness overview — taller bars mean more complete data."
            ),
        }
        return descriptions.get(vis_type, "")

    # === Dynamic short_cols Input fields depending on selected columns ===
    @app.callback(
        Output("short-cols-container", "children"),
        Input("cols-dropdown", "value")
    )
    def update_short_cols_inputs(selected_cols):
        fixed_mapping = {"precip": "pcp", "tmp": "tmp", "rhum": "rhum"}

        items = []
        for col in selected_cols:
            if col in fixed_mapping:
                # Non-editable input for fixed mappings
                items.append(
                    html.Div([
                        html.Label(f"Short name for '{col}':"),
                        dcc.Input(
                            value=fixed_mapping[col],
                            readOnly=True,
                            style={"margin-bottom": "5px", "width": "50%", "backgroundColor": "#f0f0f0"},
                        )
                    ])
                )
            else:
                # Editable input for others
                items.append(
                    html.Div([
                        html.Label(f"Short name for '{col}':"),
                        dcc.Input(
                            id={"type": "short-col-input", "index": col},
                            type="text",
                            placeholder=f"short name for {col}",
                            style={"margin-bottom": "5px", "width": "50%"}
                        )
                    ])
                )
        return items

    # === Save YAML callback (reads dynamic short_col values and station selection) ===
    @app.callback(
        Output("save-msg", "children"),
        Input("save-btn", "n_clicks"),
        State("out-dir", "value"),
        State("station-select", "value"),
        State("cols-dropdown", "value"),
        State({"type": "short-col-input", "index": ALL}, "value"),
        State("mit-cold-slider", "value"),
        State("mit-warm-slider", "value"),
        State("buffer-slider", "value"),
    )
    def save_yaml(n_clicks, out_dir, selected_stations, cols, short_col_values,
                  cold, warm, buffer_hours):
        if n_clicks == 0:
            return ""

        fixed_mapping = {"precip": "pcp", "tmp": "tmp", "rhum": "rhum"}

        short_cols = []
        editable_i = 0
        for col in cols:
            if col in fixed_mapping:
                short_cols.append(fixed_mapping[col])
            else:
                val = (short_col_values[editable_i] if editable_i < len(short_col_values) and short_col_values[editable_i] else col[:5])
                short_cols.append(val)
                editable_i += 1

        config = {
            "organized_dir": organized_dir,
            "out_dir": out_dir,
            "stations": selected_stations,
            "short_cols": short_cols,
            "cols": cols,
            "mit_cold_hours": cold,
            "mit_warm_hours": warm,
            "global_event_buffer_hours": buffer_hours,
            "event_total_pcp_threshold_4_1hr_event": 1,
            "event_total_pcp_threshold_4_larger_events": 2
        }

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        yaml_path = Path(out_dir) / "create_training.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False, allow_unicode=True)

        return f"✅ Configuration saved successfully at: {yaml_path}"

    # --------------- STOP SERVER (safe cross-env) ---------------
    @app.callback(
        Output("stop-msg", "children"),
        Input("stop-server-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def stop_server(n_clicks):
        """Attempt to stop the Dash server gracefully, fallback to kill process."""
        func = flask.request.environ.get("werkzeug.server.shutdown")

        def kill_self():
            time.sleep(1.5)
            os.kill(os.getpid(), signal.SIGTERM)

        if func is None:
            threading.Thread(target=kill_self).start()
            return "🛑 Forcing shutdown (non‑Werkzeug environment)..."

        def delayed_shutdown():
            time.sleep(1.5)
            func()

        threading.Thread(target=delayed_shutdown).start()
        return "🛑 Shutting down server..."

    # ---------------- Run the Dash app ----------------
    port = 8060
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Starting PyPrecip visualizer at {url}\nPress CTRL+C to stop manually.\n")

    webbrowser.open(url)
    app.run(debug=True, port=port, use_reloader=False)





def interactive_training_data_visualizer(cfg):
    """
    Interactive Dash app to explore and re-bin training/validation/test targets
    with editable *continuous* class intervals.
    Shows a table with sample counts for train/val/test per class.
    """

    os.makedirs(cfg.model_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def save_yaml_config(cfg_obj, intervals):
        cfg_obj.class_intervals = intervals
        model_dir = getattr(cfg_obj, "model_dir", os.getcwd())
        save_path = Path(model_dir, "updated_cfg.yaml").as_posix()
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_obj, f, sort_keys=False)
            return f"✅ Configuration saved to: {save_path}"
        except Exception as e:
            return f"⚠️ Could not save configuration: {e}"

    def load_target_data(cfg_obj):
        try:
            Xtr, Xv, Xte, ytr, yv, yte, ncls, means = _get_training_data(cfg_obj)
            return {
                "train": np.array(ytr),
                "val": np.array(yv),
                "test": np.array(yte)
            }, ncls, means
        except Exception as e:
            print(f"⚠️ Data load failed: {e}")
            return {"train": np.array([]), "val": np.array([]), "test": np.array([])}, 0, None

    def compute_sample_counts(y_dict):
        """Create a small HTML table showing sample counts per class."""
        if not any(len(v) for v in y_dict.values()):
            return html.Div("No data available.")

        unique_all = sorted(set(np.concatenate([np.unique(v) for v in y_dict.values()])))
        header = html.Tr([
            html.Th("Class"), html.Th("Train", style={"text-align": "right"}),
            html.Th("Val", style={"text-align": "right"}),
            html.Th("Test", style={"text-align": "right"}),
        ])
        rows = [header]
        for cls in unique_all:
            train_c = np.sum(y_dict["train"] == cls)
            val_c = np.sum(y_dict["val"] == cls)
            test_c = np.sum(y_dict["test"] == cls)
            rows.append(html.Tr([
                html.Td(str(cls)),
                html.Td(f"{train_c:,}", style={"text-align": "right"}),
                html.Td(f"{val_c:,}", style={"text-align": "right"}),
                html.Td(f"{test_c:,}", style={"text-align": "right"}),
            ]))
        return html.Table(rows, style={
            "border-collapse": "collapse", "width": "100%",
            "margin-top": "10px", "border": "1px solid #ccc",
        })

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    y_dict, n_classes, class_means = load_target_data(cfg)
    class_intervals = [list(i) for i in cfg.class_intervals]

    app = Dash(__name__)
    app.title = "Training / Validation / Test Data Visualizer"

    # interval editor builder
    def make_interval_inputs(intervals):
        items = []
        for i, (low, high) in enumerate(intervals):
            items.append(
                html.Tr([
                    html.Td(f"Class {i}"),
                    html.Td(dcc.Input(
                        id={"type": "low", "index": i},
                        type="number", value=low, step=0.1,
                        debounce=True, style={"width": "80px"})),
                    html.Td("–"),
                    html.Td(dcc.Input(
                        id={"type": "high", "index": i},
                        type="number", value=high, step=0.1,
                        debounce=True, style={"width": "80px"}))
                ])
            )
        return html.Table(items, style={"margin-bottom": "10px"})

    # -------------------------------------------------------------
    # LAYOUT (Plot first, Intervals + Table below)
    # -------------------------------------------------------------
    app.layout = html.Div([

        html.H2("📊 Interactive Training Data Visualizer"),

        # dataset + visualization selectors
        html.Div([
            html.Div([
                html.Label("Select Dataset:"),
                dcc.Dropdown(
                    id="data-dropdown",
                    options=[
                        {"label": "Train", "value": "train"},
                        {"label": "Validation", "value": "val"},
                        {"label": "Test", "value": "test"},
                    ],
                    value="train",
                    clearable=False,
                    style={"width": "45%", "margin-bottom": "20px"},
                ),
            ]),
            html.Div([
                html.Label("Select Visualization Type:"),
                dcc.Dropdown(
                    id="vis-type",
                    options=[
                        {"label": "Scatter Plot", "value": "scatter"},
                        {"label": "Bar Chart", "value": "bar"},
                    ],
                    value="bar",
                    clearable=False,
                    style={"width": "40%", "margin-bottom": "20px"},
                ),
            ]),
        ]),

        html.Hr(),
        html.Div(id="plot-description", style={
            "margin-bottom": "10px", "font-style": "italic", "color": "#555",
        }),
        dcc.Graph(id="main-graph"),  # <---- plot comes before intervals

        html.Hr(),
        html.H4("🎯 Class Intervals & Sample Counts"),

        # intervals + table BELOW the plot
        html.Div([
            html.Div([
                html.H5("Editable Intervals"),
                html.Div(id="intervals-container", children=make_interval_inputs(class_intervals)),
            ], style={"display": "inline-block", "verticalAlign": "top",
                      "width": "45%", "margin-right": "3%"}),

            html.Div([
                html.H5("Sample Count per Class"),
                html.Div(id="sample-table", children=compute_sample_counts(y_dict)),
            ], style={"display": "inline-block", "verticalAlign": "top", "width": "45%"})
        ]),

        html.Div([
            html.Button("🔄 Update Classes", id="update-btn",
                        style={"margin-top": "10px", "background-color": "#27ae60",
                               "color": "white", "font-size": "14px", "margin-right": "12px"}),
            html.Button("💾 Save Configuration", id="save-btn",
                        style={"margin-top": "10px", "background-color": "#2980b9",
                               "color": "white", "font-size": "14px"}),
        ]),
        html.Div(id="update-msg", style={"margin-top": "8px", "color": "green"}),
        html.Div(id="save-msg", style={"margin-top": "4px", "color": "dodgerblue"}),

        html.Hr(),
        html.Button("🛑 Stop Server", id="stop-server-btn", n_clicks=0,
                    style={"font-size": "16px", "background-color": "#c0392b", "color": "white"}),
        html.Div(id="stop-msg", style={"margin-top": "10px", "font-weight": "bold"}),

    ], style={"margin": "40px"})

    # -------------------------------------------------------------
    # Keep intervals continuous
    # -------------------------------------------------------------
    @app.callback(
        Output({"type": "low", "index": ALL}, "value"),
        Input({"type": "high", "index": ALL}, "value"),
        State({"type": "low", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def keep_continuous(highs, lows):
        updated = list(lows)
        for i in range(len(lows) - 1):
            if highs[i] is not None:
                updated[i + 1] = highs[i]
        return updated

    # -------------------------------------------------------------
    # Update intervals → refresh data + sample table
    # -------------------------------------------------------------
    @app.callback(
        Output("update-msg", "children"),
        Output("sample-table", "children"),
        Input("update-btn", "n_clicks"),
        State({"type": "low", "index": ALL}, "value"),
        State({"type": "high", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def update_intervals(n_clicks, lows, highs):
        nonlocal y_dict, class_intervals, n_classes, class_means, cfg
        intervals = [[l, h] for l, h in zip(lows, highs) if l is not None and h is not None]
        if not intervals:
            return "⚠️ Invalid intervals.", compute_sample_counts(y_dict)

        # ensure continuity & expand last bin to max value if needed
        max_y_val = 0
        for key in ("train", "val", "test"):
            if len(y_dict[key]) > 0:
                max_y_val = max(max_y_val, float(np.max(y_dict[key])))
        if intervals[-1][1] < max_y_val:
            intervals[-1][1] = max_y_val

        cfg.class_intervals = intervals
        class_intervals = intervals
        y_dict, n_classes, class_means = load_target_data(cfg)
        return (f"✅ Updated & recalculated y sets ({len(intervals)} classes).",
                compute_sample_counts(y_dict))

    # -------------------------------------------------------------
    # Save configuration
    # -------------------------------------------------------------
    @app.callback(
        Output("save-msg", "children"),
        Input("save-btn", "n_clicks"),
        State({"type": "low", "index": ALL}, "value"),
        State({"type": "high", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def save_config(n_clicks, lows, highs):
        intervals = [[l, h] for l, h in zip(lows, highs)
                     if l is not None and h is not None]
        return save_yaml_config(cfg, intervals)

    # -------------------------------------------------------------
    # Plot section
    # -------------------------------------------------------------
    @app.callback(
        Output("main-graph", "figure"),
        Input("data-dropdown", "value"),
        Input("vis-type", "value"),
        Input("update-btn", "n_clicks"),
    )
    def update_plot(selected_set, vis_type, _):
        y = np.array(y_dict[selected_set])
        if y.size == 0:
            return go.Figure().add_annotation(
                text="No data", x=0.5, y=0.5, showarrow=False
            )

        if vis_type == "scatter":
            fig = go.Figure(go.Scattergl(
                x=np.arange(len(y)), y=y,
                mode="markers",
                marker=dict(size=3, color=y, colorscale="Viridis", opacity=0.6),
            ))
            fig.update_layout(
                title=f"Scatter Plot — {selected_set.capitalize()} Set",
                xaxis_title="Sample Index", yaxis_title="Target Value",
                template="plotly_white"
            )
            return fig

        unique, counts = np.unique(y, return_counts=True)
        mean_count, std_dev = np.mean(counts), np.std(counts)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[str(u) for u in unique], y=counts,
                             marker_color="mediumseagreen"))
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(unique)-0.5,
            y0=mean_count, y1=mean_count,
            line=dict(color="red", dash="dash")
        )
        fig.update_layout(
            title=f"Class Distribution — {selected_set.capitalize()} Set",
            xaxis_title="Class", yaxis_title="Sample Count",
            template="plotly_white"
        )
        return fig

    # -------------------------------------------------------------
    # Stop server
    # -------------------------------------------------------------
    @app.callback(
        Output("stop-msg", "children"),
        Input("stop-server-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_server(n_clicks):
        func = flask.request.environ.get("werkzeug.server.shutdown")

        def kill_self():
            time.sleep(1.5)
            os.kill(os.getpid(), signal.SIGTERM)
        if func is None:
            threading.Thread(target=kill_self).start()
            return "🛑 Forcing shutdown..."
        def delayed_shutdown():
            time.sleep(1.5)
            func()
        threading.Thread(target=delayed_shutdown).start()
        return "🛑 Shutting down server..."

    # -------------------------------------------------------------
    # Run server
    # -------------------------------------------------------------
    port = 8062
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Visualizer Ready\n→ {url}\nPress CTRL+C to stop.\n")
    webbrowser.open(url)
    app.run(debug=True, port=port, use_reloader=False)





def trained_aimodel_visualizer(cfg):
    """
    Launch an interactive Dash app for visualizing a trained AI model,
    its training history, and performance metrics using test data.
    """

    # ------------------------------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------------------------------
    print("Loading training data via _get_training_data(cfg)...")
    Xtr, Xv, Xte, ytr, yv, yte, n_classes, class_means = _get_training_data(cfg)
    X_test, y_test = Xte, yte
    target_station = cfg.target_station

    # ------------------------------------------------------------------------------------------
    # Resolve model and history paths
    # ------------------------------------------------------------------------------------------
    model_path = os.path.join(cfg.model_dir, f"NowcastMdl_st{target_station}_1h.keras")
    hist_dir = os.path.join(cfg.model_dir, "histories")
    hist_path = os.path.join(hist_dir, f"NowcastMdl_st{target_station}_1h.pckl")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading history from: {hist_path}")
    with open(hist_path, "rb") as f:
        history = pickle.load(f)

    # ------------------------------------------------------------------------------------------
    # Evaluate model
    # ------------------------------------------------------------------------------------------
    print("Evaluating model…")
    y_prob = model.predict(X_test, batch_size=1024, verbose=0)
    y_pred_cls = y_prob.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred_cls)
    y_true_mm = np.array([class_means[c] for c in y_test])
    y_pred_mm = np.array([class_means[c] for c in y_pred_cls])
    rmse = math.sqrt(mean_squared_error(y_true_mm, y_pred_mm))

    cm = confusion_matrix(y_test, y_pred_cls, labels=range(n_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_percent = cm.astype(float) / row_sums * 100

    metrics = calc_metrics(y_test, y_pred_cls, class_means)
    print(f"\nStation {target_station} — Accuracy: {acc:.3f}, RMSE (mm): {rmse:.3f}")

    # ------------------------------------------------------------------------------------------
    # Build Dash App
    # ------------------------------------------------------------------------------------------
    app = dash.Dash(__name__)
    app.title = f"PyPrecip AI Model Visualizer — Station {target_station}"

    faded_style = {
        "font-size": "13px",
        "color": "rgba(80, 80, 80, 0.7)",
        "margin-top": "8px",
        "font-style": "italic"
    }

    # Order metrics logically if possible
    ordered_keys = ["ACC", "RMSE", "RSE", "POD", "FAR", "CSI"]
    metric_items = [(k, metrics[k]) for k in ordered_keys if k in metrics]

    app.layout = html.Div([
        html.H2(f"📊 PyPrecip AI Model Visualizer — Station {target_station}",
                style={'textAlign': 'center'}),

        # --- Performance Summary ---
        html.Div([
            html.H4("Performance Summary (on Test Data)"),
            html.Ul([html.Li(f"{k}: {v:.3f}") for k, v in metric_items]),
            html.Div([
                html.P("Definitions of performance metrics:",
                       style={"marginBottom": "4px", "font-weight": "bold"}),
                html.Ul([
                    html.Li("ACC — Accuracy: fraction of correctly predicted rainfall categories out of all test samples; measures overall classification success."),
                    html.Li("RMSE — Root Mean Square Error: magnitude of average deviation (mm) between predicted and observed mean rainfall; smaller values reflect higher precision."),
                    html.Li("RSE — Relative Squared Error: total squared prediction error normalized by the observed variance; lower values show better variance fitting and model efficiency."),
                    html.Li("POD — Probability of Detection: fraction of actual rainfall events successfully predicted (sensitivity)."),
                    html.Li("FAR — False Alarm Ratio: fraction of predicted rainfall events that did not occur (false alerts); low FAR means reliable detection."),
                    html.Li("CSI — Critical Success Index: unified event-based skill combining hits, misses, and false alarms; higher CSI indicates stronger overall event‑forecast ability.")
                ], style=faded_style)
            ])
        ], style={'marginBottom': 45, 'marginTop': 20}),

        # --- Accuracy and Loss History ---
        html.Div([
            html.Div([
                dcc.Graph(
                    id='accuracy-history',
                    figure=go.Figure([
                        go.Scatter(y=history['sparse_categorical_accuracy'], mode='lines', name='Train Accuracy'),
                        go.Scatter(y=history['val_sparse_categorical_accuracy'], mode='lines', name='Validation Accuracy')
                    ]).update_layout(title="Accuracy History", xaxis_title="Epoch", yaxis_title="Accuracy")
                ),
                html.Div(
                    "Tracks model accuracy throughout training. "
                    "Convergence of both training and validation curves indicates stable learning; "
                    "large separation signals overfitting or data imbalance.",
                    style=faded_style
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(
                    id='loss-history',
                    figure=go.Figure([
                        go.Scatter(y=history['loss'], mode='lines', name='Train Loss'),
                        go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss')
                    ]).update_layout(title="Loss History", xaxis_title="Epoch", yaxis_title="Loss")
                ),
                html.Div(
                    "Depicts error reduction during optimization. "
                    "Declining and converging curves denote effective model generalization.",
                    style=faded_style
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'marginBottom': 50}),

        html.Hr(),

        # --- Confusion Matrices ---
        html.Div([
            html.Div([
                html.H4("Confusion Matrix (Counts, Test Data)"),
                dcc.Graph(
                    id='cm-counts',
                    figure=px.imshow(
                        cm,
                        x=[f"Pred {i}" for i in range(n_classes)],
                        y=[f"True {i}" for i in range(n_classes)],
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix (Counts, Test Data)",
                        aspect="auto"
                    )
                ),
                html.Div(
                    "Displays number of test samples per actual vs predicted class. "
                    "Ideally, most predictions concentrate along the diagonal (correct classification).",
                    style=faded_style
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Confusion Matrix (% per True Class, Test Data)"),
                dcc.Graph(
                    id='cm-percent',
                    figure=px.imshow(
                        cm_percent,
                        x=[f"Pred {i}" for i in range(n_classes)],
                        y=[f"True {i}" for i in range(n_classes)],
                        text_auto=".0f",
                        color_continuous_scale="Blues",
                        title="Confusion Matrix (% per True Class, Test Data)",
                        aspect="auto"
                    )
                ),
                html.Div(
                    "Normalized representation of classification accuracy by true class for test data. "
                    "Diagonal percentages near 100 % indicate higher category‑specific skill.",
                    style=faded_style
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'marginBottom': 50, 'marginTop': 20}),

        html.Hr(),

        # --- Probability explorer ---
        html.H4("Prediction Probability Explorer (Test Data)"),
        dcc.Slider(0, len(y_test) - 50, 10, value=0, id='sample-slider',
                   marks=None, tooltip={"placement": "bottom"}),
        dcc.Graph(id='heatmap-preds'),
        html.Div(
            "Displays predicted probability distributions for subsets of the test dataset. "
            "Rows correspond to individual samples, columns to prediction classes. "
            "Brighter cells signal higher confidence; the green ✓ marks the true class.",
            style=faded_style
        ),

        html.Hr(),
        html.Button(
            "🛑 Stop Server",
            id="stop-server-btn",
            n_clicks=0,
            style={
                "font-size": "16px",
                "background-color": "#c0392b",
                "color": "white",
                "margin-top": "20px"
            }
        ),
        html.Div(id="stop-msg", style={"margin-top": "10px", "font-weight": "bold"})
    ])

    # ------------------------------------------------------------------------------------------
    # Callback to update subset heatmap dynamically
    # ------------------------------------------------------------------------------------------
    @app.callback(
        Output('heatmap-preds', 'figure'),
        Input('sample-slider', 'value')
    )
    def update_heatmap(start_idx):
        n_show = 30
        end_idx = min(start_idx + n_show, len(y_test))
        subset_probs = y_prob[start_idx:end_idx]

        fig = px.imshow(
            subset_probs,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Predicted Probability Distribution (Samples {start_idx}-{end_idx}, Test Data)"
        )

        for j, i in enumerate(range(start_idx, end_idx)):
            fig.add_annotation(
                x=y_test[i],
                y=j,
                text="✓",
                showarrow=False,
                font=dict(color="lime", size=12, family="Arial Black")
            )
        return fig

    # ------------------------------------------------------------------------------------------
    # Safe shutdown
    # ------------------------------------------------------------------------------------------
    @app.callback(
        Output("stop-msg", "children"),
        Input("stop-server-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def stop_server(n_clicks):
        func = flask.request.environ.get("werkzeug.server.shutdown")

        def kill_self():
            time.sleep(1.5)
            os.kill(os.getpid(), signal.SIGTERM)

        if func is None:
            threading.Thread(target=kill_self).start()
            return "🛑 Forcing shutdown (non‑Werkzeug environment)..."

        def delayed_shutdown():
            time.sleep(1.5)
            func()

        threading.Thread(target=delayed_shutdown).start()
        return "🛑 Shutting down server..."

    # ------------------------------------------------------------------------------------------
    # Launch Dash server
    # ------------------------------------------------------------------------------------------
    port = 8050
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Starting PyPrecip visualizer at {url}\nPress CTRL+C to stop manually.\n")
    webbrowser.open(url)
    app.run(debug=True, port=port, use_reloader=False)

