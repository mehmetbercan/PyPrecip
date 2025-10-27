from pathlib import Path
import json
import yaml
import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import plotly.graph_objects as go
import pandas as pd
import webbrowser
import flask
import threading
import time
import os
import signal
import missingno as msno
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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