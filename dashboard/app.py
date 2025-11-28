import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# === Cargar predicciones ===
df = pd.read_csv("../data/processed/model_predictions.csv")

# === App con Bootstrap ===
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

# === Layout Moderno ===
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Rocket League ML Dashboard", className="fw-bold text-light"),
        ]),
        color="#0d6efd", dark=True, className="mb-4 shadow-sm"
    ),

    dbc.Row([
        dbc.Col([
            html.Label("Filtrar por modo de juego", className="fw-bold"),
            dcc.Dropdown(
                options=[{"label": m, "value": m} for m in sorted(df["game_mode"].unique())],
                id="mode_filter",
                placeholder="Selecciona un modo...",
                className="mb-4"
            )
        ], width=4)
    ], justify="center"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Predicciones del Modelo", className="fw-bold"),
            dbc.CardBody([dcc.Graph(id="pie_graph")])
        ], className="shadow-sm mb-4"), width=6),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Real vs Predicho", className="fw-bold"),
            dbc.CardBody([dcc.Graph(id="compare_graph")])
        ], className="shadow-sm mb-4"), width=6),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Distribución de Diferencia de Goles", className="fw-bold"),
            dbc.CardBody([dcc.Graph(id="goal_diff_graph")])
        ], className="shadow-sm mb-5"), width=12)
    ])
], fluid=True)

# === Callbacks ===
@app.callback(
    [Output("pie_graph", "figure"),
     Output("compare_graph", "figure"),
     Output("goal_diff_graph", "figure")],
    [Input("mode_filter", "value")]
)
def update_graphs(selected_mode):
    filtered = df.copy()
    if selected_mode:
        filtered = filtered[filtered["game_mode"] == selected_mode]

    pred_counts = filtered["predicted_winner"].value_counts().reset_index()
    pred_counts.columns = ["winner", "count"]
    pie_fig = px.pie(pred_counts, values="count", names="winner", title="Porcentaje de victorias predichas")

    compare = filtered.groupby(["winner", "predicted_winner"]).size().reset_index(name="count")
    compare_fig = px.bar(compare, x="winner", y="count", color="predicted_winner",
                         barmode="group", title="Ganador real vs ganador predicho")

    goal_diff_fig = px.histogram(filtered, x="goal_difference", nbins=15,
                                 title="Distribución de diferencia de goles")

    return pie_fig, compare_fig, goal_diff_fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)