import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load and preprocess data
data = pd.read_csv("D:/Downloads/K-means-Clustering-for-customer-segmentations-main/K-means-Clustering-for-customer-segmentations-main/R implementations/Mall_Customers.csv")
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([html.H1("Customer Segmentation Dashboard", className="text-center mb-4")]),
    
    # Control panel for clustering parameters
    dbc.Row([
        dbc.Col([
            html.Label("Select Clustering Algorithm"),
            dcc.Dropdown(
                id="algorithm-dropdown",
                options=[
                    {"label": "KMeans", "value": "KMeans"},
                    {"label": "Agglomerative Clustering", "value": "Agglomerative"},
                    {"label": "DBSCAN", "value": "DBSCAN"}
                ],
                value="KMeans",
                clearable=False
            ),
            html.Br(),
            html.Label("Number of Clusters (KMeans & Agglomerative)"),
            dcc.Slider(id="num-clusters", min=2, max=10, step=1, value=5),
            html.Br(),
            html.Label("DBSCAN Epsilon (Radius)"),
            dcc.Slider(id="dbscan-epsilon", min=0.1, max=1.0, step=0.1, value=0.5),
            html.Br(),
            html.Label("DBSCAN Minimum Samples"),
            dcc.Slider(id="dbscan-min-samples", min=2, max=10, step=1, value=5),
        ], width=3),
        
        # Visuals display panel
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id="clustering-graph"), width=6),
                dbc.Col(dcc.Graph(id="clustering-graph-3d"), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="metric-comparison"), width=6),
                dbc.Col([
                    html.H5("Metrics Summary"),
                    html.Div(id="cluster-details"),
                    html.Br(),
                    html.H5("Algorithm Justification & Recommendations"),
                    html.Div(id="algorithm-justification"),
                    html.Br(),
                    html.H5("Cluster Centroids / Distribution"),
                    html.Div(id="cluster-centroids"),
                ], width=6),
            ]),
        ], width=9),
    ]),
])

# Callback to update graph, cluster details, justification, and metrics
@app.callback(
    Output("clustering-graph", "figure"),
    Output("clustering-graph-3d", "figure"),
    Output("metric-comparison", "figure"),
    Output("cluster-details", "children"),
    Output("algorithm-justification", "children"),
    Output("cluster-centroids", "children"),
    Input("algorithm-dropdown", "value"),
    Input("num-clusters", "value"),
    Input("dbscan-epsilon", "value"),
    Input("dbscan-min-samples", "value")
)
def update_graph_and_metrics(algorithm, num_clusters, dbscan_epsilon, dbscan_min_samples):
    # Run clustering
    if algorithm == "KMeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        title = f"KMeans Clustering (K={num_clusters})"
        centroids = model.cluster_centers_
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=num_clusters)
        labels = model.fit_predict(X_scaled)
        title = f"Agglomerative Clustering (K={num_clusters})"
        centroids = None  # No centroids for Agglomerative
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=dbscan_epsilon, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X_scaled)
        title = f"DBSCAN Clustering (ε={dbscan_epsilon}, min_samples={dbscan_min_samples})"
        centroids = None  # No centroids for DBSCAN

    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 'N/A'
    davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 'N/A'
    inertia = model.inertia_ if algorithm == "KMeans" else 'N/A'

    # 2D scatter plot
    fig_2d = px.scatter(data, x="Annual Income (k$)", y="Spending Score (1-100)", color=labels,
                        title=title, color_continuous_scale="Viridis")
    
    # 3D scatter plot
    fig_3d = px.scatter_3d(data, x="Annual Income (k$)", y="Spending Score (1-100)", z="Age", color=labels,
                           title=f"{title} (3D)", color_continuous_scale="Viridis")

    # Comparison graph
    metric_comparison_fig = px.bar(
        x=["Silhouette Score", "Davies-Bouldin Index", "Inertia (KMeans only)"],
        y=[silhouette, davies_bouldin, inertia],
        labels={'x': "Metric", 'y': "Value"},
        title="Clustering Quality Metrics"
    )

    # Cluster centroids or distribution
    cluster_centroid_info = ""
    if centroids is not None:
        centroid_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=["Annual Income (k$)", "Spending Score (1-100)"])
        cluster_centroid_info = dbc.Table.from_dataframe(centroid_df, striped=True, bordered=True, hover=True)
    else:
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_centroid_info = html.P(f"Cluster Counts: {cluster_counts.to_dict()}")

    # Algorithm justification text
    if algorithm == "KMeans":
        justification = (
            "KMeans clustering minimizes within-cluster variance and is fast, especially for spherical clusters. "
            "Recommended for large datasets with well-separated clusters. Adjust K to improve results."
        )
    elif algorithm == "Agglomerative":
        justification = (
            "Agglomerative clustering can form arbitrarily shaped clusters and is ideal for hierarchical data. "
            "However, it is slower for large datasets and sensitive to the choice of distance metrics."
        )
    elif algorithm == "DBSCAN":
        justification = (
            "DBSCAN is robust for clusters of varying densities and noisy data. "
            "Good for non-spherical clusters and identifying outliers. Sensitive to ε and min_samples parameters."
        )

    # Display metrics
    metrics_info = [
        html.P(f"Silhouette Score: {silhouette}"),
        html.P(f"Davies-Bouldin Index: {davies_bouldin}"),
        html.P(f"Inertia: {inertia} (for KMeans)")
    ]

    return fig_2d, fig_3d, metric_comparison_fig, metrics_info, justification, cluster_centroid_info

if __name__ == "__main__":
    app.run_server(debug=True)
