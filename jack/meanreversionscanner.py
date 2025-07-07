#!/usr/bin/env python3

# Import necessary libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller  # For ADF stationarity test
from dash import Dash, html, dcc  # Web framework and HTML/graph components
from dash.dependencies import Output, Input  # Callback decorators
import plotly.graph_objs as go  # For building interactive graphs

# -------------------------------------------------------------------------
# Function to calculate Hurst Exponent (mean-reversion vs trend-following)
# -------------------------------------------------------------------------
def compute_hurst(ts):
    lags = range(2, 100)
    tau = []

    for lag in lags:
        if lag >= len(ts):
            break
        try:
            diff = ts[lag:] - ts[:-lag]
            std = np.std(diff)
            tau.append(std if std > 0 else np.nan)
        except:
            tau.append(np.nan)

    tau = np.array(tau)
    valid = ~np.isnan(tau) & (tau > 0)

    if np.sum(valid) < 10:
        return np.nan  # Not enough valid points

    log_lags = np.log(np.array(lags)[valid])
    log_tau = np.log(tau[valid])

    try:
        slope, _ = np.polyfit(log_lags, log_tau, 1)
    except:
        return np.nan

    return slope  # This slope is the Hurst exponent

# -------------------------------------------------------------------------
# Analyze all 50 assets and compute metrics
# -------------------------------------------------------------------------
def analyze_mean_reversion(filepath):
    # Load price data (rows = assets, columns = days)
    prices = pd.read_csv(filepath, sep=r'\s+', header=None)

    # Handle transpose if necessary
    if prices.shape[0] != 50 and prices.shape[1] == 50:
        prices = prices.T

    results = []

    # Loop through all 50 instruments
    for inst in range(prices.shape[0]):
        series = prices.iloc[inst].values  # Price series for this asset
        logp = np.log(series + 1e-8)       # Log-transform prices to stabilize variance

        # ADF test (stationarity check)
        try:
            adf_p = adfuller(logp)[1]  # We use just the p-value
        except:
            adf_p = np.nan

        # Hurst exponent
        try:
            hurst = compute_hurst(series)
        except:
            hurst = np.nan

        # Half-life of mean reversion (Ornstein-Uhlenbeck style)
        try:
            lagged = np.roll(logp, 1)       # Lag by 1
            lagged[0] = 0                   # Patch first value
            delta = logp - lagged           # Daily returns (log space)
            beta = np.polyfit(lagged[1:], delta[1:], 1)[0]  # Linear regression slope
            half_life = -np.log(2) / beta if beta != 0 else np.nan
        except:
            half_life = np.nan

        results.append((inst, adf_p, hurst, half_life))

    # Convert to DataFrame for easier sorting/viewing
    df = pd.DataFrame(results, columns=['inst', 'adf_p', 'hurst', 'half_life'])

    # Sort by ADF p-value (lower = more likely mean reverting)
    df.sort_values('adf_p', inplace=True)

    # Print entire table in terminal
    print("\nAll Mean-Reversion Metrics:\n", df.to_string(index=False))

    # Save to CSV
    df.to_csv("mean_reversion_metrics.csv", index=False)

    return df, prices  # Return metrics and raw price data

# -------------------------------------------------------------------------
# Build and launch Dash dashboard for visual inspection
# -------------------------------------------------------------------------
def launch_dashboard(df, prices):
    app = Dash(__name__)
    app.title = "Mean Reversion Viewer"

    # Layout of the webpage
    app.layout = html.Div([
        html.H1("Mean Reversion Time Series Viewer", style={"textAlign": "center"}),

        # Tabs: one for each asset
        dcc.Tabs(id="tabs", value='0', children=[
            dcc.Tab(label=f"Asset {int(row.inst)}", value=str(int(row.inst)))
            for _, row in df.iterrows()
        ]),

        # Graph content placeholder
        html.Div(id='tab-content')
    ])

    # Callback: updates the graph when a tab is selected
    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab(selected_inst):
        inst = int(selected_inst)
        row = df[df['inst'] == inst].iloc[0]
        ts = prices.iloc[inst].values
        logp = np.log(ts + 1e-8)

        # Mean reversion level (in price terms, not log)
        mean_log_price = np.mean(logp)
        mean_level = np.exp(mean_log_price)

        fig = go.Figure()

        # Plot the price series
        fig.add_trace(go.Scatter(y=ts, mode="lines", name=f"Asset {inst} Price"))

        # Add dashed red line = mean level
        fig.add_trace(go.Scatter(
            x=np.arange(len(ts)),
            y=[mean_level] * len(ts),
            mode="lines",
            name="Mean Reversion Level",
            line=dict(color='red', dash='dash')
        ))

        # Update graph styling
        fig.update_layout(
            title=f"Asset {inst} | ADF: {row.adf_p:.4f} | Hurst: {row.hurst:.4f} | HL: {row.half_life:.1f} | Mean≈{mean_level:.2f}",
            xaxis_title="Day",
            yaxis_title="Price",
            height=500
        )
        return dcc.Graph(figure=fig)

    # Start the server on http://127.0.0.1:8050/
    app.run(debug=True)

# -------------------------------------------------------------------------
# Main entry point of script
# -------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = "prices.txt"                         # Input data file
    df_metrics, prices_df = analyze_mean_reversion(filepath)  # Get scores
    launch_dashboard(df_metrics, prices_df)         # Start local dashboard

# Visit: http://127.0.0.1:8050/ to view plots interactively
