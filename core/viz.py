"""
viz.py — Phases 4 & 5
2D (Phase 4) and 3D (Phase 5) chart builders using Plotly.

Every function returns a plotly Figure and takes NO Streamlit objects as
input — the app/ layer calls st.plotly_chart(fig). This keeps charting
logic reusable outside Streamlit too (e.g. exporting to standalone HTML).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def suggest_chart_type(df: pd.DataFrame, col_x: str, col_y: str | None = None) -> str:
    """
    Suggests a sensible chart type based on the dtypes of the given column(s).
    Returns one of: "histogram", "bar", "scatter", "box", "line".
    """
    if col_x not in df.columns:
        raise ValueError(f"Column '{col_x}' not found in dataframe")

    x_numeric = pd.api.types.is_numeric_dtype(df[col_x])

    if col_y is None:
        # Single column: numeric -> distribution histogram, categorical -> count bar
        return "histogram" if x_numeric else "bar"

    if col_y not in df.columns:
        raise ValueError(f"Column '{col_y}' not found in dataframe")
    y_numeric = pd.api.types.is_numeric_dtype(df[col_y])

    if x_numeric and y_numeric:
        return "scatter"
    elif not x_numeric and y_numeric:
        return "box"       # categorical x, numeric y -> distribution per category
    elif x_numeric and not y_numeric:
        return "box"       # numeric x, categorical y -> same idea, flipped
    else:
        return "bar"        # both categorical -> grouped count bar


def bar_chart(df: pd.DataFrame, x: str, y: str | None = None, color: str | None = None) -> go.Figure:
    """
    If y is None: shows value counts of x (how many rows per category).
    If y is given: shows x vs y directly as bars.
    """
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in dataframe")

    if y is None:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, "count"]
        fig = px.bar(counts, x=x, y="count", title=f"Count of {x}")
    else:
        if y not in df.columns:
            raise ValueError(f"Column '{y}' not found in dataframe")
        fig = px.bar(df, x=x, y=y, color=color, title=f"{y} by {x}")

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    fig = px.line(df, x=x, y=y, color=color, title=f"{y} over {x}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def scatter_chart(
    df: pd.DataFrame, x: str, y: str, color: str | None = None, size: str | None = None
) -> go.Figure:
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=f"{y} vs {x}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def box_plot(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """x = categorical grouping column, y = numeric value column (order can be flipped by caller)."""
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    fig = px.box(df, x=x, y=y, title=f"{y} distribution by {x}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Takes a correlation matrix (from core.stats.correlation_matrix) and renders it."""
    fig = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation heatmap",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def scatter_3d(
    df: pd.DataFrame, x: str, y: str, z: str, color: str | None = None, size: str | None = None
) -> go.Figure:
    """3D scatter plot — each row becomes one point in 3D space."""
    for col in (x, y, z):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"3D scatter requires numeric columns; '{col}' is not numeric")

    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size, title=f"{z} by {x} and {y}")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def surface_3d(df: pd.DataFrame, x: str, y: str, z: str) -> go.Figure:
    """
    3D surface plot. Requires x and y to form a grid (e.g. repeated
    coordinates like latitude/longitude, or two parameters swept over a
    range) — z values are averaged wherever (x, y) combinations repeat.
    If your data doesn't naturally form a grid, use scatter_3d instead.
    """
    for col in (x, y, z):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"3D surface requires numeric columns; '{col}' is not numeric")

    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    if pivot.empty:
        raise ValueError("Could not build a grid from the given x/y columns — no valid data points")

    fig = go.Figure(data=[go.Surface(x=pivot.columns, y=pivot.index, z=pivot.values)])
    fig.update_layout(
        title=f"{z} surface over {x} and {y}",
        scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

