import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import plotly.graph_objects as go
from core.viz import (
    suggest_chart_type, bar_chart, line_chart, scatter_chart,
    histogram, box_plot, heatmap,
)
from core.stats import correlation_matrix


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "Jan", "Feb"],
        "sales": [100, 150, 130, 170, 120, 160],
        "region": ["East", "East", "East", "East", "West", "West"],
        "profit": [20, 35, 25, 40, 22, 38],
    })


# ---------- suggest_chart_type ----------

def test_suggest_chart_type_single_numeric(sample_df):
    assert suggest_chart_type(sample_df, "sales") == "histogram"


def test_suggest_chart_type_single_categorical(sample_df):
    assert suggest_chart_type(sample_df, "region") == "bar"


def test_suggest_chart_type_numeric_vs_numeric(sample_df):
    assert suggest_chart_type(sample_df, "sales", "profit") == "scatter"


def test_suggest_chart_type_categorical_vs_numeric(sample_df):
    assert suggest_chart_type(sample_df, "region", "sales") == "box"


def test_suggest_chart_type_categorical_vs_categorical(sample_df):
    assert suggest_chart_type(sample_df, "region", "month") == "bar"


def test_suggest_chart_type_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        suggest_chart_type(sample_df, "not_a_column")


# ---------- chart builders ----------

def test_bar_chart_value_counts_mode(sample_df):
    fig = bar_chart(sample_df, "region")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_bar_chart_xy_mode(sample_df):
    fig = bar_chart(sample_df, "month", "sales", color="region")
    assert isinstance(fig, go.Figure)


def test_bar_chart_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        bar_chart(sample_df, "not_a_column")


def test_line_chart(sample_df):
    fig = line_chart(sample_df, "month", "sales", color="region")
    assert isinstance(fig, go.Figure)


def test_scatter_chart(sample_df):
    fig = scatter_chart(sample_df, "sales", "profit", color="region")
    assert isinstance(fig, go.Figure)


def test_scatter_chart_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        scatter_chart(sample_df, "sales", "not_a_column")


def test_histogram(sample_df):
    fig = histogram(sample_df, "sales", bins=10)
    assert isinstance(fig, go.Figure)


def test_box_plot(sample_df):
    fig = box_plot(sample_df, "region", "sales")
    assert isinstance(fig, go.Figure)


def test_heatmap_with_correlation_matrix(sample_df):
    """Cross-module check: heatmap() should render whatever correlation_matrix() produces."""
    corr = correlation_matrix(sample_df[["sales", "profit"]])
    fig = heatmap(corr)
    assert isinstance(fig, go.Figure)


# ---------- 3D charts (Phase 5) ----------

def test_scatter_3d(sample_df):
    from core.viz import scatter_3d
    fig = scatter_3d(sample_df, "sales", "profit", "profit", color="region")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_scatter_3d_non_numeric_raises(sample_df):
    from core.viz import scatter_3d
    with pytest.raises(ValueError):
        scatter_3d(sample_df, "sales", "region", "profit")


def test_scatter_3d_unknown_column_raises(sample_df):
    from core.viz import scatter_3d
    with pytest.raises(ValueError):
        scatter_3d(sample_df, "sales", "profit", "not_a_column")


def test_surface_3d_with_grid_data():
    import numpy as np
    from core.viz import surface_3d
    x_vals = np.linspace(0, 10, 4)
    y_vals = np.linspace(0, 10, 4)
    rows = [{"x": xv, "y": yv, "z": xv + yv} for xv in x_vals for yv in y_vals]
    grid_df = pd.DataFrame(rows)
    fig = surface_3d(grid_df, "x", "y", "z")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "surface"


def test_surface_3d_non_numeric_raises(sample_df):
    from core.viz import surface_3d
    with pytest.raises(ValueError):
        surface_3d(sample_df, "region", "profit", "sales")
