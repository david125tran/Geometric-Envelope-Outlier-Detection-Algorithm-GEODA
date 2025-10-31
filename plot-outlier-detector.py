# ---------------------------------- Libraries ----------------------------------
import ast
from collections import deque
import gc
from heapq import heappush, heappop
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio



# ---------------------------------- Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Color constants
BLACK = "#44444E"
GREY = "#AAAAAA"
BLUE = "#0046FF"
RED = "#FF8F8F"
GREEN = "#7ADAA5"



# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualiziation of what's going on 

    Ex.
    Input:  "Input"
    Output:
        *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        *                                           Loading Data                                            *
        *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    """
    banner_len = len(text)
    mid = 49 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 50)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 50)



# ---------------------------------- Load Data ----------------------------------
print_banner("Loading Data")

# --- Load CSV with various NA values handled ---
df = pd.read_csv(
    os.path.join(script_dir, "data.csv"),
    na_values=["#VALUE!", "#DIV/0!", "NA", "N/A", "", "null", "Null", "NULL"]
)

# --- Build a pressure_df DataFrame ---
pressure_df = pd.DataFrame({
    "sample_id": [c for c in df.columns if c != "x"],
    "pressure_trace": [pd.to_numeric(df[c], errors="coerce").tolist()
                       for c in df.columns if c != "x"],
    "times": [pd.to_numeric(df["x"], errors="coerce").tolist()] * (len(df.columns) - 1)
})

print(pressure_df.head())



# ---------------------------------- Functions ----------------------------------
def make_plots(pressure_df: pd.DataFrame, show_limits: bool, show_circles: bool) -> None:
    """
    This function creates interactive Plotly plots from the provided pressure_df DataFrame.
    It computes average lines, applies rolling limits to detect outliers, and styles
    the plots accordingly.
    Parameters:
    - pressure_df: DataFrame containing 'sample_id' and 'pressure_trace' columns
    - show_limits: Boolean flag to indicate whether to display upper and lower limit lines
    - show_circles: Boolean flag to indicate whether to display circles around average points   
    """
    # Set default renderer to browser
    pio.renderers.default = "browser"

    # Convert stringified lists into actual Python lists
    pressure_df["pressure_trace"] = pressure_df["pressure_trace"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # --- Combine all traces into a rectangular matrix with NaN fill ---
    traces = pressure_df["pressure_trace"].tolist()
    max_len = max(len(t) for t in traces)
    M = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        M[i, :len(t)] = t

    # --- Compute average line ignoring NaNs ---
    avg_line = np.nanmean(M, axis=0)   # or np.nanmedian(M, axis=0)
    x = np.arange(max_len)

    # --- Build tidy DataFrame for Plotly ---
    df_long = pd.DataFrame(M.T, columns=pressure_df["sample_id"])
    df_long["x"] = x
    df_melted = df_long.melt(id_vars="x", var_name="sample_id", value_name="pressure")

    # --- Add average line to the tidy frame ---
    avg_df = pd.DataFrame({"x": x, "pressure": avg_line, "sample_id": ["Average"] * len(x)})
    df_plot = pd.concat([df_melted, avg_df], ignore_index=True)

    # --- Add a flagged column (default False) ---
    df_plot["flagged"] = False

    # --- Apply rolling limits to flag outliers ---
    df_plot, circles = apply_rolling_limits(df_plot, width=25.0, interp_factor=10, dy=0.25, x_scale=1.0)

    # --- Flag curves that cross the limits ---
    # Merge back limit info per-x
    merged = df_plot.merge(
        df_plot[["x", "lower_limit", "upper_limit"]].drop_duplicates("x"),
        on="x", suffixes=("", "_lim")
    )

    # Mark if any point on that curve crosses the limits
    def crosses_limits(sub):
        if sub["lower_limit"].isna().all() or sub["upper_limit"].isna().all():
            return False
        return ((sub["pressure"] < sub["lower_limit"]) |
                (sub["pressure"] > sub["upper_limit"])).any()

    # Get set of sample_ids that crossed limits
    crossed_ids = set(
        merged.groupby("sample_id", group_keys=False).apply(crosses_limits)[lambda s: s].index
    )

    # Store result in df_plot
    df_plot["flagged"] = df_plot["sample_id"].isin(crossed_ids)

    # --- Create Plotly Express figure (all lines start grey) ---
    fig = px.line(
        df_plot,
        x="x",
        y="pressure",
        color="sample_id",
        line_group="sample_id",
        title="Plots with Outlier Detection",
        labels={"x": "X", "pressure": "Y (unit)"},
        hover_data={"flagged": True},
        color_discrete_sequence=[GREY]  # force all sample traces to grey
    )

    # Determine which sample_ids have any flagged points
    flagged_samples = set(df_plot.loc[df_plot["flagged"], "sample_id"].unique())

    def _style_trace(tr):
        if tr.name == "Average Radius (per-point)":
            return
        if tr.name == "Average":
            tr.update(line=dict(width=2.0, color=BLUE), opacity=1.0)
        elif (tr.name in flagged_samples) and (show_limits == True):
            tr.update(line=dict(width=2.0, color=RED), opacity=0.65)
        elif (tr.name in flagged_samples):
            tr.update(line=dict(width=1.0, color=GREY), opacity=0.65)
        else:
            tr.update(line=dict(width=1.0, color=GREY), opacity=0.65)

    # Apply styling to each trace
    fig.for_each_trace(_style_trace)

    # --- Add upper & lower limit lines (distinct colors) ---
    # Collapse to unique (x, limit) so we draw clean lines
    upper_line = (df_plot[["x", "upper_limit"]]
                  .dropna()
                  .drop_duplicates(subset=["x"])
                  .sort_values("x"))
    lower_line = (df_plot[["x", "lower_limit"]]
                  .dropna()
                  .drop_duplicates(subset=["x"])
                  .sort_values("x"))

    if (show_limits == True):
        # --- Add limit lines to the figure ---
        if not upper_line.empty:
            fig.add_scatter(
                x=upper_line["x"], y=upper_line["upper_limit"],
                mode="lines",
                line=dict(color=GREEN, dash="dash", width=2),  # green dashed
                name="Upper Limit"
            )

        # --- Add limit lines to the figure ---
        if not lower_line.empty:
            fig.add_scatter(
                x=lower_line["x"], y=lower_line["lower_limit"],
                mode="lines",
                line=dict(color=GREEN, dash="dash", width=2),  # red dashed
                name="Lower Limit"
            )

    # --- Final layout adjustments ---
    fig.update_layout(
        xaxis_title="Time (Index)",
        yaxis_title="Pressure",
        plot_bgcolor="white",
        template="plotly_white",
        legend=dict(title=None)
    )

    # --- Add circles around Average points if show_circles==True ---
    if show_circles == True:
        # --- Add the circles around each Average point ---
        # Flatten all circles with None separators so Plotly draws disjoint loops
        if circles:
            xs, ys = [], []
            for c in circles:
                xs.extend(c["x"].tolist())
                ys.extend(c["y"].tolist())
                xs.append(None)  # separator between circles
                ys.append(None)

            fig.add_scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=RED, width=1),
                opacity=0.35,
                name="Average Radius (per-point)"
            )

    fig.show()

    # Delete circles to free memory
    del circles
    gc.collect()



def apply_rolling_limits(
    plot_df: pd.DataFrame,
    width: float,
    interp_factor,
    *,
    dy: float = 0.25,
    x_scale: float = 1.0
    ) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    """
    This function computes rolling upper and lower limits around the "Average" series
    in the provided plot_df DataFrame. It generates circles of a specified width
    around each point of the Average series and calculates the min and max y-values
    at each integer x-coordinate to define the limits.

    Parameters:
    - plot_df: DataFrame containing 'x', 'pressure', and 'sample_id'
    - width: Diameter of the circles to be drawn around each Average point
    - interp_factor: Number of interpolated circle centers between each pair of Average points
    - dy: Vertical scaling factor for the circles (default is 0.25)
    - x_scale: Horizontal scaling factor for the circles (default is 1.0)   
    """
    # --- Pull the Average series ---
    avg = (plot_df.loc[plot_df["sample_id"] == "Average", ["x", "pressure"]]
                    .sort_values("x").reset_index(drop=True))
    if avg.empty:
        raise ValueError("apply_rolling_limits: 'Average' series not found in plot_df.")
    x_avg = avg["x"].to_numpy(dtype=float)
    y_avg = avg["pressure"].to_numpy(dtype=float)

    # --- Ensure required columns exist up front ---
    plot_df = plot_df.copy()
    if "upper_limit" not in plot_df.columns:
        plot_df["upper_limit"] = np.nan
    if "lower_limit" not in plot_df.columns:
        plot_df["lower_limit"] = np.nan
    if "flagged" not in plot_df.columns:
        plot_df["flagged"] = False
    else:
        plot_df["flagged"] = False

    # --- Generate circles (use your current settings) ---
    circles_list: list[dict[str, np.ndarray]] = []
    theta = np.linspace(0.0, 2*np.pi, 60, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # visual x scaling so circles look round
    x_scale_vis = 0.25
    r = float(width)

    # densify: set how many extra centers between each avg pair (0 = none)
    interp_factor = 5  # adjust to taste

    for i in range(len(x_avg) - 1):
        x0, y0 = x_avg[i], y_avg[i]
        x1, y1 = x_avg[i + 1], y_avg[i + 1]
        if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1):
            continue

        for t in np.linspace(0.0, 1.0, interp_factor + 1, endpoint=False):
            xi = (1 - t) * x0 + t * x1
            yi = (1 - t) * y0 + t * y1
            x_circle = xi + r * x_scale_vis * cos_t
            y_circle = yi + r * sin_t
            circles_list.append({"x": x_circle, "y": y_circle})

    # final endpoint circle
    x_circle = x_avg[-1] + r * x_scale_vis * cos_t
    y_circle = y_avg[-1] + r * sin_t
    circles_list.append({"x": x_circle, "y": y_circle})

    # --- Compute min/max per integer x from all perimeter points ---
    xs_all, ys_all = [], []
    for c in circles_list:
        xs = np.asarray(c["x"])
        ys = np.asarray(c["y"])
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.any():
            xs_all.append(xs[m])
            ys_all.append(ys[m])

    x_min = int(np.nanmin(plot_df["x"]))
    x_max = int(np.nanmax(plot_df["x"]))
    full_x = pd.DataFrame({"x": np.arange(x_min, x_max + 1, dtype=int)})

    if xs_all:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)

        ix = np.rint(xs_all).astype(int)  # nearest integer x
        in_range = (ix >= x_min) & (ix <= x_max)
        ix = ix[in_range]
        ys_all = ys_all[in_range]

        if ix.size > 0:
            limits = (pd.DataFrame({"x": ix, "y": ys_all})
                        .groupby("x")["y"]
                        .agg(lower_limit="min", upper_limit="max")
                        .reset_index())
            # join to full_x and interpolate to fill any gaps
            limits = (full_x.merge(limits, on="x", how="left")
                             .interpolate("linear", limit_direction="both"))
            plot_df = plot_df.drop(columns=[c for c in ["upper_limit","lower_limit"] if c in plot_df.columns])
            plot_df = plot_df.merge(limits, on="x", how="left")
        else:
            # no perimeter points in range → keep NaNs but ensure columns exist
            plot_df = plot_df.merge(full_x.assign(lower_limit=np.nan, upper_limit=np.nan), on="x", how="left", suffixes=(None, None))
    else:
        # defensive: no circles produced → keep NaNs but ensure columns exist
        plot_df = plot_df.merge(full_x.assign(lower_limit=np.nan, upper_limit=np.nan), on="x", how="left", suffixes=(None, None))

    return plot_df, circles_list



# ---------------------------------- Make Plots ----------------------------------
print_banner("Making Plots")
make_plots(pressure_df, show_limits=True, show_circles=False)

print_banner("Plots Generated!")