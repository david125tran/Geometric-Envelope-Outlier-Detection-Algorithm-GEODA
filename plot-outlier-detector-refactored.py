# ---------------------------------- Libraries ----------------------------------
import ast
import gc
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
BLUE = "#0BA6DF"
RED = "#FF8F8F"
GREEN = "#7ADAA5"
PURPLE = "#9B5DE0"


# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualization of what's going on
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


def make_plots(pressure_df: pd.DataFrame, 
               enable_algorithm1: bool,
               ) -> None:
    """
    Creates interactive Plotly plots from pressure_df.
    Computes the Average, analytic scanline limits (no circle sampling),
    and styles/flags traces that cross the limits.
    """
    print_banner("Making Plots")

    # Set default renderer to browser
    pio.renderers.default = "browser"

    # Convert stringified lists into actual Python lists
    pressure_df["pressure_trace"] = pressure_df["pressure_trace"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Extract all the pressure traces (lists of numbers) from the pressure_df DataFrame into a 
    # plain Python list of lists, which makes it easy to stack them into a NumPy array later.
    traces = pressure_df["pressure_trace"].tolist()

    # Build a 2D NumPy array with NaN padding for unequal lengths.  This does not remove NaNs
    max_len = max(len(t) for t in traces)
    M = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        M[i, :len(t)] = t

    # Compute average line ignoring NaNs 
    avg_line = np.nanmean(M, axis=0)   # or np.nanmedian(M, axis=0)
    x = np.arange(max_len)

    # Build tidy DataFrame for Plotly Express
    df_long = pd.DataFrame(M.T, columns=pressure_df["sample_id"])
    df_long["x"] = x

    # sample_id  curve_01  curve_02  curve_03  curve_04  curve_05  curve_06  curve_07  curve_08  curve_09  curve_10  curve_11  x
    # 0          5.482199  3.990653  4.321725  4.811966  5.092594  6.235918  5.185523  5.464888  5.227666  5.902978  4.988769  0
    # 1          6.013360  5.294809  6.082067  5.245110  4.787759  4.166610  5.500618  4.607558  4.906942  5.164358  1.735080  1
    # 2          4.965411  4.170009  4.994575  4.562440  6.219304  5.483014  6.287566  5.649635  4.813290  4.942694  2.353699  2
    # 3          4.972236  5.998754  5.045114  4.797331  4.990661  5.514891  4.893120  5.621073  3.944563  5.965971  2.555585  3
    # 4          6.067569  3.704943  6.000538  4.926091  6.290930  4.760358  5.386169  5.850606  4.455401  4.894813  3.137076  4

    # Melt to tidy format
    df_melted = df_long.melt(id_vars="x", var_name="sample_id", value_name="pressure")

    #    x sample_id  pressure
    # 0  0  curve_01  5.482199
    # 1  1  curve_01  6.013360
    # 2  2  curve_01  4.965411
    # 3  3  curve_01  4.972236
    # 4  4  curve_01  6.067569

    # Add average line to the tidy frame
    avg_df = pd.DataFrame({"x": x, "pressure": avg_line, "sample_id": ["Average"] * len(x)})
    df_plot = pd.concat([df_melted, avg_df], ignore_index=True)

    # Add a flagged column (default False) to indicate outliers
    df_plot["flagged"] = False

    # Create Plotly Express figure (all lines start grey)
    fig = px.line(
        df_plot,
        x="x",
        y="pressure",
        color="sample_id",
        line_group="sample_id",
        title="Plots with Outlier Detection (Analytic Limits)",
        labels={"x": "Time (Index)", "pressure": "Pressure"},
        hover_data={"flagged": True},
        color_discrete_sequence=[GREY]  # force all sample traces to grey
    )

    # ---------------------------------- Algorithm #1: apply_rolling_limits() ----------------------------------
    if (enable_algorithm1 == True):
        # Apply analytic scanline limits (fast, no circle sampling)
        df_plot = apply_rolling_limits(
            df_plot,
            width=35.0,
            interp_factor=10,
            gradient_radius=False,
            dy=0.25,
            x_scale=1.0
        )

        # Flag curves that cross the limits
        merged = df_plot.merge(
            df_plot[["x", "lower_limit", "upper_limit"]].drop_duplicates("x"),
            on="x", suffixes=("", "_lim")
        )

        def crosses_limits(sub):
            if sub["lower_limit"].isna().all() or sub["upper_limit"].isna().all():
                return False
            return ((sub["pressure"] < sub["lower_limit"]) |
                    (sub["pressure"] > sub["upper_limit"])).any()

        crossed_ids = set(
            merged.groupby("sample_id", group_keys=False).apply(crosses_limits)[lambda s: s].index
        )
        df_plot["flagged"] = df_plot["sample_id"].isin(crossed_ids)

    # Determine which sample_ids have any flagged points
    flagged_samples = set(df_plot.loc[df_plot["flagged"], "sample_id"].unique())

    # Style traces based on their status
    def _style_trace(tr):
        if tr.name == "Average Radius (per-point)":
            return
        if tr.name == "Air Limit":
                tr.update(line=dict(width=4.0, color=PURPLE, dash="dash"), opacity=1.0)
                return
        # Average line is blue and thicker
        if tr.name == "Average":
            tr.update(line=dict(width=3.0, color=BLUE), opacity=1.0)
        # Flagged samples are red and thicker
        elif (tr.name in flagged_samples) and (enable_algorithm1 == True):
            tr.update(line=dict(width=3.0, color=RED), opacity=1.0)
        # Non-flagged samples are grey and thinner
        elif (tr.name in flagged_samples):
            tr.update(line=dict(width=2.0, color=GREY), opacity=0.9)
        # Others 
        else:
            tr.update(line=dict(width=2.0, color=GREY), opacity=0.9)

    fig.for_each_trace(_style_trace)

    if (enable_algorithm1 == True):
        # Add upper & lower limit lines (distinct colors) 
        upper_line = (df_plot[["x", "upper_limit"]]
                    .dropna()
                    .drop_duplicates(subset=["x"])
                    .sort_values("x"))
        lower_line = (df_plot[["x", "lower_limit"]]
                    .dropna()
                    .drop_duplicates(subset=["x"])
                    .sort_values("x"))

        if not upper_line.empty:
            fig.add_scatter(
                x=upper_line["x"], y=upper_line["upper_limit"],
                mode="lines",
                line=dict(color=GREEN, dash="dash", width=5),
                name="Upper Limit"
            )
        if not lower_line.empty:
            fig.add_scatter(
                x=lower_line["x"], y=lower_line["lower_limit"],
                mode="lines",
                line=dict(color=GREEN, dash="dash", width=5),
                name="Lower Limit"
            )

    # Final layout adjustments
    fig.update_layout(
        plot_bgcolor="white",
        template="plotly_white",
        legend=dict(title=None)
    )

    fig.show()




# ---------------------------------- Algorithm: apply_rolling_limits() ----------------------------------
def apply_rolling_limits(
    plot_df: pd.DataFrame,
    width: float,
    interp_factor: int,
    gradient_radius: bool,
    *,
    dy: float = 0.25,    # kept for signature compatibility (unused here)
    x_scale: float = 1.0 # kept for signature compatibility (we use a fixed visual ax below)
    ) -> tuple[pd.DataFrame, list[dict]]:
    """
    Analytic scanline computation of rolling upper/lower limits around the "Average" series.

    We treat the envelope as the Minkowski sum of the Average polyline with a disk of
    radius `width` (optionally varying with local slope if gradient_radius=True).
    Instead of sampling circle perimeters, we directly compute, for each integer x,
    the vertical extent contributed by each (interpolated) disk center.

    Returns:
      plot_df with 'lower_limit' and 'upper_limit' columns merged per x,
      and an empty 'circles' list (for API compatibility with previous code).
    """
    print_banner("Applying Algorithm: apply_rolling_limits()")

    # Pull the Average series
    avg = (plot_df.loc[plot_df["sample_id"] == "Average", ["x", "pressure"]]
                    .sort_values("x").reset_index(drop=True))
    if avg.empty:
        raise ValueError("apply_rolling_limits: 'Average' series not found in plot_df.")
    x_avg = avg["x"].to_numpy(dtype=float)
    y_avg = avg["pressure"].to_numpy(dtype=float)

    # Ensure required columns exist up front
    plot_df = plot_df.copy()
    for col in ("upper_limit", "lower_limit", "flagged"):
        if col not in plot_df.columns:
            plot_df[col] = np.nan if col != "flagged" else False
    plot_df["flagged"] = False  # reset

    # Visual x scaling to mimic earlier "circles look round" behavior
    ax = 0.25

    # Build per-point radius (constant or gradient-based)
    if gradient_radius:
        dy_avg = np.gradient(y_avg, x_avg)
        slope_mag = np.abs(dy_avg)
        slope_norm = slope_mag / (np.nanmax(slope_mag) + 1e-9)
        slope_scale = 1.5
        r_pts = width * (1 + slope_scale * slope_norm)
    else:
        r_pts = np.full_like(x_avg, float(width), dtype=float)

    # Interpolate centers along the Average path
    xs_list, ys_list, rs_list = [], [], []
    for i in range(len(x_avg) - 1):
        x0, y0, r0 = x_avg[i],   y_avg[i],   r_pts[i]
        x1, y1, r1 = x_avg[i+1], y_avg[i+1], r_pts[i+1]
        t = np.linspace(0.0, 1.0, int(interp_factor), endpoint=False) if interp_factor > 0 else np.array([], dtype=float)
        if t.size:
            xs_list.append((1 - t) * x0 + t * x1)
            ys_list.append((1 - t) * y0 + t * y1)
            rs_list.append((1 - t) * r0 + t * r1)

    # include the last point
    xs = np.concatenate(xs_list + [x_avg[-1:]]) if xs_list else x_avg.copy()
    ys = np.concatenate(ys_list + [y_avg[-1:]]) if ys_list else y_avg.copy()
    rs = np.concatenate(rs_list + [r_pts[-1:]]) if rs_list else r_pts.copy()

    # Prepare scanline grid over integer x
    x_min = int(np.floor(np.nanmin(x_avg)))
    x_max = int(np.ceil(np.nanmax(x_avg)))
    X = np.arange(x_min, x_max + 1, dtype=int)

    lower = np.full(X.shape, np.nan, dtype=float)
    upper = np.full(X.shape, np.nan, dtype=float)

    # Analytic sweep: update envelopes where each disk contributes
    for xc, yc, rc in zip(xs, ys, rs):
        dx_max = rc * ax
        if dx_max <= 0 or not np.isfinite(dx_max):
            continue
        xL = int(np.ceil(xc - dx_max))
        xR = int(np.floor(xc + dx_max))
        xL = max(xL, x_min); xR = min(xR, x_max)
        if xL > xR:
            continue

        # indices into X
        mask = (X >= xL) & (X <= xR)
        xi = X[mask].astype(float)
        dx = np.abs(xi - xc)
        # guard for numerical noise
        term = rc*rc - (dx / ax)**2
        term = np.where(term < 0.0, 0.0, term)
        vy = np.sqrt(term)

        lo = yc - vy
        hi = yc + vy

        # update min/max with NaN-aware logic
        idx = np.where(mask)[0]
        lower[idx] = np.where(np.isnan(lower[idx]), lo, np.minimum(lower[idx], lo))
        upper[idx] = np.where(np.isnan(upper[idx]), hi, np.maximum(upper[idx], hi))

    #Merge limits back into plot_df
    limits = pd.DataFrame({"x": X, "lower_limit": lower, "upper_limit": upper})
    plot_df = plot_df.drop(columns=[c for c in ["upper_limit","lower_limit"] if c in plot_df.columns])
    plot_df = plot_df.merge(limits, on="x", how="left")

    # Return plot_df
    return plot_df



# ---------------------------------- Algorithm #2: apply_air_limit() ----------------------------------
def apply_air_limit(
    plot_df: pd.DataFrame,
    *,
    x_pct: float = 0.10,   # portion of x-axis to cover (0..1]
    y_pct: float = 0.10,   # how far down to place the line, as a fraction of Y span (0..1]
    y_ref: str = "axis",   # "axis" -> use overall pressure span; "avg" -> span of Average
    baseline: str = "median_early"  # "median_early" | "mean_early" | "avg_at_zero"
) -> pd.DataFrame:
    """
    Create a horizontal lower-limit from x=0 to x = x_pct * xmax at a level that is
    `y_pct` * (chosen Y-span) BELOW an early baseline of the Average curve.
    Outside that window, limits are NaN.

    Returns plot_df with 'lower_limit' and 'upper_limit' (upper is NaN here).
    """
    print_banner("Applying Algorithm #2: apply_air_limit()")

    pdf = plot_df.copy()

    # Pull Average series
    avg = (pdf.loc[pdf["sample_id"] == "Average", ["x", "pressure"]]
              .sort_values("x")
              .dropna(subset=["x"]))
    if avg.empty:
        raise ValueError("apply_front_window_limit: 'Average' series not found.")

    x_avg = avg["x"].to_numpy(dtype=float)
    y_avg = avg["pressure"].to_numpy(dtype=float)

    # Prepare x grid & early window
    x_min = int(np.nanmin(x_avg))
    x_max = int(np.nanmax(x_avg))
    X = np.arange(x_min, x_max + 1, dtype=int)

    # clip inputs
    x_pct = float(np.clip(x_pct, 0.0, 1.0))
    y_pct = float(np.clip(y_pct, 0.0, 1.0))

    if x_pct == 0.0:
        # nothing to draw; ensure columns exist and return
        if "lower_limit" not in pdf.columns: pdf["lower_limit"] = np.nan
        if "upper_limit" not in pdf.columns: pdf["upper_limit"] = np.nan
        return pdf

    x_cut = x_min + int(np.floor((x_max - x_min) * x_pct))

    # Choose Y span
    if y_ref == "avg":
        y_span = np.nanmax(y_avg) - np.nanmin(y_avg)
    else:
        # overall axis span from all pressures
        y_all = pdf["pressure"].to_numpy(dtype=float)
        y_span = np.nanmax(y_all) - np.nanmin(y_all)

    if not np.isfinite(y_span) or y_span <= 0:
        # fallback: robust span via percentiles
        y_vals = pdf["pressure"].to_numpy(dtype=float)
        q5, q95 = np.nanpercentile(y_vals, [5, 95])
        y_span = max(1e-9, q95 - q5)

    # Early-window baseline from the Average curve
    early_mask = (x_avg >= x_min) & (x_avg <= x_cut)
    if not np.any(early_mask):
        # if interpolation grid differs slightly, allow first 5% as fallback
        x_cut_fallback = x_min + max(1, int(np.floor((x_max - x_min) * 0.05)))
        early_mask = (x_avg >= x_min) & (x_avg <= x_cut_fallback)

    if baseline == "avg_at_zero":
        # value at (or near) x_min
        base = y_avg[np.nanargmin(np.abs(x_avg - x_min))]
    elif baseline == "mean_early":
        base = float(np.nanmean(y_avg[early_mask]))
    else:  # "median_early" (robust)
        base = float(np.nanmedian(y_avg[early_mask]))

    # Horizontal lower limit: base - y_pct * y_span
    level = base - y_pct * y_span

    # Build per-x limits
    lower = np.full_like(X, np.nan, dtype=float)
    upper = np.full_like(X, np.nan, dtype=float)  # unused, but keep for API parity
    window_mask = (X >= x_min) & (X <= x_cut)
    lower[window_mask] = level  # constant horizontal line in the window

    # Merge back to plot_df (per-x unique rows)
    limits = pd.DataFrame({"x": X, "lower_limit": lower, "upper_limit": upper})
    pdf = pdf.drop(columns=[c for c in ["lower_limit", "upper_limit"] if c in pdf.columns])
    pdf = pdf.merge(limits, on="x", how="left")

    return pdf


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


# ---------------------------------- Make Plots ----------------------------------
make_plots(pressure_df, 
           enable_algorithm1=True,
           )

print_banner("Plots Generated!")
