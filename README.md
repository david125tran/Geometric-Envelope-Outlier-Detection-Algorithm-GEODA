# 🧠 Outlier-Detection-Algorithm

This **Outlier-Detection-Algorithm** is a custom time-series analysis method designed to identify outlier signals by constructing a **rolling geometric envelope** around an averaged reference trace.  
  
Unlike conventional statistical methods that rely on standard deviations or quantiles, it uses **circular geometry** to build a dynamic, visually intuitive tolerance band that adapts to local variations in the data.  The algorithm is dynamic because it adapts its geometric envelope to the local shape and variability of the averaged signal—automatically adjusting the upper and lower limits based on the data’s real-time trends rather than using fixed thresholds.  

---

## 🚀 Overview
This algorithm was developed for analyzing data generated from all kinds of sources and the approach is generalizable to any dense time-series data.  

The algorithm works by:
1. Computing an average trace from multiple aligned signals.
2. Building a **series of circles** along that average trace to form an envelope region.
3. Deriving upper and lower bounds (the top and bottom of the envelope).
4. Flagging any trace that exits the envelope as an **outlier**.

This produces a smooth, interpretable visual envelope that can dynamically adapt to nonlinear trends or noise while maintaining deterministic, parameter-based control.

---

## 🧩 Core Algorithm

**Input:**  
A collection of aligned time-series signals in tabular form (e.g., y-data vs. time).

**Steps:**
1. **Compute Average Curve**  
   - Calculate the mean y value across all traces at each time index (ignoring NaNs).  
2. **Construct Rolling Circles**  
   - Place a circle of fixed radius `width` around each average point, with optional interpolation between points (`interp_factor`) for smoother envelopes.  
3. **Derive Envelope Limits**  
   - For all circles combined, determine the **minimum (lower limit)** and **maximum (upper limit)** y-values per x-index.  
4. **Flag Outliers**  
   - Any signal crossing above or below these limits is marked as *flagged*.

**Output:**  
A DataFrame or Plotly figure showing all traces, the mean trace, and the geometric envelope with flagged outliers highlighted.

---

## ⚙️ Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `width` | `float` | Radius of each circle; controls the envelope thickness. |
| `interp_factor` | `int` | Number of interpolated circles between consecutive average points; controls smoothness. |
| `x_scale` | `float` | Visual x-scaling factor to keep circles round (default = 0.25). |
| `show_circles` | `bool` | Whether to render the full circular geometry (for visualization/debug). |

---

## 🧮 Example

