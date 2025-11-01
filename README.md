# 🧠 Outlier-Detection-Algorithm
This repository contains two complementary Python implementations of a **time-series outlier detection algorithm** designed to be generalizable to any dense time-series data.

The method identifies outlier signals by constructing a **rolling geometric envelope** around an averaged reference trace, rather than relying on fixed statistical thresholds (like standard deviations or quantiles). The algorithm is **dynamic** because it adapts its geometric envelope to the local shape and variability of the averaged signal—automatically adjusting the upper and lower limits based on the data’s real-time trends instead of using fixed thresholds.

In the real world, some systems need flexible solutions for dynamic problems. This algorithm represents one such approach. I had a clear mental image of what I wanted to build, but it took several iterations to translate that concept into code.

---

## 🚦 Two Versions
- 1️⃣- [**plot-outlier-detected.py**](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/plot-outlier-detector.py) — **Visualized Version** - This version was built for concept visualization and debugging.
It renders the full rolling circle geometry used to construct the envelope, showing how each circular segment contributes to the final tolerance band.  This script is computationally expensive due to circle generation and rendering
- 2️⃣ - [**plot-outlier-detector-refactored.py**](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/plot-outlier-detector-refactored.py) — **Production Version** - This version is the **fully refactored, production-grade implementation** of the same algorithm.
    - ⚡ No circle objects are generated; instead, envelopes are computed analytically.
    - 🧮 The algorithm uses direct geometric computation (based on analytic scanline logic) to achieve the same upper/lower bounds **orders of magnitude faster**.
    - 🧠 Additional lightweight features are added for automation pipelines, such as:
    - Dynamic NaN handling and averaging.
    - 💻 Optimized for **speed, scalability, and memory efficiency**
    
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
## 🧮 Example
- Here we have a plot where we may want to flag specific curve(s) that deviate from the norm.  Notice the one peak around `time=12 (units)`.  Is this a curve we would want to flag?
![Plot with Atypical Curve](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/Images/1.png?raw=true)
- We can compute the average at each time point **(blue line)**.  Crawl along on the rolling average to create circles to create an envelope.  
![Plot with Atypical Curve](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/Images/2.png?raw=true)
- From the envelope, we can form upper and lower limits to figure out what to flag.  
![Plot with Atypical Curve](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/Images/3.png?raw=true)
- We can then call the garbage collector to get rid of the circles to free up memory and we see our limits **(green circles)**.  Any curves thare are outside of the limits are flagged and colored as **red/pink**.  
![Plot with Atypical Curve](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/Images/4.png?raw=true)
- And then we can control the width of these limits by changing the circle width.  
![Plot with Atypical Curve](https://github.com/david125tran/Outlier-Detection-Algorithm/blob/main/Images/5.png?raw=true)




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
