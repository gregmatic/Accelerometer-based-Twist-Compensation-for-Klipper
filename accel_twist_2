# accel_twist_global_all_methods.py
# ------------------------------------------------------------
# Global-axis twist estimation with multiple estimators & honest uncertainty
# * No scikit-learn required (uses NumPy SVD, Statsmodels)
# * Methods:
#     (1) WLS on per-position means (weights = 1/Var(mean))
#     (2) OLS with cluster-robust SEs (clusters = position) on all samples
#     (3) Robust Linear Model (RLM, Huber) on per-position means
#     (4) Theil–Sen (median of pairwise slopes) on per-position means
# * Nonparametric uncertainty: cluster bootstrap (resampling positions)
# * Angles measured in ONE global reference basis (from ALL files)
#
# References:
# - statsmodels WLS and robust covariance (cluster): https://www.statsmodels.org/stable/examples/notebooks/generated/wls.html
#   and https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
# - Cluster-robust inference discussion: https://stackoverflow.com/questions/30553838
# - Robust Linear Models (Huber/Tukey): https://www.statsmodels.org/stable/rlm.html
# - Theil–Sen definition: https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
# - Note: scipy.stats.linregress SE assumes i.i.d. residuals; not suitable for clustered repeats:
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# ------------------------------------------------------------

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations

# -----------------------
# 0) Utilities
# -----------------------
def find_files(pattern="lis2dw-*.csv"):
    flist = glob.glob(pattern)
    if not flist:
        raise RuntimeError(f"No files matched pattern '{pattern}' in cwd={os.getcwd()}")
    return flist

def parse_position(path):
    base = os.path.basename(path)
    m = re.search(r"^lis2dw-(-?\d+(?:\.\d+)?)\.csv$", base)
    if not m:
        raise RuntimeError(f"Filename does not match expected pattern: {base}")
    return float(m.group(1))

def orthonormal_basis_from_axis(axis, ref_vec=None):
    """Build right-handed basis (u1,u2,axis) with u1 aligned to ref_vec projected to plane."""
    v = axis / np.linalg.norm(axis)
    if ref_vec is None or np.linalg.norm(ref_vec) < 1e-12:
        ref_vec = np.array([1.0, 0.0, 0.0])
    u1 = ref_vec - np.dot(ref_vec, v) * v
    if np.linalg.norm(u1) < 1e-12:
        ref_alt = np.array([0.0, 1.0, 0.0])
        u1 = ref_alt - np.dot(ref_alt, v) * v
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1); u2 /= np.linalg.norm(u2)
    return u1, u2

def compute_angles_deg(acc, u1, u2):
    x = acc @ u1
    y = acc @ u2
    theta = np.arctan2(y, x)
    theta = np.unwrap(theta)  # guard against ±pi wrap
    return np.degrees(theta)

def svd_pca_axis(all_acc):
    """
    Global axis via SVD (no sklearn):
    After centering, components are rows of Vt; smallest-variance component is Vt[-1].
    """
    X = all_acc - all_acc.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[-1, :]
    axis /= np.linalg.norm(axis)
    return axis

def per_position_stats(theta_by_position):
    rows = []
    for pos in sorted(theta_by_position.keys()):
        th = np.asarray(theta_by_position[pos])
        n = th.size
        mu = th.mean()
        sd = th.std(ddof=1)
        se = sd / np.sqrt(n)
        rows.append((pos, n, mu, sd, se))
    df = pd.DataFrame(rows, columns=["position_mm","n","theta_mean_deg","theta_sd_deg","theta_se_deg"])
    return df.sort_values("position_mm").reset_index(drop=True)

# --- Estimators ---
def fit_wls_on_means(pos_stats):
    X = sm.add_constant(pos_stats["position_mm"].values)
    y = pos_stats["theta_mean_deg"].values
    # weights = 1 / Var(mean); add tiny epsilon to avoid divide-by-zero
    w = 1.0 / (pos_stats["theta_se_deg"].values ** 2 + 1e-12)
    res = sm.WLS(y, X, weights=w).fit()
    b0, b1 = res.params
    se = res.bse[1]
    ci = res.conf_int()
    ci_low, ci_high = ci[1, 0], ci[1, 1]
    return {"name":"WLS (means)", "b0":b0, "b1":b1, "se":se, "ci":(ci_low, ci_high), "res":res}

def fit_ols_cluster_all(positions_all, theta_all):
    X = sm.add_constant(np.asarray(positions_all))
    y = np.asarray(theta_all)
    groups = np.asarray(positions_all)  # cluster by position
    res = sm.OLS(y, X).fit(cov_type='cluster',
                           cov_kwds={'groups': groups,
                                     'use_correction': True,
                                     'df_correction': True})
    b0, b1 = res.params
    se = res.bse[1]
    ci = res.conf_int()
    ci_low, ci_high = ci[1, 0], ci[1, 1]
    return {"name":"OLS (cluster-robust)", "b0":b0, "b1":b1, "se":se, "ci":(ci_low, ci_high), "res":res}

def fit_rlm_on_means(pos_stats, norm="Huber"):
    X = sm.add_constant(pos_stats["position_mm"].values)
    y = pos_stats["theta_mean_deg"].values
    if norm.lower().startswith("tuke"):
        M = sm.robust.norms.TukeyBiweight()
        name = "RLM Tukey"
    else:
        M = sm.robust.norms.HuberT()
        name = "RLM Huber"
    res = sm.RLM(y, X, M=M).fit()
    b0, b1 = res.params
    se = res.bse[1] if hasattr(res, "bse") else np.nan
    if hasattr(res, "conf_int"):
        ci = res.conf_int()
        ci_low, ci_high = ci[1, 0], ci[1, 1]
    else:
        ci_low = ci_high = np.nan
    return {"name":name, "b0":b0, "b1":b1, "se":se, "ci":(ci_low, ci_high), "res":res}

def theil_sen_on_means(pos_stats):
    """
    Theil–Sen slope on per-position means: median of pairwise slopes; intercept as median(y - m x).
    """
    x = pos_stats["position_mm"].values
    y = pos_stats["theta_mean_deg"].values
    idx = range(len(x))
    slopes = [(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(idx, 2) if x[j] != x[i]]
    m = np.median(slopes)
    b = np.median(y - m * x)
    return m, b, np.array(slopes)

# --- Cluster bootstrap (resample positions) ---
def cluster_bootstrap_slopes(theta_by_position, B=2000, rng_seed=42, method="WLS"):
    rng = np.random.default_rng(rng_seed)
    positions = np.array(sorted(theta_by_position.keys()))
    slopes = []

    if method.upper() == "WLS":
        pos_stats_full = per_position_stats(theta_by_position)
        x_all = pos_stats_full["position_mm"].values
        y_mean_all = pos_stats_full["theta_mean_deg"].values
        se_all = pos_stats_full["theta_se_deg"].values
        for _ in range(B):
            sel = rng.choice(len(positions), size=len(positions), replace=True)
            x = x_all[sel]; y = y_mean_all[sel]; se = se_all[sel]
            X = sm.add_constant(x)
            w = 1.0 / (se**2 + 1e-12)
            slopes.append(sm.WLS(y, X, weights=w).fit().params[1])

    elif method.upper() == "OLSALL":
        for _ in range(B):
            sel_pos = rng.choice(positions, size=len(positions), replace=True)
            x_list, y_list = [], []
            for p in sel_pos:
                th = np.asarray(theta_by_position[p])
                x_list.append(np.full_like(th, p, dtype=float))
                y_list.append(th)
            x = np.concatenate(x_list); y = np.concatenate(y_list)
            X = sm.add_constant(x)
            slopes.append(sm.OLS(y, X).fit().params[1])

    elif method.upper() == "THEILSEN":
        pos_stats_full = per_position_stats(theta_by_position)
        x_all = pos_stats_full["position_mm"].values
        y_mean_all = pos_stats_full["theta_mean_deg"].values
        for _ in range(B):
            sel = rng.choice(len(positions), size=len(positions), replace=True)
            x = x_all[sel]; y = y_mean_all[sel]
            idx = range(len(x))
            s = [(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(idx, 2) if x[j] != x[i]]
            slopes.append(np.median(s))
    else:
        raise ValueError("Unknown method for bootstrap. Choose 'WLS', 'OLSALL', or 'THEILSEN'.")
    return np.array(slopes)

# -----------------------
# 1) Load all files + build GLOBAL basis
# -----------------------
def main():
    print("cwd:", os.getcwd())
    files = find_files("lis2dw-*.csv")

    all_acc = []
    per_file = []  # (position, file, acc-array)
    for fp in files:
        pos = parse_position(fp)
        df = pd.read_csv(fp)
        if df.shape[1] < 4:
            print(f"[WARN] {os.path.basename(fp)} has <4 columns; need accel in cols 1..3; skipping.")
            continue
        if len(df) < 120:
            print(f"[WARN] {os.path.basename(fp)} too few rows ({len(df)}); skipping.")
            continue
        data = df.iloc[50:-50].copy()
        acc = data.iloc[:, 1:4].to_numpy(dtype=float)   # assumes columns [1,2,3] are X,Y,Z
        if acc.shape[0] < 10:
            print(f"[WARN] {os.path.basename(fp)} insufficient samples after trim; skipping.")
            continue
        per_file.append((pos, fp, acc))
        all_acc.append(acc)

    if not per_file:
        raise RuntimeError("No usable files after trimming/filtering.")

    all_acc = np.vstack(all_acc)
    global_axis = svd_pca_axis(all_acc)

    # Make axis sign stable wrt global mean
    gmean = all_acc.mean(axis=0)
    if np.dot(global_axis, gmean) < 0:
        global_axis = -global_axis

    u1, u2 = orthonormal_basis_from_axis(global_axis, ref_vec=gmean)
    print("Global axis:", np.round(global_axis, 6))
    print("u1:", np.round(u1, 6), "u2:", np.round(u2, 6))

    # -----------------------
    # 2) Angles in GLOBAL basis
    # -----------------------
    theta_by_position = {}
    positions_all, theta_all = [], []
    for pos, fp, acc in sorted(per_file, key=lambda t: t[0]):
        th_deg = compute_angles_deg(acc, u1, u2)
        # DO NOT subtract per-file offsets; the basis is unified
        theta_by_position[pos] = th_deg
        positions_all.extend([pos]*len(th_deg))
        theta_all.extend(th_deg)
        print(f"[OK] {os.path.basename(fp)} -> pos={pos}, N={len(th_deg)}, "
              f"mean={th_deg.mean():.3f}°, sd={th_deg.std(ddof=1):.3f}°")

    positions_all = np.asarray(positions_all, float)
    theta_all = np.asarray(theta_all, float)

    # -----------------------
    # 3) Per-position summary
    # -----------------------
    pos_stats = per_position_stats(theta_by_position)
    print("\nPer-position stats:")
    print(pos_stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # -----------------------
    # 4) Estimation suite
    # -----------------------
    est_wls = fit_wls_on_means(pos_stats)
    est_ols_cr = fit_ols_cluster_all(positions_all, theta_all)
    est_rlm = fit_rlm_on_means(pos_stats, norm="Huber")

    ts_slope, ts_intercept, ts_pair_slopes = theil_sen_on_means(pos_stats)
    # Cluster bootstrap CIs
    ts_boot = cluster_bootstrap_slopes(theta_by_position, B=2000, rng_seed=42, method="THEILSEN")
    ts_ci = (np.percentile(ts_boot, 2.5), np.percentile(ts_boot, 97.5))

    wls_boot = cluster_bootstrap_slopes(theta_by_position, B=2000, rng_seed=43, method="WLS")
    wls_ci_boot = (np.percentile(wls_boot, 2.5), np.percentile(wls_boot, 97.5))

    # -----------------------
    # 5) Report
    # -----------------------
    def fmt_ci(ci): return f"[{ci[0]:.6f}, {ci[1]:.6f}]"

    print("\n=== Twist slope estimates (deg/mm) ===")
    print(f"{est_wls['name']}:    slope = {est_wls['b1']:.6f}  SE = {est_wls['se']:.6f}  CI95 = {fmt_ci(est_wls['ci'])}")
    print(f"  (WLS cluster-bootstrap CI95 = {fmt_ci(wls_ci_boot)})")
    print(f"{est_ols_cr['name']}: slope = {est_ols_cr['b1']:.6f}  SE = {est_ols_cr['se']:.6f}  CI95 = {fmt_ci(est_ols_cr['ci'])}")
    print(f"{est_rlm['name']}:    slope = {est_rlm['b1']:.6f}  SE = {est_rlm['se']:.6f}  CI95 = {fmt_ci(est_rlm['ci'])}")
    print(f"Theil–Sen (means):  slope = {ts_slope:.6f}  (cluster-bootstrap CI95 = {fmt_ci(ts_ci)})")
    print("\nIntercepts:")
    print(f"WLS b0 = {est_wls['b0']:.6f}, OLS-CR b0 = {est_ols_cr['b0']:.6f}, "
          f"RLM b0 = {est_rlm['b0']:.6f}, Theil–Sen b0 = {ts_intercept:.6f}")

    # -----------------------
    # 6) Visualization
    # -----------------------
    plt.figure(figsize=(12, 6))

    # Boxplot of all samples per position (global basis)
    sorted_positions = pos_stats["position_mm"].values.tolist()
    box_data = [theta_by_position[p] for p in sorted_positions]
    plt.boxplot(box_data, positions=sorted_positions, widths=1.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'), showfliers=False)

    # Per-position mean ±95% CI (from SE of mean)
    means = pos_stats["theta_mean_deg"].values
    ci95 = 1.96 * pos_stats["theta_se_deg"].values
    plt.errorbar(sorted_positions, means, yerr=ci95, fmt='o', color='k', ecolor='k', capsize=3,
                 label='Per-position mean ± 95% CI')

    # Fitted lines
    xx = np.linspace(min(sorted_positions), max(sorted_positions), 200)
    yy_wls = est_wls['b0'] + est_wls['b1'] * xx
    yy_ols = est_ols_cr['b0'] + est_ols_cr['b1'] * xx
    yy_rlm = est_rlm['b0'] + est_rlm['b1'] * xx
    yy_ts  = ts_intercept + ts_slope * xx
    plt.plot(xx, yy_wls, 'r-',  label=f"WLS: {est_wls['b1']:.6f}±{est_wls['se']:.6f}")
    plt.plot(xx, yy_ols, 'g--', label=f"OLS-CR: {est_ols_cr['b1']:.6f}±{est_ols_cr['se']:.6f}")
    plt.plot(xx, yy_rlm, 'm-.', label=f"RLM: {est_rlm['b1']:.6f}±{est_rlm['se']:.6f}")
    plt.plot(xx, yy_ts,  'b:',  label=f"Theil–Sen: {ts_slope:.6f} (boot CI {fmt_ci(ts_ci)})")

    plt.xlabel("Position (mm)")
    plt.ylabel("Theta (deg) — global basis")
    plt.title("Twist vs Position — Global Axis & Multiple Estimators")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
