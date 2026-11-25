"""
examples/plot_sweep.py
Generates the NACA 0012 angle-of-attack sweep comparison plot:
Conventional surface loads (C_L, min C_f) vs. Internal scale metrics (T_M, j_bar).

Usage:
  1. Generate immediate reference figure (no CFD files required):
     python examples/plot_sweep.py

  2. Process real SU2 sweep CSV exports (flow_aoa0.csv, flow_aoa4.csv, ...):
     python examples/plot_sweep.py --data-dir ./su2_runs --re 3000000 --uinf 51.0
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aero_scale_diag import STANDARD_WINDOWS, compute_scale_observables


def run_preview_sweep() -> dict:
    """Produces a physically consistent NACA 0012 Re=3e6 reference sweep for layout verification."""
    aoa = np.array([0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0])
    cl = np.array([0.00, 0.44, 0.88, 1.09, 1.28, 1.42, 1.45, 1.31])
    min_cf = np.array([0.0018, 0.0014, 0.0008, 0.0004, 0.00005, -0.0003, -0.0007, -0.0012])

    # Internal boundary-layer scale migration rising as adverse pressure gradient thickens shear layer
    t_m = np.array([0.014, 0.016, 0.024, 0.048, 0.115, 0.235, 0.310, 0.365])
    j_bar = np.array([1.82, 1.85, 1.94, 2.15, 2.48, 2.86, 3.04, 3.18])

    return {"aoa": aoa, "cl": cl, "min_cf": min_cf, "T_M": t_m, "j_bar": j_bar, "is_preview": True}


def run_su2_sweep(data_dir: str, re_c: float, u_inf: float) -> dict:
    """Reads SU2 volume CSV exports named flow_aoa<angle>.csv and computes real observables."""
    files = sorted(glob.glob(os.path.join(data_dir, "flow_aoa*.csv")))
    if not files:
        raise FileNotFoundError(f"No flow_aoa*.csv files found in {data_dir}")

    records = []
    for fpath in files:
        match = re.search(r"aoa([+-]?\d+(?:\.\d+)?)", os.path.basename(fpath), re.IGNORECASE)
        if not match:
            continue
        alpha = float(match.group(1))
        arr = np.genfromtxt(fpath, delimiter=",", names=True)
        obs = compute_scale_observables(
            arr["x"], arr["y"], arr["u"], arr["v"],
            re_c=re_c, u_inf=u_inf, bounds=STANDARD_WINDOWS["bl"],
        )
        surf_path = fpath.replace("flow_aoa", "surface_aoa")
        cl_val, cf_val = np.nan, np.nan
        if os.path.exists(surf_path):
            s_arr = np.genfromtxt(surf_path, delimiter=",", names=True)
            if "Skin_Friction_Coefficient_x" in s_arr.dtype.names:
                cf_val = float(np.min(s_arr["Skin_Friction_Coefficient_x"]))
        records.append((alpha, cl_val, cf_val, obs["T_M"], obs["j_bar"]))

    records.sort(key=lambda r: r[0])
    arr_rec = np.array(records)
    return {
        "aoa": arr_rec[:, 0],
        "cl": arr_rec[:, 1],
        "min_cf": arr_rec[:, 2],
        "T_M": arr_rec[:, 3],
        "j_bar": arr_rec[:, 4],
        "is_preview": False,
    }


def make_plot(res: dict, out_path: str = "naca0012_benchmark.png") -> None:
    aoa = res["aoa"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    color_cl = "#1f77b4"
    ax1.plot(aoa, res["cl"], "o-", color=color_cl, linewidth=2, label="Lift Coefficient ($C_L$)")
    ax1.set_ylabel("$C_L$", color=color_cl, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_cl)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1_r = ax1.twinx()
    color_cf = "#d62728"
    ax1_r.plot(aoa, res["min_cf"], "s--", color=color_cf, linewidth=1.8, label="Min Skin Friction ($\\min C_f$)")
    ax1_r.axhline(0.0, color=color_cf, linestyle=":", alpha=0.6)
    ax1_r.set_ylabel("$\\min C_f$ (Separation $\\leq 0$)", color=color_cf, fontsize=10)
    ax1_r.tick_params(axis="y", labelcolor=color_cf)

    title_tag = " [Synthetic Reference Curve]" if res["is_preview"] else " [SU2 RANS Re = 3e6]"
    ax1.set_title(f"NACA 0012 Alpha Sweep: Surface Loads vs. Dyadic Scale Diagnostics{title_tag}", fontsize=11)

    color_tm = "#2ca02c"
    ax2.plot(aoa, res["T_M"], "o-", color=color_tm, linewidth=2.2, label="High-Tail Fraction ($T_M$)")
    ax2.set_ylabel("High-Tail Fraction ($T_M$)", color=color_tm, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Angle of Attack $\\alpha$ (deg)", fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_tm)
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2_r = ax2.twinx()
    color_jb = "#9467bd"
    ax2_r.plot(aoa, res["j_bar"], "^--", color=color_jb, linewidth=1.8, label="Scale Centroid ($\\bar{j}$)")
    ax2_r.set_ylabel("Scale Centroid ($\\bar{j}$)", color=color_jb, fontsize=10)
    ax2_r.tick_params(axis="y", labelcolor=color_jb)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved benchmark chart to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NACA 0012 Sweep Diagnostics")
    parser.add_argument("--data-dir", type=str, help="Directory with SU2 flow_aoa*.csv exports")
    parser.add_argument("--re", type=float, default=3e6, help="Chord Reynolds number")
    parser.add_argument("--uinf", type=float, default=51.0, help="Freestream velocity (m/s)")
    parser.add_argument("--out", type=str, default="naca0012_benchmark.png", help="Output PNG path")
    args = parser.parse_args()

    results = run_su2_sweep(args.data_dir, args.re, args.uinf) if args.data_dir else run_preview_sweep()
    make_plot(results, out_path=args.out)
