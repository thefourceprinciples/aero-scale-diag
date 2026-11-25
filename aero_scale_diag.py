"""
aero_scale_diag.py
Dyadic scale-redistribution post-processor for CFD velocity fields (SU2 / CSV).
"""

import argparse
import json
import numpy as np
from scipy.interpolate import griddata
from scipy.signal.windows import tukey

STANDARD_WINDOWS = {
    "bl": (0.05, 1.00, 0.00, 0.15),    # Upper-surface boundary layer (x/c, y/c)
    "wake": (1.00, 2.50, -0.35, 0.35), # Near-wake shear region (x/c, y/c)
}


def _tukey_2d(ny: int, nx: int, alpha: float = 0.25) -> np.ndarray:
    """2D Tukey taper to eliminate crop-boundary Gibbs aliasing in FFT."""
    return np.outer(tukey(ny, alpha=alpha), tukey(nx, alpha=alpha))


def compute_scale_observables(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    re_c: float,
    u_inf: float = 1.0,
    bounds: tuple[float, float, float, float] = STANDARD_WINDOWS["wake"],
    grid_shape: tuple[int, int] = (128, 256),
    frontier_m: int = 4,
    alpha_env: float = 1.0,
    beta_front: float = 1.0,
    taper_alpha: float = 0.25,
) -> dict:
    """
    Computes dyadic L^3 scale norms q_j and derived aerodynamic observables
    (T_M, j_bar, E_M, F_frontier_M) inside a declared spatial window.
    """
    xmin, xmax, ymin, ymax = bounds
    ny, nx = grid_shape

    # 1. Resample unstructured/stretched CFD data onto uniform Cartesian grid
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)

    u_norm = griddata((x, y), u / u_inf, (grid_x, grid_y), method="linear", fill_value=0.0)
    v_norm = griddata((x, y), v / u_inf, (grid_x, grid_y), method="linear", fill_value=0.0)

    # 2. Mean subtraction and smooth boundary apodization
    win = _tukey_2d(ny, nx, alpha=taper_alpha)
    u_tilde = win * (u_norm - np.mean(u_norm))
    v_tilde = win * (v_norm - np.mean(v_norm))

    # 3. Radial dyadic bandpass decomposition
    u_hat = np.fft.fft2(u_tilde)
    v_hat = np.fft.fft2(v_tilde)

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)

    j_max = int(np.floor(np.log2(min(nx, ny) / 2))) - 1
    m_eff = min(frontier_m, max(1, j_max - 2))

    q_shells = {}
    for j in range(j_max + 1):
        k_low = 2**j if j > 0 else 0.5
        k_high = 2 ** (j + 1)
        band = (k_mag >= k_low) & (k_mag < k_high)

        du_j = np.real(np.fft.ifft2(u_hat * band))
        dv_j = np.real(np.fft.ifft2(v_hat * band))
        vel_mag_j = np.sqrt(du_j**2 + dv_j**2)

        l3_norm = float(np.mean(vel_mag_j**3) ** (1.0 / 3.0))
        q_shells[j] = re_c * l3_norm

    # 4. Derived dimensionless observables
    j_arr = np.array(sorted(q_shells.keys()))
    q_arr = np.array([q_shells[j] for j in j_arr])
    q_energy = float(np.sum(q_arr**2)) + 1e-30

    h_arr = np.array([
        np.sum([2.0 ** (-alpha_env * abs(j - m)) * q_shells[m] for m in j_arr])
        for j in j_arr
    ])

    tail = j_arr >= m_eff
    e_m = float(np.max(h_arr[tail])) if np.any(tail) else 0.0
    t_m = float(np.sum(q_arr[tail] ** 2) / q_energy)
    j_bar = float(np.sum(j_arr * (q_arr**2)) / q_energy)
    f_frontier = float(
        sum(
            (2.0 ** (-beta_front * r)) * q_shells[m_eff - r]
            for r in range(1, m_eff + 1)
            if (m_eff - r) in q_shells
        )
    )

    return {
        "Re_c": re_c,
        "frontier_M": m_eff,
        "j_max": j_max,
        "T_M": round(t_m, 6),
        "j_bar": round(j_bar, 4),
        "E_M": round(e_m, 4),
        "F_frontier_M": round(f_frontier, 4),
        "q_shells": {str(k): round(v, 4) for k, v in q_shells.items()},
    }


def classify_regime(
    obs: dict,
    baseline_tm: float = 0.02,
    s1_lift_drop: bool = False,
    s2_load_var: bool = False,
    s3_coeff_mismatch: bool = False,
    grid_sensitive: bool = False,
) -> dict:
    """Evaluates field observables alongside surface/load channels (S1, S2, S3)."""
    mpp_elevated = obs["T_M"] > max(baseline_tm * 5.0, 0.01)

    if mpp_elevated and grid_sensitive:
        label = "NUMERICAL_CONTAMINATION"
    elif s3_coeff_mismatch and not mpp_elevated and not s1_lift_drop:
        label = "SENSOR_OR_MODEL_MISMATCH"
    elif mpp_elevated and (s1_lift_drop or s2_load_var):
        label = "SEPARATION_OR_BUFFET"
    elif mpp_elevated:
        label = "CANDIDATE_PRECURSOR"
    else:
        label = "ATTACHED_NOMINAL"

    return {
        "classification": label,
        "mpp_elevated": mpp_elevated,
        "observables": {
            "T_M": obs["T_M"],
            "j_bar": obs["j_bar"],
            "E_M": obs["E_M"],
            "F_frontier_M": obs["F_frontier_M"],
        },
    }


def _synthetic_benchmark() -> None:
    """Runs baseline vs. perturbed wake verification test."""
    rng = np.random.default_rng(42)
    x = rng.uniform(1.0, 2.5, 15000)
    y = rng.uniform(-0.35, 0.35, 15000)

    u_base = 1.0 - 0.3 * np.exp(-((y / 0.1) ** 2)) + 0.002 * np.sin(40 * x) * np.cos(40 * y)
    v_base = 0.01 * np.sin(6 * x) * np.exp(-((y / 0.1) ** 2))

    u_pert = u_base + 0.06 * np.sin(55 * x) * np.cos(55 * y) * np.exp(-((y / 0.12) ** 2))
    v_pert = v_base + 0.06 * np.cos(55 * x) * np.sin(55 * y) * np.exp(-((y / 0.12) ** 2))

    base = compute_scale_observables(x, y, u_base, v_base, re_c=3e6, bounds=STANDARD_WINDOWS["wake"])
    pert = compute_scale_observables(x, y, u_pert, v_pert, re_c=3e6, bounds=STANDARD_WINDOWS["wake"])
    diag = classify_regime(pert, baseline_tm=base["T_M"], s1_lift_drop=True)

    print(json.dumps({"baseline": base, "perturbed": pert, "diagnostic": diag}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerodynamic Scale-Redistribution Diagnostics")
    parser.add_argument("--csv", type=str, help="Path to CFD slice CSV with columns x,y,u,v")
    parser.add_argument("--re", type=float, default=3e6, help="Chord Reynolds number")
    parser.add_argument("--uinf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--window", choices=["bl", "wake"], default="wake", help="Analysis window")
    parser.add_argument("--baseline-tm", type=float, default=0.02, help="Attached baseline T_M")
    args = parser.parse_args()

    if not args.csv:
        _synthetic_benchmark()
    else:
        arr = np.genfromtxt(args.csv, delimiter=",", names=True)
        obs = compute_scale_observables(
            arr["x"], arr["y"], arr["u"], arr["v"],
            re_c=args.re, u_inf=args.uinf, bounds=STANDARD_WINDOWS[args.window],
        )
        print(json.dumps(classify_regime(obs, baseline_tm=args.baseline_tm), indent=2))
