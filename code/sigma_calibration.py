"""Frozen-network single-point segment diameter (sigma) calibration for PolySAFT."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from pcsaft_eos import solve

torch.set_default_dtype(torch.float64)


def predict_eos_density(
    params: dict[str, float],
    T_K: np.ndarray,
    P_Pa: np.ndarray,
    M_g_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate float64 PC-SAFT mass density (kg/m^3) for arrays of thermodynamic states."""
    T = torch.tensor(T_K, dtype=torch.float64)
    P = torch.tensor(P_Pa, dtype=torch.float64)
    M = torch.tensor(M_g_mol, dtype=torch.float64)
    m_seg = torch.full_like(T, float(params["m_per_mw"])) * M
    sig = torch.full_like(T, float(params["sigma"]))
    eps = torch.full_like(T, float(params["eps_k"]))
    rr, info = solve(m_seg, sig, eps, T, P, differentiable=False)
    rho = (rr * M / 1000.0).detach().numpy()
    ok = info["ok"].detach().numpy() & np.isfinite(rho) & (rho > 0)
    return rho, ok


def calibrate_sigma_single_point(
    params: dict[str, float],
    mw_g_mol: float,
    T_K: float,
    P_Pa: float,
    rho_obs_kg_m3: float,
    bounds: tuple[float, float] = (2.5, 5.0),
) -> dict:
    """Calibrate segment diameter sigma in [2.5, 5.0] A on one experimental liquid density point."""
    T_arr = np.array([T_K], dtype=float)
    P_arr = np.array([P_Pa], dtype=float)
    M_arr = np.array([mw_g_mol], dtype=float)
    trials: list[tuple[float, float, bool]] = []

    def objective(sigma_val: float) -> float:
        trial_p = {
            "m_per_mw": float(params["m_per_mw"]),
            "sigma": float(sigma_val),
            "eps_k": float(params["eps_k"]),
        }
        rho, ok = predict_eos_density(trial_p, T_arr, P_arr, M_arr)
        valid = bool(ok.all())
        loss = float(np.mean(np.log(rho / rho_obs_kg_m3) ** 2)) if valid else float("inf")
        trials.append((float(sigma_val), loss, valid))
        return loss

    objective(float(params["sigma"]))
    objective(bounds[0])
    objective(bounds[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minimize_scalar(
            objective,
            method="bounded",
            bounds=bounds,
            options={"maxiter": 77, "xatol": 1e-7},
        )
    finite = [t for t in trials if t[2] and np.isfinite(t[1])]
    if not finite:
        raise RuntimeError("No admissible sigma root found within [2.5, 5.0] A")
    best = min(finite, key=lambda t: (t[1], abs(t[0] - float(params["sigma"])), t[0]))
    return {
        "original_sigma": float(params["sigma"]),
        "selected_sigma": float(best[0]),
        "support_log_mse": float(best[1]),
        "objective_calls": len(trials),
    }
