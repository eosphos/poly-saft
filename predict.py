"""Direct SMILES-to-PC-SAFT parameter and melt density prediction with PolySAFT.

Supports:
  1. Five-fold ensemble prediction (15 MD-trained checkpoints: 5 folds x 3 seeds).
  2. Single-fold or shared last-layer experimental calibration checkpoint prediction.
  3. Frozen-network single-point sigma calibration given one low-pressure experimental
     density measurement (T_cal_K, P_cal_MPa, rho_cal_kg_m3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from model import PolySAFTModel, smiles_to_graph
from pcsaft_eos import solve
from sigma_calibration import calibrate_sigma_single_point

torch.set_default_dtype(torch.float64)


def load_checkpoints(
    mode: str = "ensemble",
    fold: int | None = None,
) -> list[tuple[str, PolySAFTModel]]:
    """Load requested PolySAFT checkpoint(s) on CPU in float64."""
    loaded: list[tuple[str, PolySAFTModel]] = []
    if mode == "ensemble":
        folds = [fold] if fold is not None else list(range(5))
        for f in folds:
            for seed in range(3):
                ckpt_path = ROOT / f"models/polysaft/runs/fold{f}/seed{seed}/density.pt"
                model = PolySAFTModel()
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                loaded.append((f"fold{f}_seed{seed}", model))
    elif mode in ("low_p", "multi_p"):
        tag = "low_pressure" if mode == "low_p" else "multi_pressure"
        folds = [fold] if fold is not None else [0, 3]
        for f in folds:
            if f not in (0, 3):
                raise ValueError("Shared experimental calibration checkpoints are defined for folds 0 and 3 (Folds 1 and 4 in paper).")
            for seed in range(3):
                ckpt_path = (
                    ROOT
                    / f"models/experimental_calibration/runs/network_{tag}_fold{f}_seed{seed}/best.pt"
                )
                model = PolySAFTModel()
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                loaded.append((f"{mode}_fold{f}_seed{seed}", model))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return loaded


def predict_polymer(
    smiles: str,
    mw_g_mol: float = 100000.0,
    temperatures_K: list[float] | None = None,
    pressures_MPa: list[float] | None = None,
    mode: str = "ensemble",
    fold: int | None = None,
    calibrate_point: tuple[float, float, float] | None = None,
) -> dict:
    """Predict effective PC-SAFT parameters and melt densities for a repeat-unit SMILES.

    Note: Ensemble density (`rho_mean_kg_m3`) is computed by solving the PC-SAFT liquid
    density root independently for each checkpoint and averaging the resulting mass
    densities across checkpoints; `parameters` reports summary mean and sample standard
    deviation of predicted `(m/M, sigma, eps/k_B)` across those checkpoints.
    """
    if mw_g_mol <= 0:
        raise ValueError(f"mw_g_mol must be strictly positive (in g/mol); got {mw_g_mol}.")
    if temperatures_K is None:
        temperatures_K = [450.0]
    if pressures_MPa is None:
        pressures_MPa = [0.101325]
    if any(t <= 0 for t in temperatures_K):
        raise ValueError(f"All temperatures_K must be strictly positive (in K); got {temperatures_K}.")
    if any(p <= 0 for p in pressures_MPa):
        raise ValueError(f"All pressures_MPa must be strictly positive absolute pressures (in MPa); got {pressures_MPa}.")
    if calibrate_point is not None:
        t_c, p_c, rho_c = calibrate_point
        if t_c <= 0 or p_c <= 0 or rho_c <= 0:
            raise ValueError(
                f"calibrate_point (T_K, P_MPa, rho_kg_m3) requires strictly positive temperature, absolute pressure, and density; got {calibrate_point}."
            )
    if len(pressures_MPa) == 1 and len(temperatures_K) > 1:
        pressures_MPa = pressures_MPa * len(temperatures_K)
    elif len(temperatures_K) == 1 and len(pressures_MPa) > 1:
        temperatures_K = temperatures_K * len(pressures_MPa)
    if len(temperatures_K) != len(pressures_MPa):
        raise ValueError("temperatures_K and pressures_MPa must have matching lengths or length 1.")

    graph = Batch.from_data_list([smiles_to_graph(smiles)])
    checkpoints = load_checkpoints(mode=mode, fold=fold)

    T_tensor = torch.tensor(temperatures_K, dtype=torch.float64)
    P_tensor = torch.tensor([p * 1e6 for p in pressures_MPa], dtype=torch.float64)

    per_run = []
    for run_id, model in checkpoints:
        with torch.no_grad():
            raw = model(graph)
            params = {k: float(v.item()) for k, v in raw.items()}

        sigma_used = params["sigma"]
        if calibrate_point is not None:
            t_cal, p_cal_mpa, rho_cal = calibrate_point
            cal_res = calibrate_sigma_single_point(
                params=params,
                mw_g_mol=mw_g_mol,
                T_K=t_cal,
                P_Pa=p_cal_mpa * 1e6,
                rho_obs_kg_m3=rho_cal,
            )
            sigma_used = cal_res["selected_sigma"]

        m_seg = torch.full_like(T_tensor, params["m_per_mw"] * mw_g_mol)
        sig_t = torch.full_like(T_tensor, sigma_used)
        eps_t = torch.full_like(T_tensor, params["eps_k"])
        rho_mol_m3, info = solve(m_seg, sig_t, eps_t, T_tensor, P_tensor, differentiable=False)
        if not bool(info["ok"].all()):
            raise RuntimeError(f"EOS root solver did not converge for {run_id}")
        rho_kg_m3 = (rho_mol_m3 * mw_g_mol / 1000.0).detach().numpy()

        per_run.append(
            {
                "run_id": run_id,
                "m_per_mw_mol_g": params["m_per_mw"],
                "sigma_prior_A": params["sigma"],
                "sigma_used_A": sigma_used,
                "eps_k_K": params["eps_k"],
                "densities_kg_m3": rho_kg_m3.tolist(),
            }
        )

    m_vals = np.array([r["m_per_mw_mol_g"] for r in per_run])
    sig_vals = np.array([r["sigma_used_A"] for r in per_run])
    eps_vals = np.array([r["eps_k_K"] for r in per_run])
    rho_mat = np.array([r["densities_kg_m3"] for r in per_run])

    states = []
    for idx, (t_k, p_mpa) in enumerate(zip(temperatures_K, pressures_MPa)):
        col = rho_mat[:, idx]
        states.append(
            {
                "T_K": float(t_k),
                "P_MPa": float(p_mpa),
                "rho_mean_kg_m3": float(col.mean()),
                "rho_std_kg_m3": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
            }
        )

    ddof = 1 if len(per_run) > 1 else 0
    return {
        "smiles": smiles,
        "mw_g_mol": float(mw_g_mol),
        "mode": mode,
        "n_checkpoints": len(per_run),
        "calibrated_single_point": (
            {
                "T_K": float(calibrate_point[0]),
                "P_MPa": float(calibrate_point[1]),
                "rho_obs_kg_m3": float(calibrate_point[2]),
            }
            if calibrate_point is not None
            else None
        ),
        "parameters": {
            "m_per_mw_mol_g": {"mean": float(m_vals.mean()), "std": float(m_vals.std(ddof=ddof))},
            "sigma_A": {"mean": float(sig_vals.mean()), "std": float(sig_vals.std(ddof=ddof))},
            "eps_k_K": {"mean": float(eps_vals.mean()), "std": float(eps_vals.std(ddof=ddof))},
        },
        "states": states,
        "per_checkpoint": per_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", type=str, required=True, help="Repeat-unit SMILES with '*', '[*]', or '[*:n]' attachment points, e.g. '*CC(C)*'")
    parser.add_argument("--mw", type=float, default=100000.0, help="Polymer chain molar mass in g/mol (default: 100000)")
    parser.add_argument("--T", type=float, nargs="+", default=[450.0], help="Temperature(s) in K")
    parser.add_argument("--P", type=float, nargs="+", default=[0.101325], help="Absolute pressure(s) in MPa (default: 0.101325 = 1 atm)")
    parser.add_argument("--mode", choices=["ensemble", "low_p", "multi_p"], default="ensemble")
    parser.add_argument("--fold", type=int, choices=range(5), default=None, help="Optional zero-based fold index (0-4)")
    parser.add_argument(
        "--calibrate-point",
        type=float,
        nargs=3,
        metavar=("T_K", "P_MPa", "RHO_KG_M3"),
        default=None,
        help="Optional single low-pressure experimental density point for frozen-network sigma calibration",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save full JSON prediction")
    args = parser.parse_args()

    cal_pt = tuple(args.calibrate_point) if args.calibrate_point is not None else None
    result = predict_polymer(
        smiles=args.smiles,
        mw_g_mol=args.mw,
        temperatures_K=args.T,
        pressures_MPa=args.P,
        mode=args.mode,
        fold=args.fold,
        calibrate_point=cal_pt,
    )
    summary = {k: v for k, v in result.items() if k != "per_checkpoint"}
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
