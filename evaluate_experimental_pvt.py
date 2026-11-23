"""Reproduce all experimental polymer melt PVT benchmarks reported in the PolySAFT paper.

Covers:
  1. Five-fold uncalibrated melt density evaluation, 15-model ensemble predictions,
     held-out structural test-fold comparisons, and fixed literature GC-PC-SAFT
     baselines across all 12 experimental melt PVT datasets (1,012 states; Table 7
     and Supplementary Table S8a).
  2. Cross-polymer shared-head Low-P and Multi-P calibration, global multiplicative
     density-scaling controls, and independent sample transfer on Folds 1 and 4
     (Table 8 and Supplementary Table S6).
  3. Frozen-network single-point and multi-point segment-diameter (sigma) calibration,
     closed-form density scaling, fixed literature GC and constant MD-prior controls,
     and isothermal compression-increment errors across the seven wide-pressure melts
     (742 full-range and 589 high-pressure states; Table 9, Supplementary Section S4.3,
     Supplementary Table S8b, and Figure 7c).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from model import PolySAFTModel, smiles_to_graph
from sigma_calibration import predict_eos_density

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def hierarchical_metric(frame: pd.DataFrame, col: str) -> float:
    """Hierarchical equal weighting across pressure levels, curves, and states (isotherms for PIB)."""
    if str(frame["polymer"].iloc[0]) == "PIB":
        return float(frame.groupby("temperature_K")[col].mean().mean())
    return float(
        frame.groupby(["nominal_pressure_MPa", "curve_id"])[col]
        .mean()
        .groupby(level=0)
        .mean()
        .mean()
    )


def calibrate_sigma_multi_point(
    params: dict[str, float],
    T_K: np.ndarray,
    P_Pa: np.ndarray,
    M_g_mol: np.ndarray,
    rho_obs_kg_m3: np.ndarray,
    bounds: tuple[float, float] = (2.5, 5.0),
) -> dict:
    """Calibrate segment diameter sigma in [2.5, 5.0] A on n >= 1 experimental liquid density points."""
    trials: list[tuple[float, float, bool]] = []

    def objective(sigma_val: float) -> float:
        trial_p = {
            "m_per_mw": float(params["m_per_mw"]),
            "sigma": float(sigma_val),
            "eps_k": float(params["eps_k"]),
        }
        rho, ok = predict_eos_density(trial_p, T_K, P_Pa, M_g_mol)
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


def load_model(ckpt_path: Path) -> PolySAFTModel:
    model = PolySAFTModel()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def predict_params_from_smiles(model: PolySAFTModel, smiles: str) -> dict[str, float]:
    graph = Batch.from_data_list([smiles_to_graph(smiles)])
    with torch.no_grad():
        raw = model(graph)
        return {k: float(v.item()) for k, v in raw.items()}


def load_gc_priors(exp_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    gc_df = pd.read_csv(exp_dir / "literature_gc_parameters.csv")
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in gc_df.iterrows():
        out[(str(r["prior_name"]), str(r["polymer"]))] = {
            "m_per_mw": float(r["m_per_mw_mol_g"]),
            "sigma": float(r["sigma_A"]),
            "eps_k": float(r["eps_k_K"]),
        }
    md_df = pd.read_csv(ROOT / "data/fitted_pcsaft_parameters_897.csv").set_index("polymer_id")
    fold0_train = json.loads((ROOT / "splits/fold0.json").read_text(encoding="utf-8"))["sets"]["train"]
    sub_tr = md_df.loc[fold0_train]
    out[("Constant_MD_Prior_Fold1", "ALL")] = {
        "m_per_mw": float(np.exp(np.mean(np.log(sub_tr["m_per_mw_mol_g"])))),
        "sigma": float(np.exp(np.mean(np.log(sub_tr["sigma_A"])))),
        "eps_k": float(np.exp(np.mean(np.log(sub_tr["eps_k_K"])))),
    }
    return out


def run_fivefold_uncalibrated_benchmark(
    sources_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    gc_priors: dict[tuple[str, str], dict[str, float]],
) -> tuple[list[dict], pd.DataFrame]:
    """Reproduce Table 7 and Supplementary Table S8(a) across all 12 datasets (1,012 states)."""
    models = {
        (f, s): load_model(ROOT / f"models/polysaft/runs/fold{f}/seed{s}/density.pt")
        for f in range(5)
        for s in range(3)
    }
    rows = []
    for _, meta in sources_df.iterrows():
        ds = str(meta["dataset_id"])
        pol = str(meta["polymer"])
        smiles = str(meta["repeat_smiles"])
        held_out_label = str(meta["md_held_out_fold"])
        sub = eval_df[eval_df["dataset_id"] == ds].copy()
        T = sub["temperature_K"].to_numpy()
        P = sub["pressure_abs_Pa"].to_numpy()
        M = sub["chain_mw_g_mol"].to_numpy()
        obs = sub["density_exp_kg_m3"].to_numpy()

        fold_mapes, fold_biases = [], []
        held_out_seed_mapes: list[float] = []
        all_15_densities = []

        held_out_zero_idx = (
            int(held_out_label.split()[-1]) - 1 if held_out_label.startswith("Fold ") else None
        )

        for f in range(5):
            seed_mapes, seed_biases = [], []
            for s in range(3):
                pars = predict_params_from_smiles(models[(f, s)], smiles)
                rho, ok = predict_eos_density(pars, T, P, M)
                if not bool(ok.all()):
                    raise RuntimeError(f"Root failure on {ds} fold{f}_seed{s}")
                all_15_densities.append(rho)
                tmp = sub.copy()
                tmp["ape"] = np.abs(rho / obs - 1.0) * 100.0
                tmp["signed_err"] = (rho / obs - 1.0) * 100.0
                m_val = hierarchical_metric(tmp, "ape")
                b_val = hierarchical_metric(tmp, "signed_err")
                seed_mapes.append(m_val)
                seed_biases.append(b_val)
                if held_out_zero_idx is not None and f == held_out_zero_idx:
                    held_out_seed_mapes.append(m_val)
            fold_mapes.append(float(np.mean(seed_mapes)))
            fold_biases.append(float(np.mean(seed_biases)))

        ens_rho = np.mean(np.stack(all_15_densities, axis=0), axis=0)
        tmp_ens = sub.copy()
        tmp_ens["ape"] = np.abs(ens_rho / obs - 1.0) * 100.0
        tmp_ens["signed_err"] = (ens_rho / obs - 1.0) * 100.0

        tihic_mape = None
        if ("Tihic_2008", pol) in gc_priors:
            rho_t, ok_t = predict_eos_density(gc_priors[("Tihic_2008", pol)], T, P, M)
            assert bool(ok_t.all())
            tmp_t = sub.copy()
            tmp_t["ape"] = np.abs(rho_t / obs - 1.0) * 100.0
            tihic_mape = hierarchical_metric(tmp_t, "ape")

        peters_mape = None
        if ("Peters_2012", pol) in gc_priors:
            rho_p, ok_p = predict_eos_density(gc_priors[("Peters_2012", pol)], T, P, M)
            assert bool(ok_p.all())
            tmp_p = sub.copy()
            tmp_p["ape"] = np.abs(rho_p / obs - 1.0) * 100.0
            peters_mape = hierarchical_metric(tmp_p, "ape")

        rows.append(
            {
                "dataset_id": ds,
                "polymer": pol,
                "sample_grade": str(meta["sample_grade"]),
                "citation_short": str(meta["citation_short"]),
                "n_states": int(len(sub)),
                "five_fold_mape_mean_percent": float(np.mean(fold_mapes)),
                "five_fold_mape_std_percent": float(np.std(fold_mapes, ddof=1)),
                "ensemble_15_mape_percent": hierarchical_metric(tmp_ens, "ape"),
                "five_fold_signed_bias_mean_percent": float(np.mean(fold_biases)),
                "five_fold_signed_bias_std_percent": float(np.std(fold_biases, ddof=1)),
                "held_out_fold": held_out_label,
                "held_out_fold_mape_mean_percent": (
                    float(np.mean(held_out_seed_mapes)) if held_out_seed_mapes else None
                ),
                "held_out_fold_mape_std_percent": (
                    float(np.std(held_out_seed_mapes, ddof=1)) if held_out_seed_mapes else None
                ),
                "tihic_gc_mape_percent": tihic_mape,
                "peters_gc_mape_percent": peters_mape,
            }
        )
    return rows, pd.DataFrame(rows)


def balanced_log_scale(fit_df: pd.DataFrame, model: PolySAFTModel) -> float:
    """Compute equal-polymer hierarchical mean log(rho_obs / rho_pred) across the 139 fit points."""
    pol_deltas = []
    for pol in ["PP", "PVAc", "PCL", "PC"]:
        sub = fit_df[fit_df["polymer"] == pol].copy()
        smiles = str(sub["repeat_smiles"].iloc[0])
        pars = predict_params_from_smiles(model, smiles)
        rho, ok = predict_eos_density(
            pars,
            sub["temperature_K"].to_numpy(),
            sub["pressure_absolute_Pa"].to_numpy(),
            sub["chain_molar_mass_g_mol"].to_numpy(),
        )
        assert bool(ok.all())
        sub["log_ratio"] = np.log(sub["density_kg_m3"].to_numpy() / rho)
        pol_deltas.append(hierarchical_metric(sub, "log_ratio"))
    return float(np.exp(np.mean(pol_deltas)))


def run_shared_calibration_benchmark(
    exp_dir: Path,
    eval_df: pd.DataFrame,
    gc_priors: dict[tuple[str, str], dict[str, float]],
) -> dict:
    """Reproduce Table 8 and Supplementary Table S6 across Folds 1 and 4 (zero-based 0 and 3)."""
    low_fit = pd.read_csv(exp_dir / "shared_calibration/calibration_low_pressure_fit.csv")
    multi_fit = pd.read_csv(exp_dir / "shared_calibration/calibration_multi_pressure_fit.csv")
    hp_test = pd.read_csv(exp_dir / "shared_calibration/calibration_high_pressure_test.csv")
    cal_pols = ["PP", "PVAc", "PCL", "PC"]
    transfer_ds = ["HDPE", "PS_Chang2023", "PS_Park1999"]

    regimes = [
        ("uncalibrated", lambda f, s: ROOT / f"models/polysaft/runs/fold{f}/seed{s}/density.pt"),
        (
            "low_p_calibrated",
            lambda f, s: ROOT
            / f"models/experimental_calibration/runs/network_low_pressure_fold{f}_seed{s}/best.pt",
        ),
        (
            "multi_p_calibrated",
            lambda f, s: ROOT
            / f"models/experimental_calibration/runs/network_multi_pressure_fold{f}_seed{s}/best.pt",
        ),
    ]

    hp_results: dict[str, dict] = {}
    transfer_results: dict[str, dict] = {ds: {} for ds in transfer_ds}

    for regime_name, path_fn in regimes:
        per_fold_pol: dict[int, dict[str, list[float]]] = {0: {p: [] for p in cal_pols}, 3: {p: [] for p in cal_pols}}
        per_fold_equal: dict[int, list[float]] = {0: [], 3: []}
        all_6_pol: dict[str, list[float]] = {p: [] for p in cal_pols}
        all_6_equal: list[float] = []
        tr_runs: dict[str, list[float]] = {ds: [] for ds in transfer_ds}

        for f in [0, 3]:
            for s in range(3):
                model = load_model(path_fn(f, s))
                pol_vals = []
                for pol in cal_pols:
                    sub = hp_test[hp_test["polymer"] == pol].copy()
                    pars = predict_params_from_smiles(model, str(sub["repeat_smiles"].iloc[0]))
                    rho, ok = predict_eos_density(
                        pars,
                        sub["temperature_K"].to_numpy(),
                        sub["pressure_absolute_Pa"].to_numpy(),
                        sub["chain_molar_mass_g_mol"].to_numpy(),
                    )
                    assert bool(ok.all())
                    sub["ape"] = np.abs(rho / sub["density_kg_m3"].to_numpy() - 1.0) * 100.0
                    m_val = hierarchical_metric(sub, "ape")
                    per_fold_pol[f][pol].append(m_val)
                    all_6_pol[pol].append(m_val)
                    pol_vals.append(m_val)
                eq_mean = float(np.mean(pol_vals))
                per_fold_equal[f].append(eq_mean)
                all_6_equal.append(eq_mean)

                for ds in transfer_ds:
                    sub_t = eval_df[eval_df["dataset_id"] == ds].copy()
                    pars_t = predict_params_from_smiles(model, str(sub_t["repeat_smiles"].iloc[0]))
                    rho_t, ok_t = predict_eos_density(
                        pars_t,
                        sub_t["temperature_K"].to_numpy(),
                        sub_t["pressure_abs_Pa"].to_numpy(),
                        sub_t["chain_mw_g_mol"].to_numpy(),
                    )
                    assert bool(ok_t.all())
                    sub_t["ape"] = np.abs(rho_t / sub_t["density_exp_kg_m3"].to_numpy() - 1.0) * 100.0
                    tr_runs[ds].append(hierarchical_metric(sub_t, "ape"))

        hp_results[regime_name] = {
            "two_fold_6run_equal_polymer_mape": {
                "mean": float(np.mean(all_6_equal)),
                "std": float(np.std(all_6_equal, ddof=1)),
            },
            "two_fold_6run_per_polymer_mape": {
                p: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                for p, v in all_6_pol.items()
            },
            "fold1_3seed": {
                "equal_polymer_mape": {
                    "mean": float(np.mean(per_fold_equal[0])),
                    "std": float(np.std(per_fold_equal[0], ddof=1)),
                },
                "per_polymer_mape": {
                    p: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                    for p, v in per_fold_pol[0].items()
                },
            },
            "fold4_3seed": {
                "equal_polymer_mape": {
                    "mean": float(np.mean(per_fold_equal[3])),
                    "std": float(np.std(per_fold_equal[3], ddof=1)),
                },
                "per_polymer_mape": {
                    p: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                    for p, v in per_fold_pol[3].items()
                },
            },
        }
        for ds in transfer_ds:
            transfer_results[ds][regime_name] = {
                "mean": float(np.mean(tr_runs[ds])),
                "std": float(np.std(tr_runs[ds], ddof=1)),
            }

    # Global multiplicative density scaling controls (Table S6 bottom row)
    global_scale_summary = {}
    for label, fit_df in [("low_p_global_scaling", low_fit), ("multi_p_global_scaling", multi_fit)]:
        run_means = []
        for f in [0, 3]:
            for s in range(3):
                base_model = load_model(ROOT / f"models/polysaft/runs/fold{f}/seed{s}/density.pt")
                scale_factor = balanced_log_scale(fit_df, base_model)
                pol_vals = []
                for pol in cal_pols:
                    sub = hp_test[hp_test["polymer"] == pol].copy()
                    pars = predict_params_from_smiles(base_model, str(sub["repeat_smiles"].iloc[0]))
                    rho, ok = predict_eos_density(
                        pars,
                        sub["temperature_K"].to_numpy(),
                        sub["pressure_absolute_Pa"].to_numpy(),
                        sub["chain_molar_mass_g_mol"].to_numpy(),
                    )
                    assert bool(ok.all())
                    sub["ape"] = np.abs((rho * scale_factor) / sub["density_kg_m3"].to_numpy() - 1.0) * 100.0
                    pol_vals.append(hierarchical_metric(sub, "ape"))
                run_means.append(float(np.mean(pol_vals)))
        global_scale_summary[label] = {
            "mean": float(np.mean(run_means)),
            "std": float(np.std(run_means, ddof=1)),
        }

    # Fixed GC on high-P calibration test states (Table 8 right column)
    hp_gc = {}
    for prior_name, pol in [("Tihic_2008", "PP"), ("Peters_2012", "PP"), ("Tihic_2008", "PVAc")]:
        sub = hp_test[hp_test["polymer"] == pol].copy()
        rho, ok = predict_eos_density(
            gc_priors[(prior_name, pol)],
            sub["temperature_K"].to_numpy(),
            sub["pressure_absolute_Pa"].to_numpy(),
            sub["chain_molar_mass_g_mol"].to_numpy(),
        )
        assert bool(ok.all())
        sub["ape"] = np.abs(rho / sub["density_kg_m3"].to_numpy() - 1.0) * 100.0
        hp_gc[f"{pol}_{prior_name}"] = hierarchical_metric(sub, "ape")

    return {
        "calibrated_melts_high_pressure_299_states": hp_results,
        "global_multiplicative_scaling_controls_299_states": global_scale_summary,
        "fixed_gc_high_pressure_299_states": hp_gc,
        "independent_sample_transfer_6runs": transfer_results,
    }


def evaluate_single_prior_sigma_and_scale(
    params: dict[str, float],
    query_sub: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    n_budget: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate uncalibrated, n-point sigma calibration, and n-point density scaling on one polymer."""
    anc = anchor_sub[anchor_sub["candidate_rank"] <= n_budget]
    T_a = anc["temperature_K"].to_numpy()
    P_a = anc["pressure_abs_Pa"].to_numpy()
    M_a = anc["chain_mw_g_mol"].to_numpy()
    obs_a = anc["density_exp_kg_m3"].to_numpy()

    T_q = query_sub["temperature_K"].to_numpy()
    P_q = query_sub["pressure_abs_Pa"].to_numpy()
    M_q = query_sub["chain_mw_g_mol"].to_numpy()
    obs_q = query_sub["density_exp_kg_m3"].to_numpy()

    rho_q0, ok_q0 = predict_eos_density(params, T_q, P_q, M_q)
    rho_a0, ok_a0 = predict_eos_density(params, T_a, P_a, M_a)
    assert bool(ok_q0.all() and ok_a0.all())

    scale_factor = float(np.exp(np.mean(np.log(obs_a / rho_a0))))
    rho_q_scale = rho_q0 * scale_factor

    cal = calibrate_sigma_multi_point(params, T_a, P_a, M_a, obs_a)
    rho_q_sigma, ok_qs = predict_eos_density(dict(params, sigma=cal["selected_sigma"]), T_q, P_q, M_q)
    assert bool(ok_qs.all())

    q_out = query_sub.copy()
    q_out["rho_uncal"] = rho_q0
    q_out["rho_sigma"] = rho_q_sigma
    q_out["rho_scale"] = rho_q_scale
    q_out["ape_uncal"] = np.abs(rho_q0 / obs_q - 1.0) * 100.0
    q_out["ape_sigma"] = np.abs(rho_q_sigma / obs_q - 1.0) * 100.0
    q_out["ape_scale"] = np.abs(rho_q_scale / obs_q - 1.0) * 100.0

    q_hp = q_out[q_out["in_sigma_high_pressure_extrapolation"]]
    metrics = {
        "sigma_prior_A": float(params["sigma"]),
        "sigma_calibrated_A": float(cal["selected_sigma"]),
        "full_uncal": hierarchical_metric(q_out, "ape_uncal"),
        "full_sigma": hierarchical_metric(q_out, "ape_sigma"),
        "full_scale": hierarchical_metric(q_out, "ape_scale"),
        "hp_uncal": hierarchical_metric(q_hp, "ape_uncal"),
        "hp_sigma": hierarchical_metric(q_hp, "ape_sigma"),
        "hp_scale": hierarchical_metric(q_hp, "ape_scale"),
    }
    return metrics, q_out


def run_sigma_calibration_benchmark(
    exp_dir: Path,
    eval_df: pd.DataFrame,
    gc_priors: dict[tuple[str, str], dict[str, float]],
) -> tuple[dict, pd.DataFrame]:
    """Reproduce Table 9, Supplementary Table S8(b), SI Section S4.3, and Figure 7(c)."""
    anchors = pd.read_csv(exp_dir / "single_point_sigma_anchors.csv")
    pairs = pd.read_csv(exp_dir / "compression_endpoint_pairs.csv")
    wide = eval_df[eval_df["in_wide_pressure_set"]].copy()
    polymers = ["PP", "PVAc", "PCL", "PC", "PES", "PMMA", "PIB"]

    fold1_models = [
        load_model(ROOT / f"models/polysaft/runs/fold0/seed{s}/density.pt") for s in range(3)
    ]

    seed_records = []
    pair_seed_records = []
    multipt_records = []

    for s, model in enumerate(fold1_models):
        for pol in polymers:
            q_sub = wide[wide["dataset_id"] == pol]
            a_sub = anchors[anchors["dataset_id"] == pol]
            pars = predict_params_from_smiles(model, str(q_sub["repeat_smiles"].iloc[0]))
            m1, q_pred = evaluate_single_prior_sigma_and_scale(pars, q_sub, a_sub, n_budget=1)
            seed_records.append({"seed": s, "polymer": pol, "prior": "PolySAFT", **m1})

            # Isothermal compression increment |Delta c| (Figure 7c)
            p_sub = pairs[pairs["dataset_id"] == pol]
            if len(p_sub) > 0:
                idx_map = q_pred.set_index("state_id")
                err_uncal, err_sigma = [], []
                for _, pr in p_sub.iterrows():
                    lo = idx_map.loc[str(pr["low_state_id"])]
                    hi = idx_map.loc[str(pr["high_state_id"])]
                    c_obs = float(np.log(hi["density_exp_kg_m3"] / lo["density_exp_kg_m3"]))
                    c_u = float(np.log(hi["rho_uncal"] / lo["rho_uncal"]))
                    c_s = float(np.log(hi["rho_sigma"] / lo["rho_sigma"]))
                    err_uncal.append(abs(c_u - c_obs) * 100.0)
                    err_sigma.append(abs(c_s - c_obs) * 100.0)
                pair_seed_records.append(
                    {
                        "seed": s,
                        "polymer": pol,
                        "n_pairs": int(len(p_sub)),
                        "delta_c_uncal_pp": float(np.mean(err_uncal)),
                        "delta_c_sigma_pp": float(np.mean(err_sigma)),
                    }
                )

            if pol in ("PCL", "PIB"):
                for nb in (2, 4):
                    m_nb, _ = evaluate_single_prior_sigma_and_scale(pars, q_sub, a_sub, n_budget=nb)
                    multipt_records.append({"seed": s, "polymer": pol, "n_anchors": nb, **m_nb})

    df_seeds = pd.DataFrame(seed_records)
    per_pol_summary = {}
    for pol in polymers:
        sub = df_seeds[df_seeds["polymer"] == pol]
        per_pol_summary[pol] = {
            k: {"mean": float(sub[k].mean()), "std": float(sub[k].std(ddof=1))}
            for k in ["full_uncal", "full_sigma", "full_scale", "hp_uncal", "hp_sigma", "hp_scale"]
        }

    def cohort_poly_summary(pol_list: list[str]) -> dict:
        by_seed = df_seeds[df_seeds["polymer"].isin(pol_list)].groupby("seed")[
            ["full_uncal", "full_sigma", "full_scale", "hp_uncal", "hp_sigma", "hp_scale"]
        ].mean()
        return {
            k: {"mean": float(by_seed[k].mean()), "std": float(by_seed[k].std(ddof=1))}
            for k in by_seed.columns
        }

    # Deterministic priors: Tihic, Peters, and Fold-1 Constant MD Prior (Table 9 & Table S8b)
    det_rows = []
    for prior_name, pol in [
        ("Tihic_2008", "PP"),
        ("Peters_2012", "PP"),
        ("Tihic_2008", "PVAc"),
        ("Tihic_2008", "PMMA"),
        ("Peters_2012", "PMMA"),
        ("Tihic_2008", "PIB"),
        ("Peters_2012", "PIB"),
    ]:
        m_det, _ = evaluate_single_prior_sigma_and_scale(
            gc_priors[(prior_name, pol)],
            wide[wide["dataset_id"] == pol],
            anchors[anchors["dataset_id"] == pol],
            n_budget=1,
        )
        det_rows.append({"prior": prior_name, "polymer": pol, **m_det})

    const_pars = gc_priors[("Constant_MD_Prior_Fold1", "ALL")]
    for pol in polymers:
        m_c, _ = evaluate_single_prior_sigma_and_scale(
            const_pars,
            wide[wide["dataset_id"] == pol],
            anchors[anchors["dataset_id"] == pol],
            n_budget=1,
        )
        det_rows.append({"prior": "Constant_MD_Prior_Fold1", "polymer": pol, **m_c})

    df_det = pd.DataFrame(det_rows)
    cols_6 = ["full_uncal", "full_sigma", "full_scale", "hp_uncal", "hp_sigma", "hp_scale"]

    def det_cohort(prior_name: str, pol_list: list[str]) -> dict:
        sub = df_det[(df_det["prior"] == prior_name) & (df_det["polymer"].isin(pol_list))]
        return {k: float(sub[k].mean()) for k in cols_6}

    df_pairs_seed = pd.DataFrame(pair_seed_records)
    pair_summary = {
        pol: {
            "n_pairs": int(g["n_pairs"].iloc[0]),
            "uncalibrated_mean_pp": float(g["delta_c_uncal_pp"].mean()),
            "sigma_1pt_mean_pp": float(g["delta_c_sigma_pp"].mean()),
        }
        for pol, g in df_pairs_seed.groupby("polymer")
    }

    df_mp = pd.DataFrame(multipt_records)
    multipt_summary = {
        f"{pol}_n{nb}": {
            "full_sigma_mean": float(g["full_sigma"].mean()),
            "full_sigma_std": float(g["full_sigma"].std(ddof=1)),
        }
        for (pol, nb), g in df_mp.groupby(["polymer", "n_anchors"])
    }

    summary = {
        "n_polymers": len(polymers),
        "n_full_range_states": int(len(wide)),
        "n_high_pressure_states": int(wide["in_sigma_high_pressure_extrapolation"].sum()),
        "polysaft_fold1_per_polymer": per_pol_summary,
        "cohorts": {
            "all_seven_melts_polysaft": cohort_poly_summary(polymers),
            "all_seven_melts_constant_md_prior": det_cohort("Constant_MD_Prior_Fold1", polymers),
            "md_tihic_subset_PP_PVAc_PMMA_polysaft": cohort_poly_summary(["PP", "PVAc", "PMMA"]),
            "md_tihic_subset_PP_PVAc_PMMA_tihic_gc": det_cohort("Tihic_2008", ["PP", "PVAc", "PMMA"]),
            "md_peters_subset_PP_PMMA_polysaft": cohort_poly_summary(["PP", "PMMA"]),
            "md_peters_subset_PP_PMMA_peters_gc": det_cohort("Peters_2012", ["PP", "PMMA"]),
            "md_peters_subset_PP_PMMA_tihic_gc": det_cohort("Tihic_2008", ["PP", "PMMA"]),
            "md_peters_subset_PP_PMMA_constant_md_prior": det_cohort(
                "Constant_MD_Prior_Fold1", ["PP", "PMMA"]
            ),
        },
        "deterministic_per_polymer": df_det[df_det["prior"] != "Constant_MD_Prior_Fold1"].to_dict(
            orient="records"
        ),
        "compression_increment_abs_error_pp": pair_summary,
        "multi_point_anchor_checks": multipt_summary,
    }
    return summary, df_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional directory to write detailed CSV and JSON benchmark tables",
    )
    args = parser.parse_args()
    dest = args.output.resolve() if args.output is not None else None

    exp_dir = ROOT / "data/experimental_pvt"
    sources_df = pd.read_csv(exp_dir / "experimental_sources.csv")
    eval_df = pd.read_csv(exp_dir / "experimental_melt_pvt_evaluation.csv")
    gc_priors = load_gc_priors(exp_dir)

    table7_rows, df_table7 = run_fivefold_uncalibrated_benchmark(sources_df, eval_df, gc_priors)
    table8_summary = run_shared_calibration_benchmark(exp_dir, eval_df, gc_priors)
    table9_summary, df_sigma_seeds = run_sigma_calibration_benchmark(exp_dir, eval_df, gc_priors)

    full_summary = {
        "table7_and_s8a_fivefold_uncalibrated": table7_rows,
        "table8_and_s6_shared_head_calibration": table8_summary,
        "table9_s8b_and_fig7c_sigma_calibration": table9_summary,
    }

    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
        df_table7.to_csv(dest / "table7_s8a_uncalibrated_12datasets.csv", index=False)
        df_sigma_seeds.to_csv(dest / "table9_sigma_calibration_by_seed.csv", index=False)
        (dest / "experimental_benchmarks_summary.json").write_text(
            json.dumps(full_summary, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
