"""Differentiable pure-component PC-SAFT equation of state and liquid root solver.

Implements the non-associating pure-component Perturbed-Chain Statistical
Associating Fluid Theory (Gross & Sadowski, Ind. Eng. Chem. Res. 2001, 40,
1244-1260) in float64 PyTorch, with safeguarded Newton-bisection root finding
and Implicit Function Theorem backward differentiation.
"""
from __future__ import annotations

import torch
from torch import Tensor

N_A = 6.02214076e23
K_B = 1.380649e-23
R_GAS = N_A * K_B
PI = torch.pi

A_MAT = torch.tensor(
    [
        [
            0.9105631445,
            0.6361281449,
            2.6861347891,
            -26.547362491,
            97.759208784,
            -159.59154087,
            91.297774084,
        ],
        [
            -0.3084016918,
            0.1860531159,
            -2.5030047259,
            21.419793629,
            -65.255885330,
            83.318680481,
            -33.746922930,
        ],
        [
            -0.0906148351,
            0.4527842806,
            0.5962700728,
            -1.7241829131,
            -4.1302112531,
            13.776631870,
            -8.6728470368,
        ],
    ],
    dtype=torch.float64,
)

B_MAT = torch.tensor(
    [
        [
            0.7240946941,
            2.2382791861,
            -4.0025849485,
            -21.003576815,
            26.855641363,
            206.55133841,
            -355.60235612,
        ],
        [
            -0.5755498075,
            0.6995095521,
            3.8925673390,
            -17.215471648,
            192.67226447,
            -161.82646165,
            -165.20769346,
        ],
        [
            0.0976883116,
            -0.2557574982,
            -9.1558561530,
            20.642075974,
            -38.804430052,
            93.626774077,
            -29.666905585,
        ],
    ],
    dtype=torch.float64,
)


def packing_coefficient(m: Tensor, sigma: Tensor, eps: Tensor, temperature: Tensor) -> Tensor:
    """Return molar packing coefficient C(T) = (pi/6) * N_A * m * d(T)^3 [m^3/mol]."""
    d = sigma * (1.0 - 0.12 * torch.exp(-3.0 * eps / temperature)) * 1e-10
    return PI / 6.0 * N_A * m * d.pow(3)


def helmholtz(m: Tensor, sigma: Tensor, eps: Tensor, temperature: Tensor, rho: Tensor) -> Tensor:
    """Compute reduced residual Helmholtz free energy a^res / (R * T) for pure components."""
    eta = packing_coefficient(m, sigma, eps, temperature) * rho
    if bool(torch.any((eta <= 0.0) | (eta >= 0.57))):
        raise ValueError("PC-SAFT evaluation outside valid packing interval (0, 0.57)")
    hs = (4.0 * eta - 3.0 * eta.square()) / (1.0 - eta).square()
    g = (1.0 - eta / 2.0) / (1.0 - eta).pow(3)
    hc = m * hs - (m - 1.0) * torch.log(g)
    f1 = (m - 1.0) / m
    f2 = f1 * (m - 2.0) / m
    a = A_MAT.to(m)
    b = B_MAT.to(m)
    ac = a[0] + f1[..., None] * a[1] + f2[..., None] * a[2]
    bc = b[0] + f1[..., None] * b[1] + f2[..., None] * b[2]
    powers = eta[..., None] ** torch.arange(7, device=m.device, dtype=m.dtype)
    i1, i2 = (ac * powers).sum(-1), (bc * powers).sum(-1)
    c1 = 1.0 / (
        1.0
        + m * (8.0 * eta - 2.0 * eta.square()) / (1.0 - eta).pow(4)
        + (1.0 - m)
        * (20.0 * eta - 27.0 * eta.square() + 12.0 * eta.pow(3) - 2.0 * eta.pow(4))
        / ((1.0 - eta) * (2.0 - eta)).square()
    )
    ms3 = m.square() * (sigma * 1e-10).pow(3)
    disp = -PI * rho * N_A * ms3 * (
        2.0 * i1 * eps / temperature + m * c1 * i2 * (eps / temperature).square()
    )
    return hc + disp


def pressure(m: Tensor, sigma: Tensor, eps: Tensor, temperature: Tensor, rho: Tensor) -> Tensor:
    """Evaluate pressure P [Pa] via exact autograd differentiation of a^res with respect to rho."""
    with torch.enable_grad():
        if not rho.requires_grad:
            rho = rho.detach().requires_grad_(True)
        a = helmholtz(m, sigma, eps, temperature, rho)
        da = torch.autograd.grad(a.sum(), rho, create_graph=True)[0]
        return rho * R_GAS * temperature * (1.0 + rho * da)


def pressure_and_slope(
    m: Tensor, sigma: Tensor, eps: Tensor, temperature: Tensor, rho: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate pressure P [Pa] and isothermal derivative dP/drho [Pa*m^3/mol]."""
    with torch.enable_grad():
        if not rho.requires_grad:
            rho = rho.detach().requires_grad_(True)
        p = pressure(m, sigma, eps, temperature, rho)
        slope = torch.autograd.grad(p.sum(), rho, create_graph=True)[0]
    return p, slope


def solve(
    m: Tensor,
    sigma: Tensor,
    eps: Tensor,
    temperature: Tensor,
    target: Tensor,
    differentiable: bool = False,
    max_iter: int = 80,
) -> tuple[Tensor, dict]:
    """Solve for the condensed-phase molar density root rho* [mol/m^3] below eta = 0.57.

    When differentiable=True, attaches exact first-order Implicit Function Theorem
    gradients drho*/d(m, sigma, eps, T, P) at the converged root.
    """
    md, sd, ed, td, pd = [x.detach() for x in (m, sigma, eps, temperature, target)]
    c_pack = packing_coefficient(md, sd, ed, td)
    lo = 0.25 / c_pack
    hi = 0.569999 / c_pack
    r = 0.50 / c_pack

    def calc(x: Tensor) -> tuple[Tensor, Tensor]:
        p_val, d_val = pressure_and_slope(md, sd, ed, td, x)
        return p_val.detach(), d_val.detach()

    pl, _ = calc(lo)
    ph, _ = calc(hi)
    bracket = (pl < pd) & (ph > pd)

    if not bool(bracket.all()):
        previous_r = hi.clone()
        previous_f = ph - pd
        for eta in torch.linspace(0.56, 0.005, 113, dtype=md.dtype, device=md.device):
            scan_r = eta / c_pack
            scan_p, _ = calc(scan_r)
            scan_f = scan_p - pd
            found = (~bracket) & (previous_f > 0.0) & (scan_f < 0.0)
            lo = torch.where(found, scan_r, lo)
            hi = torch.where(found, previous_r, hi)
            bracket |= found
            previous_r = scan_r
            previous_f = scan_f
        r = torch.minimum(torch.maximum(r, lo), hi)

    tolerance = torch.maximum(pd.abs() * 1e-9, torch.full_like(pd, 1e-4))
    iterations = 0
    for it in range(max_iter):
        p, d = calc(r)
        f = p - pd
        converged = (f.abs() <= tolerance) & bracket & torch.isfinite(p) & (d > 0.0)
        iterations = it + 1
        if bool(converged.all()):
            break
        lo = torch.where((f < 0.0) & (~converged), r, lo)
        hi = torch.where((f >= 0.0) & (~converged), r, hi)
        nr = r - f / d
        safe = (nr > lo) & (nr < hi) & torch.isfinite(nr) & (d > 0.0)
        proposal = torch.where(safe, nr, 0.5 * (lo + hi))
        r = torch.where(converged, r, proposal).detach()

    p, d = calc(r)
    eta = c_pack * r
    ok = (
        bracket
        & torch.isfinite(r)
        & (r > 0.0)
        & torch.isfinite(p)
        & (d > 0.0)
        & ((p - pd).abs() <= tolerance)
        & (eta < 0.57)
    )
    info = {
        "ok": ok,
        "pressure": p,
        "slope": d,
        "eta": eta,
        "relative_residual": (p - pd).abs() / pd.abs().clamp_min(1.0),
        "iterations": iterations,
    }
    if differentiable:
        if not bool(ok.all()):
            raise RuntimeError(
                f"Implicit differentiation rejected {int((~ok).sum())} non-converged roots"
            )
        live = pressure(m, sigma, eps, temperature, r) - target
        r = r - (live - live.detach()) / d
    return r, info
