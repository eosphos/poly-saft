# PolySAFT

**PolySAFT** predicts pure-component PC-SAFT equation-of-state parameters (`m/M`, `sigma`, `eps/k_B`) and polymer melt densities directly from a polymer repeat-unit SMILES and chain molar mass. It combines a message-passing graph neural network with an implicitly differentiable PC-SAFT solver, and supports optional single-point `sigma` calibration when one low-pressure experimental density measurement is available for a target polymer sample.

> **PolySAFT: Learning Effective Polymer Equations of State from Molecular Structure through Differentiable Perturbed-Chain Statistical Associating Fluid Theory**

## Installation

Requires Python >= 3.10. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command-Line Usage

```bash
# 1. Predict PC-SAFT parameters and melt densities from a repeat-unit SMILES
python predict.py --smiles "*CC(C)*" --mw 100000 --T 473.15 513.15 --P 0.101325 50.0

# 2. Predict high-pressure melt density using single-point sigma calibration
#    from one low-pressure density measurement: --calibrate-point T_K P_MPa RHO_KG_M3
python predict.py --smiles "*CC(OC(=O)C)*" --mw 100000 --T 373.15 --P 80.0 --calibrate-point 333.15 0.1 1161.0

# 3. Reproduce all experimental polymer melt PVT benchmarks in the paper
python evaluate_experimental_pvt.py --output results_experimental_pvt
```

### Python API

```python
from predict import predict_polymer

result = predict_polymer(
    smiles="*CC(C)*",
    mw_g_mol=100000.0,
    temperatures_K=[473.15, 513.15],
    pressures_MPa=[0.101325, 50.0],
)
print(result["parameters"])
print(result["states"])
```

## Input and Output Conventions

- **`--smiles`**: Repeat-unit SMILES containing polymerization attachment points `*`, `[*]`, or `[*:n]`, such as `*CC(C)*` for polypropylene.
- **`--mw`**: Polymer chain molar mass in `g/mol` (default: `100000.0`).
- **`--T`**: Temperature(s) in `K`.
- **`--P`**: Absolute pressure(s) in `MPa` (default: `0.101325` MPa = 1 atm; convert gauge pressures to absolute before input).
- **`--mode`**: `ensemble` (default 15-checkpoint ensemble), `low_p`, or `multi_p` (shared experimental calibration checkpoints). Pass `--output result.json` to save full per-checkpoint outputs.
- **Outputs**: Specific segment number `m/M` (`mol/g`), segment diameter `sigma` (`Å`), dispersion energy `eps/k_B` (`K`), and liquid mass density `rho` (`kg/m^3`).

## Datasets and Literature Sources

- `data/fitted_pcsaft_parameters_897.csv` and `splits/`: Repeat-unit SMILES, chain molar masses, fitted 1-atm PC-SAFT parameters, and five-fold structure-cluster splits for all 897 amorphous homopolymers (`POLY_001`–`POLY_897`).
- `data/experimental_pvt/`: High-temperature melt and equilibrium liquid PVT datasets compiled from the literature (`experimental_sources.csv` and `experimental_melt_pvt_evaluation.csv`), low-pressure `sigma` calibration anchors (`single_point_sigma_anchors.csv`), cross-polymer calibration splits (`shared_calibration/`), pressure-response diagnostic subsets (`compression_endpoint_pairs.csv` and `pp_identifiability_diagnostic.csv`), and published polymer GC-PC-SAFT parameters (`literature_gc_parameters.csv`).

| Polymer | Sample Grade | Reference | DOI / Link |
|---|---|---|---|
| PP | SABIC PP505P | Wang, J. (2019). *Mendeley Data*, v1. | https://doi.org/10.17632/5skzzf2r6b.1 |
| PECH | Scientific Polymer Products | Rodgers, P. A. (1993). *J. Appl. Polym. Sci.*, 48(6), 1061–1080. | https://doi.org/10.1002/app.1993.070480613 |
| PMPS | Max Planck Institute sample | Paluch, M. et al. (2003). *Phys. Rev. E*, 68(3), 031802. | https://doi.org/10.1103/PhysRevE.68.031802 |
| HDPE | J-REX HD KF251A | Lin, L., Joe, M., & Park, H. E. (2026). *Transp. Phenom.*, 1(1), 20260007. | https://doi.org/10.1515/tp-2026-0007 |
| PCL, PC, PES | Capa 6500; Lexan 141; SUMIKAEXCEL 4100P | Pionteck, J. (2018). *Polymers*, 10(6), 578. | https://doi.org/10.3390/polym10060578 |
| PS | POLYREX PG-33 | Chang, Y.-J. (2023). Ph.D. Dissertation, University of Wisconsin–Madison. | https://asset.library.wisc.edu/1711.dl/7EYNDYBEBXM4N8Q/R/file-cd05e.pdf |
| PIB | Vistanex LM | Beret, S., & Prausnitz, J. M. (1975). *Macromolecules*, 8(4), 536–538; Eichinger, B. E., & Flory, P. J. (1968). *Macromolecules*, 1(3), 285–286. | https://doi.org/10.1021/ma60046a028 ; https://doi.org/10.1021/ma60003a016 |
| PVAc | Union Carbide AYAT | McKinney, J. E., & Goldstein, M. (1974). *J. Res. Natl. Bur. Stand., Sect. A*, 78A(3), 331–353. | https://doi.org/10.6028/jres.078A.018 |
| PS | GPPS G440K | Park, S. S. (1999). M.A.Sc. Thesis, University of Toronto. | https://www.collectionscanada.ca/obj/s4/f2/dsk1/tape8/PQDD_0003/MQ46081.pdf |
| PMMA | Polysciences sample | Walsh, D. J. et al. (1992). *Polymer*, 33(22), 4793–4797. | https://doi.org/10.1016/0032-3861(92)90694-R |
| GC-PC-SAFT | Published polymer parameters (Table 5) | Tihic, A. et al. (2008). *Ind. Eng. Chem. Res.*, 47(15), 5092–5101. | https://doi.org/10.1021/ie0710768 |
| GC-PC-SAFT | Published polymer parameters (Table 2) | Peters, F. T., Laube, F. S., & Sadowski, G. (2012). *Fluid Phase Equilib.*, 324, 70–79. | https://doi.org/10.1016/j.fluid.2012.03.009 |

## License

Released under the MIT License (`LICENSE`).
