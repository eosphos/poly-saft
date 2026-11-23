"""Methyl-capped polymer repeat-unit RDKit preprocessing and atom/bond featurization."""
from __future__ import annotations

import re

from rdkit import Chem

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 6)),
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "S", "OTHER"],
    "is_aromatic": [False, True],
    "num_hs": list(range(0, 9)),
}

BOND_FEATURES = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "conjugated": [False, True],
    "in_ring": [False, True],
    "stereo": ["STEREONONE", "STEREOZ", "STEREOE", "OTHER"],
}

ATOM_FDIM = sum(len(values) + 1 for values in ATOM_FEATURES.values())
BOND_FDIM = sum(len(values) + 1 for values in BOND_FEATURES.values())


def one_hot(value, choices: list) -> list[int]:
    """Encode a categorical value with an additional slot for unknown categories."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else len(choices)
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.Atom) -> list[int]:
    """Return the one-hot atom feature vector."""
    features: list[int] = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    features += one_hot(str(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
    features += one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    return features


def bond_features(bond: Chem.Bond) -> list[int]:
    """Return the one-hot bond feature vector."""
    features: list[int] = []
    features += one_hot(str(bond.GetBondType()), BOND_FEATURES["bond_type"])
    features += one_hot(bond.GetIsConjugated(), BOND_FEATURES["conjugated"])
    features += one_hot(bond.IsInRing(), BOND_FEATURES["in_ring"])
    features += one_hot(str(bond.GetStereo()), BOND_FEATURES["stereo"])
    return features


def cap_polymerization_wildcards(smiles: str) -> str:
    """Replace polymerization attachment wildcards ('*', '[*]', '[*:n]') with methyl caps."""
    normalized = re.sub(r"\[\d*\*(?::\d+)?\]", "*", smiles)
    capped = re.sub(r"\*[=#]", "[CH3]", normalized)
    capped = re.sub(r"[=#]\*", "[CH3]", capped)
    return capped.replace("*", "[CH3]")


def polymer_smiles_to_mol(smiles: str) -> Chem.Mol:
    """Parse a polymer repeat-unit SMILES into an explicit-hydrogen methyl-capped RDKit molecule."""
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if sum(1 for atom in raw.GetAtoms() if atom.GetAtomicNum() == 0) < 1:
        raise ValueError(
            f"Repeat-unit SMILES must contain at least one polymerization attachment point ('*', '[*]', or '[*:n]'): {smiles}"
        )
    mol = Chem.MolFromSmiles(cap_polymerization_wildcards(smiles))
    if mol is None:
        raise ValueError(f"Invalid capped repeat-unit SMILES: {smiles}")
    return Chem.AddHs(mol)
