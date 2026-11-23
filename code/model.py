"""PolySAFT four-layer attention MPNN encoder and bounded PC-SAFT parameter head."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool

from graph_representation import (
    ATOM_FDIM,
    BOND_FDIM,
    atom_features,
    bond_features,
    polymer_smiles_to_mol,
)

PARAMETER_BOUNDS = {
    "m_per_mw": (0.008, 0.052),
    "sigma": (2.5, 5.0),
    "eps_k": (100.0, 500.0),
}


def smiles_to_graph(smiles: str) -> Data:
    """Convert a polymer repeat-unit SMILES string into a PyTorch Geometric graph."""
    mol = polymer_smiles_to_mol(smiles)
    edges, attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges += [[i, j], [j, i]]
        attrs += [bf, bf]
    return Data(
        x=torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float64),
        edge_index=(
            torch.tensor(edges, dtype=torch.long).T.contiguous()
            if edges
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        edge_attr=(
            torch.tensor(attrs, dtype=torch.float64)
            if attrs
            else torch.zeros((0, BOND_FDIM), dtype=torch.float64)
        ),
    )


class AttentionMPNNLayer(MessagePassing):
    """Attention-weighted message-passing layer over directed bond edges."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__(aggr="add")
        self.msg_nn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
        )
        self.attn_nn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=torch.float64),
        )
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        inp = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_nn(inp) * torch.sigmoid(self.attn_nn(inp))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.update_nn(torch.cat([x, aggr_out], dim=-1))


class PolySAFTEncoder(nn.Module):
    """Four-layer attention MPNN with residual LayerNorm and concatenated mean-max readout."""

    def __init__(
        self,
        z_dim: int = 96,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(ATOM_FDIM, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
        )
        self.mp_layers = nn.ModuleList(
            [AttentionMPNNLayer(hidden_dim, BOND_FDIM, hidden_dim) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, dtype=torch.float64) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, z_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim, dtype=torch.float64),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        x = self.node_encoder(data.x)
        for layer, norm in zip(self.mp_layers, self.layer_norms):
            x = norm(x + self.dropout(layer(x, data.edge_index, data.edge_attr)))
        batch = (
            data.batch
            if getattr(data, "batch", None) is not None
            else torch.zeros(len(x), dtype=torch.long, device=x.device)
        )
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.readout(pooled)


class ParameterHead(nn.Module):
    """MLP mapping molecular graph embeddings to bounded PC-SAFT parameters (m/M, sigma, eps/k_B)."""

    def __init__(self, z_dim: int = 96, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, dtype=torch.float64),
        )

    def forward(self, z_mol: Tensor) -> dict[str, Tensor]:
        raw = self.mlp(z_mol)
        m_lo, m_hi = PARAMETER_BOUNDS["m_per_mw"]
        sig_lo, sig_hi = PARAMETER_BOUNDS["sigma"]
        eps_lo, eps_hi = PARAMETER_BOUNDS["eps_k"]
        return {
            "m_per_mw": m_lo + (m_hi - m_lo) * torch.sigmoid(raw[:, 0]),
            "sigma": sig_lo + (sig_hi - sig_lo) * torch.sigmoid(raw[:, 1]),
            "eps_k": eps_lo + (eps_hi - eps_lo) * torch.sigmoid(raw[:, 2]),
        }


class PolySAFTModel(nn.Module):
    """Structure-to-parameter PolySAFT neural network."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PolySAFTEncoder()
        self.param_head = ParameterHead()

    def forward(self, data: Data | Batch) -> dict[str, Tensor]:
        return self.param_head(self.encoder(data))
