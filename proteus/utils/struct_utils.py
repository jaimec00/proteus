import torch
import torch.nn.functional as F

from proteus.types import Tuple, Float, Int, Bool, T


def normalize_vec(vec: Float[T, "..."]) -> Float[T, "..."]:
    """normalize vectors along last dim"""
    return F.normalize(vec, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def get_backbone(C: Float[T, "BL 14 3"]) -> Float[T, "BL 4 3"]:
    """extract N, CA, C and compute virtual CB from full coords"""
    n = C[:, 0, :]
    ca = C[:, 1, :]
    c = C[:, 2, :]

    b1 = ca - n
    b2 = c - ca
    b3 = torch.linalg.cross(b1, b2, dim=-1)

    cb = ca - 0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

    return torch.stack([n, ca, c, cb], dim=1)

@torch.no_grad()
def get_CA_raw_and_CB_unit(C: Float[T, "BL 14 3"]) -> Float[T, "BL 4 3"]:
    C_backbone = get_backbone(C)
    CA, CB = C_backbone[:, 1, :], C_backbone[:, 3, :]
    CACB = CB - CA
    CACB_unit = normalize_vec(CACB)
    return CA, CACB_unit