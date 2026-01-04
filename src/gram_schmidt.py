# src/gram_schmidt.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GSResult:
    """Result of projecting a vector onto an orthonormal Gram–Schmidt basis."""
    coeffs: np.ndarray          # (k_eff,)
    basis: np.ndarray           # (D, k_eff) orthonormal columns
    reconstruction: np.ndarray  # (D,)


def gram_schmidt_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Classical Gram–Schmidt on columns of A.

    Args:
        A: (D, K) matrix whose columns are candidate basis vectors
        eps: threshold for dropping near-zero / dependent vectors

    Returns:
        Q: (D, k_eff) matrix with orthonormal columns.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    D, K = A.shape
    Q_cols: list[np.ndarray] = []

    for j in range(K):
        v = A[:, j].copy()
        for q in Q_cols:
            v -= np.dot(q, v) * q
        n = np.linalg.norm(v)
        if n > eps:
            Q_cols.append(v / n)

    if not Q_cols:
        raise ValueError(
            "Gram–Schmidt produced no basis vectors (columns were all zero/dependent)."
        )

    Q = np.stack(Q_cols, axis=1)  # (D, k_eff)
    return Q


def build_poly_basis(D: int, k: int) -> np.ndarray:
    """
    Build a (D, k) polynomial basis over bin index, with x mapped to [-1, 1].

    Column j is x**j for j=0..k-1 (constant, linear, quadratic, ...)

    Args:
        D: residual dimensionality (len(residuals_flat))
        k: requested number of basis vectors

    Returns:
        B: (D, k) candidate basis matrix
    """
    if D <= 0:
        raise ValueError("D must be positive.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > D:
        raise ValueError(f"k={k} cannot exceed D={D}.")

    x = np.linspace(-1.0, 1.0, D, dtype=float)
    B = np.vstack([x**j for j in range(k)]).T  # (D, k)
    return B


def project_vector_onto_gs_basis(
    v: np.ndarray,
    *,
    k: int,
    basis_type: str = "poly",
    eps: float = 1e-12,
) -> GSResult:
    """
    Build a candidate basis, Gram–Schmidt orthonormalize it, and project v.

    Args:
        v: (D,) vector to project
        k: requested number of dimensions to keep (upper bound; can drop dependent cols)
        basis_type: currently supports "poly"
        eps: GS stability threshold

    Returns:
        GSResult with coeffs, orthonormal basis Q, reconstruction Q@coeffs
    """
    v = np.asarray(v, dtype=float).ravel()
    D = v.size

    if k > D:
        raise ValueError(f"k={k} cannot exceed D={D}.")

    if basis_type == "poly":
        B = build_poly_basis(D, k)
    else:
        raise ValueError(f"Unknown basis_type={basis_type!r}. Supported: 'poly'.")

    Q = gram_schmidt_columns(B, eps=eps)  # (D, k_eff)
    coeffs = Q.T @ v                      # (k_eff,)
    recon = Q @ coeffs                    # (D,)

    return GSResult(coeffs=coeffs, basis=Q, reconstruction=recon)
