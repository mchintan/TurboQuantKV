"""Lloyd-Max optimal scalar quantization codebook for Beta-distributed coordinates."""

import math

import torch


def _beta_pdf(t: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Unnormalized Beta PDF: t^(alpha-1) * (1-t)^(beta-1) on [0, 1]."""
    return t.clamp(min=1e-30).pow(alpha - 1) * (1 - t).clamp(min=1e-30).pow(beta - 1)


def _symmetric_beta_pdf(x: torch.Tensor, alpha: float, beta_param: float) -> torch.Tensor:
    """PDF of the symmetric distribution on [-1, 1].

    After WHT on a unit vector in R^d, each coordinate follows a distribution
    where x^2 ~ Beta(0.5, (d-1)/2). The marginal distribution of x itself
    is symmetric around 0 with PDF proportional to (1 - x^2)^((d-3)/2)
    for d >= 3 (since alpha=0.5, the |x|^(2*0.5-1) = |x|^0 = 1 term vanishes).
    """
    # PDF proportional to (1 - x^2)^((beta_param - 1)) where beta_param = (d-1)/2
    # The exponent is (d-3)/2 = beta_param - 1
    return (1 - x * x).clamp(min=1e-30).pow(beta_param - 1)


def compute_lloyd_max_centroids(
    n_bits: int,
    block_size: int,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    num_integration_points: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal Lloyd-Max quantization centroids for the post-WHT distribution.

    After applying WHT to a unit vector in R^block_size, each coordinate
    follows a symmetric distribution on [-1, 1] with known PDF.

    Args:
        n_bits: Number of quantization bits (2, 3, or 4).
        block_size: WHT block size (determines distribution shape).
        max_iterations: Max Lloyd-Max iterations.
        tolerance: Convergence threshold on centroid movement.
        num_integration_points: Grid density for numerical integration.

    Returns:
        centroids: (2^n_bits,) tensor of reconstruction levels, sorted ascending.
        boundaries: (2^n_bits + 1,) tensor of decision boundaries including -1 and 1.
    """
    n_levels = 2 ** n_bits
    beta_param = (block_size - 1) / 2.0

    # Integration grid on [-1, 1]
    x = torch.linspace(-1 + 1e-6, 1 - 1e-6, num_integration_points, dtype=torch.float64)
    dx = x[1] - x[0]
    pdf = _symmetric_beta_pdf(x, 0.5, beta_param).to(torch.float64)
    # Normalize
    pdf = pdf / (pdf.sum() * dx)

    # Initialize centroids uniformly
    centroids = torch.linspace(-0.9, 0.9, n_levels, dtype=torch.float64)

    for _ in range(max_iterations):
        # Step 1: Compute boundaries as midpoints between adjacent centroids
        boundaries = torch.empty(n_levels + 1, dtype=torch.float64)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        # Step 2: Compute new centroids as conditional expectations
        new_centroids = torch.zeros(n_levels, dtype=torch.float64)
        for i in range(n_levels):
            mask = (x >= boundaries[i]) & (x < boundaries[i + 1])
            if i == n_levels - 1:
                mask = (x >= boundaries[i]) & (x <= boundaries[i + 1])

            weighted_x = (x * pdf * mask.float()).sum() * dx
            weight = (pdf * mask.float()).sum() * dx

            if weight > 1e-15:
                new_centroids[i] = weighted_x / weight
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        delta = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if delta < tolerance:
            break

    # Final boundaries
    boundaries = torch.empty(n_levels + 1, dtype=torch.float64)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(1, n_levels):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids.float(), boundaries.float()


class CodebookManager:
    """Lazy-computes and caches Lloyd-Max centroids for (n_bits, block_size) pairs."""

    def __init__(self):
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(
        self, n_bits: int, block_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get centroids and boundaries, computing if not cached.

        Returns:
            (centroids, boundaries) on the requested device.
        """
        key = (n_bits, block_size)
        if key not in self._cache:
            centroids, boundaries = compute_lloyd_max_centroids(n_bits, block_size)
            self._cache[key] = (centroids, boundaries)

        centroids, boundaries = self._cache[key]
        return centroids.to(device), boundaries.to(device)


# Global singleton
_codebook_manager = CodebookManager()


def get_codebook(n_bits: int, block_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Get centroids and boundaries from the global cache."""
    return _codebook_manager.get(n_bits, block_size, device)
